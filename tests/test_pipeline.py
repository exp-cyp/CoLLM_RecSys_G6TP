"""Exercise orchestration with fake child processes; never start model training."""

from __future__ import annotations

import io
import json
import subprocess
from contextlib import nullcontext

import pytest
import yaml

from collm import checkpoints, common, pipeline
from collm.common import load_config, read_json, write_json


@pytest.fixture
def job(tmp_path, monkeypatch):
    cfg = load_config("configs/smoke.yaml")
    cfg["output_dir"] = "outputs/new_job"
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(checkpoints, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "ProjectLock", lambda: nullcontext())
    monkeypatch.setattr(pipeline, "KeepAwake", lambda: nullcontext())
    context = {"schema_version": 2, "contract_sha256": "matching", "config": cfg}
    monkeypatch.setattr(pipeline, "preflight", lambda config: context)
    directory = tmp_path / ".cache/jobs/test_job"
    directory.mkdir(parents=True)
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    calls = []

    def outcome(config, stage, current_context, require_complete=False):
        return {
            "status": "complete",
            "steps_completed": 2,
            "artifact_sha256": stage,
            "checkpoint_step": 1,
            "best_valid_auc": 0.6,
            "reused": False,
        }

    monkeypatch.setattr(pipeline, "stage_outcome", outcome)

    def runner(command, env, on_line):
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert env["COLLM_STOP_FILE"] == str(directory / "STOP")
        module = command[3]
        stage = command[4] if module == "collm.train" else "evaluate"
        calls.append(stage)
        if stage in ("stage1", "stage2"):
            on_line(json.dumps({"stage": stage, "step": 1}))
        else:
            on_line(f"running {stage}")
        if stage == "evaluate":
            write_json(tmp_path / "results/new_job/status.json", {"status": "complete"})

    def finalizer(config, directory, metadata):
        calls.append("finalize")
        return {
            "results_dir": "results/new_job",
            "source_zip": "submission/test.zip",
            "completion_status": "complete",
        }

    return cfg, directory, context, calls, runner, finalizer


def test_one_job_runs_in_order_and_persists_events(job, capsys):
    cfg, directory, _, calls, runner, finalizer = job
    result = pipeline.run_job(directory, runner=runner, finalizer=finalizer)
    assert calls == ["mf", "stage1", "stage2", "evaluate", "finalize"]
    assert result["status"] == "complete"
    assert result["source_zip"] == "submission/test.zip"
    assert read_json(directory / "job.json") == result
    assert "COLLM_EVENT " in (directory / "pipeline.log").read_text(encoding="utf-8")
    assert "COLLM_EVENT " in capsys.readouterr().out


def test_real_job_requires_published_git_before_training(job):
    cfg, directory, _, calls, runner, finalizer = job
    cfg["run_kind"] = "experiment"
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def unpublished():
        raise RuntimeError("Publish this commit first")

    result = pipeline.run_job(
        directory, runner=runner, finalizer=finalizer, published_check=unpublished
    )
    assert result["status"] == "failed"
    assert "Publish this commit" in result["error"]
    assert calls == []


def test_real_job_propagates_verified_commit(job):
    cfg, directory, _, _, runner, finalizer = job
    cfg["run_kind"] = "experiment"
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def guarded_runner(command, env, on_line):
        assert env["COLLM_GIT_COMMIT"] == "verified"
        runner(command, env, on_line)

    result = pipeline.run_job(
        directory,
        runner=guarded_runner,
        finalizer=finalizer,
        published_check=lambda: {"commit": "verified", "branch": "main", "upstream": "origin/main"},
    )
    assert result["status"] == "complete"


def test_stop_before_start_never_runs_a_child(job):
    _, directory, _, calls, runner, finalizer = job
    (directory / "STOP").touch()
    result = pipeline.run_job(directory, runner=runner, finalizer=finalizer)
    assert result["status"] == "stopped"
    assert calls == []


def test_stop_after_finished_stage_preserves_completed_status(job):
    _, directory, _, calls, runner, finalizer = job

    def stop_after_mf(command, env, on_line):
        runner(command, env, on_line)
        (directory / "STOP").touch()

    result = pipeline.run_job(directory, runner=stop_after_mf, finalizer=finalizer)
    assert result["status"] == "stopped"
    assert result["stages"]["mf"]["status"] == "complete"
    assert calls == ["mf"]


def test_child_failure_stops_later_stages_and_packaging(job):
    _, directory, _, calls, runner, finalizer = job

    def fail_stage1(command, env, on_line):
        if command[3:5] == ["collm.train", "stage1"]:
            raise subprocess.CalledProcessError(1, command)
        runner(command, env, on_line)

    result = pipeline.run_job(directory, runner=fail_stage1, finalizer=finalizer)
    assert result["status"] == "failed"
    assert result["stages"]["stage1"]["status"] == "failed"
    assert calls == ["mf"]


def test_time_limited_stage_is_not_labeled_fully_complete(job, monkeypatch):
    _, directory, _, _, runner, finalizer = job
    original = pipeline.stage_outcome

    def partial(config, stage, context, require_complete=False):
        result = original(config, stage, context, require_complete)
        if stage == "stage1":
            result["status"] = "time_limit"
        return result

    monkeypatch.setattr(pipeline, "stage_outcome", partial)
    result = pipeline.run_job(directory, runner=runner, finalizer=finalizer)
    assert result["status"] == "time_limit"


def test_duplicate_job_does_not_overwrite_finished_metadata(job):
    _, directory, _, _, runner, finalizer = job
    first = pipeline.run_job(directory, runner=runner, finalizer=finalizer)
    with pytest.raises(FileExistsError, match="already started"):
        pipeline.run_job(directory, runner=runner, finalizer=finalizer)
    assert read_json(directory / "job.json") == first


def test_concurrent_duplicate_cannot_overwrite_active_job(job, monkeypatch):
    _, directory, _, _, runner, finalizer = job
    active = {"status": "running", "started_at": "already-running"}

    class LostRace:
        def __enter__(self):
            write_json(directory / "job.json", active)
            raise pipeline.JobAlreadyRunning("Other process owns the project lock")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(pipeline, "ProjectLock", LostRace)
    with pytest.raises(pipeline.JobAlreadyRunning):
        pipeline.run_job(directory, runner=runner, finalizer=finalizer)
    assert read_json(directory / "job.json") == active


def test_project_lock_rejects_concurrent_owner_and_releases(tmp_path):
    path = tmp_path / "train.lock"
    with pipeline.ProjectLock(path):
        with pytest.raises(pipeline.JobAlreadyRunning):
            with pipeline.ProjectLock(path):
                pass
    with pipeline.ProjectLock(path):
        pass


def make_completed_stage(cfg, stage, context, parents):
    root = checkpoints.stage_path(cfg, stage)
    directory = root / "checkpoints/selected"
    directory.mkdir(parents=True)
    for name in checkpoints.ARTIFACT_FILES[stage]:
        (directory / name).write_bytes(f"{stage}:{name}".encode())
    selected = checkpoints.select_checkpoint(root, stage, directory, context, parents, 2, 0.6)
    write_json(root / "summary.json", {"status": "complete", "steps_completed": 2})
    write_json(root / "status.json", {"status": "complete"})
    return selected


def test_reuse_only_copies_verified_completed_prefix(job, monkeypatch):
    cfg, _, context, _, _, _ = job
    monkeypatch.setattr(pipeline, "stage_outcome", pipeline_stage_outcome)
    old_cfg = {**cfg, "output_dir": "outputs/old_job"}
    old_root = common.project_path(old_cfg["output_dir"], "outputs")
    old_root.mkdir(parents=True)
    write_json(old_root / "run.json", context)
    make_completed_stage(old_cfg, "mf", context, {})
    write_json(old_root / "stage1/status.json", {"status": "interrupted"})
    reused = pipeline.reuse_completed_stages(cfg, "outputs/old_job", context)
    assert list(reused) == ["mf"]
    assert reused["mf"]["reused"] is True
    assert (common.project_path(cfg["output_dir"], "outputs") / "mf/best.json").exists()
    assert not (common.project_path(cfg["output_dir"], "outputs") / "stage1").exists()


def test_reuse_rejects_changed_contract_or_parent(job, monkeypatch):
    cfg, _, context, _, _, _ = job
    monkeypatch.setattr(pipeline, "stage_outcome", pipeline_stage_outcome)
    old_cfg = {**cfg, "output_dir": "outputs/old_job"}
    old_root = common.project_path(old_cfg["output_dir"], "outputs")
    old_root.mkdir(parents=True)
    write_json(old_root / "run.json", context)
    make_completed_stage(old_cfg, "mf", context, {})
    adapter = make_completed_stage(old_cfg, "stage1", context, {})
    make_completed_stage(
        old_cfg, "stage2", context, {"mf": "wrong", "stage1": adapter["artifact_sha256"]}
    )
    with pytest.raises(ValueError, match="parents"):
        pipeline.reuse_completed_stages(cfg, "outputs/old_job", context)


def test_stream_child_merges_utf8_output_without_new_console(tmp_path, monkeypatch):
    captured = {}

    class Child:
        stdout = io.StringIO("first line\nsecond line\n")

        def wait(self):
            return 0

    def popen(command, **kwargs):
        captured.update(kwargs)
        return Child()

    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    lines = []
    pipeline.stream_child(["fake"], {"COLLM_STOP_FILE": str(tmp_path / "STOP")}, lines.append)
    assert lines == ["first line", "second line"]
    assert captured["stderr"] == subprocess.STDOUT
    assert captured["encoding"] == "utf-8"
    if pipeline.os.name == "nt":
        assert captured["creationflags"] == subprocess.CREATE_NO_WINDOW


def test_changed_reuse_contract_is_rejected_before_copy(job):
    cfg, _, context, _, _, _ = job
    source = common.project_path("outputs/old_job", "outputs")
    source.mkdir(parents=True)
    write_json(source / "run.json", {"contract_sha256": "different"})
    with pytest.raises(ValueError, match="different model, data"):
        pipeline.reuse_completed_stages(cfg, "outputs/old_job", context)
    assert not common.project_path(cfg["output_dir"], "outputs").exists()


# Save the production helper before fixtures monkeypatch pipeline.stage_outcome.
pipeline_stage_outcome = pipeline.stage_outcome
