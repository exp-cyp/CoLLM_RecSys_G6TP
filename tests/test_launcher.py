"""Launcher tests use fake child processes; they never open a UI or train."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

from collm import launcher
from collm.launcher import (
    JobController,
    TrainWindow,
    artifact_paths,
    create_job,
    load_settings,
    progress_text,
    reusable_job,
    scoped_path,
    settings_signature,
    write_json,
)


@pytest.fixture
def settings(tmp_path):
    cfg = {
        "seed": 42,
        "run_kind": "experiment",
        "device": "xpu",
        "threads": 6,
        "budget_hours": 6,
        "output_dir": "outputs/local",
        "model": {
            "id": "Qwen/Qwen2-0.5B",
            "revision": "a" * 40,
            "path": "models/Qwen2-0.5B",
            "max_length": 256,
        },
        "data": {"train_limit": 4096, "valid_limit": 512, "test_limit": 1024},
    }
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, path


def test_job_snapshot_is_unique_and_does_not_create_output_or_results(tmp_path, settings):
    cfg, _ = settings
    original = deepcopy(cfg)
    first, first_cfg = create_job(cfg, tmp_path)
    second, second_cfg = create_job(cfg, tmp_path)
    assert first != second
    assert first_cfg["output_dir"] != second_cfg["output_dir"]
    assert load_settings(first / "config.yaml") == first_cfg
    assert cfg == original
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "results").exists()
    pointer = json.loads((tmp_path / ".cache/jobs/latest.json").read_text(encoding="utf-8"))
    assert pointer["job_dir"] == second.relative_to(tmp_path).as_posix()


def test_settings_compatibility_ignores_only_output_location(settings):
    cfg, _ = settings
    other = deepcopy(cfg)
    other["output_dir"] = "outputs/another"
    assert settings_signature(cfg) == settings_signature(other)
    other["data"]["train_limit"] = 2048
    assert settings_signature(cfg) != settings_signature(other)


def test_reuse_requires_compatible_unfinished_job_and_existing_output(tmp_path, settings):
    cfg, _ = settings
    directory, snapshot = create_job(cfg, tmp_path)
    write_json(directory / "job.json", {"status": "stopped"})
    assert reusable_job(cfg, tmp_path) is None
    (tmp_path / snapshot["output_dir"]).mkdir(parents=True)
    assert reusable_job(cfg, tmp_path) is None
    write_json(tmp_path / snapshot["output_dir"] / "run.json", {"schema_version": 2})
    assert reusable_job(cfg, tmp_path)["output_dir"] == snapshot["output_dir"]
    write_json(directory / "job.json", {"status": "complete"})
    assert reusable_job(cfg, tmp_path) is None


def test_malicious_latest_pointer_cannot_escape_jobs(tmp_path, settings):
    cfg, _ = settings
    write_json(tmp_path / ".cache/jobs/latest.json", {"job_dir": "../outside"})
    assert reusable_job(cfg, tmp_path) is None
    with pytest.raises(ValueError):
        scoped_path(tmp_path.parent, tmp_path)


def test_readiness_checks_git_and_preflight_without_pipeline(tmp_path, settings):
    _, config_path = settings
    calls = []
    events = []

    def runner(command, on_line, cwd):
        calls.append(command)
        return 0

    controller = JobController(lambda kind, value: events.append((kind, value)), tmp_path, runner)
    controller.check_ready(config_path)
    assert [call[3] for call in calls] == ["collm.git_state", "collm.run"]
    assert all("--check" in call for call in calls)
    assert not (tmp_path / ".cache/jobs").exists()


def test_failed_git_gate_does_not_create_job_or_start_pipeline(tmp_path, settings):
    _, config_path = settings
    events = []
    calls = []

    def runner(command, on_line, cwd):
        calls.append(command)
        return 1

    controller = JobController(lambda kind, value: events.append((kind, value)), tmp_path, runner)
    controller.run_training(config_path)
    assert len(calls) == 1
    assert calls[0][3] == "collm.git_state"
    assert not (tmp_path / ".cache/jobs").exists()
    assert events[-1][0] == "error"


def test_final_click_rechecks_and_then_starts_only_snapshot_job(tmp_path, settings):
    _, config_path = settings
    calls = []
    events = []

    def runner(command, on_line, cwd):
        calls.append(command)
        if command[3] == "collm.pipeline":
            directory = tmp_path / command[command.index("--job-dir") + 1]
            write_json(
                directory / "job.json", {"status": "complete", "completion_status": "complete"}
            )
            on_line('COLLM_EVENT {"status":"running","stage":"stage1"}')
        return 0

    controller = JobController(lambda kind, value: events.append((kind, value)), tmp_path, runner)
    controller.run_training(config_path)
    assert [call[3] for call in calls] == ["collm.git_state", "collm.run", "collm.pipeline"]
    assert events[-1] == ("finished", {"status": "complete", "completion_status": "complete"})
    assert any(kind == "progress" and value["stage"] == "stage1" for kind, value in events)


def test_continue_passes_previous_output_to_new_pipeline_job(tmp_path, settings):
    cfg, config_path = settings
    previous, snapshot = create_job(cfg, tmp_path)
    write_json(previous / "job.json", {"status": "stopped"})
    output = tmp_path / snapshot["output_dir"]
    output.mkdir(parents=True)
    write_json(output / "run.json", {"schema_version": 2})
    calls = []

    def runner(command, on_line, cwd):
        calls.append(command)
        if command[3] == "collm.pipeline":
            current = tmp_path / command[command.index("--job-dir") + 1]
            write_json(current / "job.json", {"status": "complete"})
        return 0

    controller = JobController(lambda *args: None, tmp_path, runner)
    controller.run_training(config_path)
    command = calls[-1]
    assert command[command.index("--reuse-from") + 1] == snapshot["output_dir"]
    assert controller.job_dir != previous
    assert load_settings(previous / "config.yaml") == snapshot


def test_safe_stop_writes_request_file_without_killing_process(tmp_path, settings):
    cfg, _ = settings
    controller = JobController(lambda *args: None, tmp_path)
    controller.job_dir, _ = create_job(cfg, tmp_path)
    assert controller.request_stop()
    assert (controller.job_dir / "STOP").read_text(encoding="utf-8").startswith("User requested")


def test_stop_during_readiness_cancels_before_pipeline(tmp_path, settings):
    _, config_path = settings
    events = []
    calls = []
    controller = None

    def runner(command, on_line, cwd):
        calls.append(command)
        controller.request_stop()
        return 0

    controller = JobController(lambda kind, value: events.append((kind, value)), tmp_path, runner)
    controller.run_training(config_path)
    assert all(call[3] != "collm.pipeline" for call in calls)
    assert not (tmp_path / ".cache/jobs").exists()
    assert events[-1][0] == "finished" and events[-1][1]["status"] == "stopped"


def test_controller_rejects_second_concurrent_start(tmp_path, settings):
    _, config_path = settings
    events = []
    controller = JobController(
        lambda kind, value: events.append((kind, value)),
        tmp_path,
        lambda *args: pytest.fail("No command should run"),
    )
    controller._lock.acquire()
    try:
        controller.run_training(config_path)
    finally:
        controller._lock.release()
    assert events[-1][0] == "error"


def test_progress_uses_current_stage_and_nested_step_counts():
    text = progress_text(
        {
            "status": "running",
            "current_stage": "stage2",
            "stages": {"stage2": {"step": 25, "max_steps": 512}},
        }
    )
    assert "CoLLM-MF" in text
    assert "25/512" in text
    assert progress_text({"status": "time_limit"}) is None


def test_artifact_paths_prefer_human_report_and_explicit_zip(tmp_path, settings):
    cfg, _ = settings
    directory = tmp_path / "results/local"
    directory.mkdir(parents=True)
    report = directory / "report.html"
    report.write_text("<p>Report</p>", encoding="utf-8")
    archive = tmp_path / "submission/result.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"fixture archive")
    state = {
        "status": "time_limit",
        "results_dir": "results/local",
        "source_zip": "submission/result.zip",
    }
    assert artifact_paths(state, cfg, tmp_path) == (report, archive)
    # Never show an unrelated stale source archive when this job did not supply one.
    assert artifact_paths({"status": "complete"}, cfg, tmp_path) == (report, None)


def test_time_limited_finished_job_enables_artifacts_without_opening_ui(
    tmp_path, settings, monkeypatch
):
    cfg, _ = settings
    directory, _ = create_job(cfg, tmp_path)
    result_dir = tmp_path / "results/current"
    result_dir.mkdir(parents=True)
    (result_dir / "report.html").write_text("report", encoding="utf-8")
    archive = tmp_path / "submission/current.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"archive")

    class Widget:
        def __init__(self):
            self.options = {}

        def configure(self, **options):
            self.options.update(options)

    class Variable:
        def set(self, value):
            self.value = value

    window = TrainWindow.__new__(TrainWindow)
    for name in ("stop_button", "check_button", "train_button", "results_button", "zip_button"):
        setattr(window, name, Widget())
    window.status_text = Variable()
    window.controller = SimpleNamespace(job_dir=directory)
    window.ready = True
    window.preview = False
    window.close_after_stop = False
    window.refresh_settings = lambda: None
    window.append_log = lambda text: None
    monkeypatch.setattr(
        launcher,
        "artifact_paths",
        lambda state, selected: artifact_paths(state, selected, tmp_path),
    )
    window.finish(
        {
            "status": "time_limit",
            "results_dir": "results/current",
            "source_zip": "submission/current.zip",
        }
    )
    assert "บางส่วน" in window.status_text.value
    assert window.results_button.options["state"] == "normal"
    assert window.zip_button.options["state"] == "normal"
    assert window.result_path == result_dir / "report.html"
