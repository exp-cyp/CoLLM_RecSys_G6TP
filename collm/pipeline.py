"""Run one explicit Train job with a project lock, logs, reuse, and delivery."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from .checkpoints import load_checkpoint, stage_path
from .common import ROOT, load_config, project_path, read_json, results_path, write_json
from .run import preflight

STAGES = ("mf", "stage1", "stage2")
EVENT_PREFIX = "COLLM_EVENT "


def now_iso():
    return datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")


class JobAlreadyRunning(RuntimeError):
    pass


class JobAlreadyStarted(FileExistsError):
    pass


class ProjectLock:
    """Nonblocking OS lock; process exit releases it even after a crash."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else ROOT / ".cache/train.lock"
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise JobAlreadyRunning("Another training job is running in this project") from error
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is None:
            return
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None


class KeepAwake:
    """Temporarily prevent idle system sleep; never change display/lid policy."""

    def __init__(self):
        self.active = False

    def __enter__(self):
        if os.name == "nt":
            import ctypes

            self.active = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def check_stop(stop_file):
    if Path(stop_file).exists():
        raise KeyboardInterrupt("Stop requested; completed work has been preserved")


def expected_parents(cfg, stage, context):
    if stage != "stage2":
        return {}
    _, mf = load_checkpoint(cfg, "mf", context, parents={})
    _, adapter = load_checkpoint(cfg, "stage1", context, parents={})
    return {"mf": mf["artifact_sha256"], "stage1": adapter["artifact_sha256"]}


def stage_outcome(cfg, stage, context, require_complete=False):
    """A return code alone is insufficient: verify weights and persisted status."""
    parents = expected_parents(cfg, stage, context)
    _, selected = load_checkpoint(cfg, stage, context, parents)
    root = stage_path(cfg, stage)
    summary = read_json(root / "summary.json")
    status = read_json(root / "status.json").get("status")
    accepted = {"complete"} if require_complete else {"complete", "time_limit"}
    if status not in accepted or summary.get("status") != status:
        raise ValueError(f"{stage} has no usable completed status: {status}")
    return {
        "status": status,
        "steps_completed": summary["steps_completed"],
        "checkpoint_step": selected["step"],
        "best_valid_auc": selected["valid_auc"],
        "artifact_sha256": selected["artifact_sha256"],
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "reused": False,
    }


def reuse_completed_stages(cfg, source_output_dir, context):
    """Copy a verified completed prefix; restart the first incomplete stage fresh.

    This reuses finished stages, not an optimizer/sampler state within a stage.
    Copied summaries retain the original training settings and provenance.
    """
    source_root = project_path(source_output_dir, "outputs")
    target_root = project_path(cfg["output_dir"], "outputs")
    if source_root == target_root:
        raise ValueError("Continuation must use a new output_dir")
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError("Reuse destination must be a fresh output_dir")
    previous = read_json(source_root / "run.json")
    if previous.get("contract_sha256") != context["contract_sha256"]:
        raise ValueError("Previous job has a different model, data, or training contract")
    source_cfg = {**cfg, "output_dir": source_root.relative_to(ROOT).as_posix()}
    reused = {}
    for stage in STAGES:
        source = stage_path(source_cfg, stage)
        if not (source / "status.json").exists():
            break
        if read_json(source / "status.json").get("status") != "complete":
            break
        details = stage_outcome(source_cfg, stage, context, require_complete=True)
        for item in [source, *source.rglob("*")]:
            if not item.resolve().is_relative_to(source_root):
                raise ValueError("Cannot copy a stage containing links outside its source run")
        target_root.mkdir(parents=True, exist_ok=True)
        write_json(target_root / "run.json", context)
        destination = stage_path(cfg, stage)
        temporary = target_root / f".{stage}.copying-{uuid4().hex}"
        shutil.copytree(source, temporary)
        if not temporary.resolve().is_relative_to(
            target_root
        ) or not destination.resolve().is_relative_to(target_root):
            raise ValueError("Stage copy escaped its destination run")
        temporary.replace(destination)
        # Check the copied files too, including exact parents for Stage 2.
        stage_outcome(cfg, stage, context, require_complete=True)
        reused[stage] = {
            **details,
            "reused": True,
            "source_output_dir": source_root.relative_to(ROOT).as_posix(),
        }
    return reused


def stream_child(command, env, on_line):
    """Forward merged output without killing the process on a Stop request."""
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            on_line(line.rstrip("\r\n"))
        return_code = process.wait()
    except (Exception, KeyboardInterrupt):
        # Ctrl+C or a closed GUI/log pipe must still wait for child snapshots.
        Path(env["COLLM_STOP_FILE"]).touch()
        if process.stdout is not None:
            for line in process.stdout:
                try:
                    on_line(line.rstrip("\r\n"))
                except (OSError, ValueError):
                    pass
        process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
    if return_code:
        check_stop(env["COLLM_STOP_FILE"])
        raise subprocess.CalledProcessError(return_code, command)


def run_job(job_dir, reuse_from=None, *, runner=None, finalizer=None, published_check=None):
    """Execute one user-created job; return its final persisted metadata."""
    directory = project_path(job_dir, ".cache/jobs")
    config_path = directory / "config.yaml"
    cfg = load_config(config_path)
    if reuse_from is not None:
        reuse_from = project_path(reuse_from, "outputs").relative_to(ROOT).as_posix()
    metadata_path = directory / "job.json"
    if metadata_path.exists() and read_json(metadata_path).get("started_at"):
        raise JobAlreadyStarted(
            "This job has already started; create a new job to continue or retry"
        )
    directory.mkdir(parents=True, exist_ok=True)
    stop_file = directory / "STOP"
    started = time.monotonic()
    metadata = {
        "job_id": directory.name,
        "status": "preparing",
        "started_at": now_iso(),
        "current_stage": "preflight",
        "config": cfg,
        "output_dir": cfg["output_dir"],
        "results_dir": results_path(cfg).relative_to(ROOT).as_posix(),
        "source_zip": None,
        "stages": {stage: {"status": "pending", "reused": False} for stage in STAGES},
        "error": None,
        "reuse_from": reuse_from,
    }
    runner = runner or stream_child
    env = os.environ.copy()
    env.update(
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        COLLM_STOP_FILE=str(stop_file),
    )

    with (directory / "pipeline.log").open("a", encoding="utf-8") as log:

        def message(text):
            log.write(text + "\n")
            log.flush()
            print(text, flush=True)

        def publish(event):
            metadata["elapsed_seconds"] = time.monotonic() - started
            write_json(metadata_path, metadata)
            message(EVENT_PREFIX + json.dumps({"event": event, **metadata}, ensure_ascii=False))

        def child_line(line):
            message(line)
            try:
                progress = json.loads(line)
            except (ValueError, TypeError):
                return
            stage = metadata["current_stage"]
            if (
                stage in ("stage1", "stage2")
                and isinstance(progress, dict)
                and progress.get("stage") == stage
                and isinstance(progress.get("step"), int)
            ):
                metadata["stages"][stage].update(
                    step=progress["step"], max_steps=cfg[stage]["max_steps"]
                )
                publish("progress")

        try:
            with ProjectLock(), KeepAwake():
                # Recheck under the lock in case two processes opened one job
                # before either process published its first status.
                if metadata_path.exists() and read_json(metadata_path).get("started_at"):
                    raise JobAlreadyStarted(
                        "This job has already started; create a new job to continue or retry"
                    )
                check_stop(stop_file)
                if cfg["run_kind"] == "experiment":
                    if published_check is None:
                        from .git_state import require_published

                        published_check = require_published
                    metadata["git"] = published_check()
                    env["COLLM_GIT_COMMIT"] = metadata["git"]["commit"]
                publish("preparing")
                context = preflight(cfg)
                if reuse_from is not None:
                    reused = reuse_completed_stages(cfg, reuse_from, context)
                    metadata["stages"].update(reused)
                    publish("reused")
                for stage in STAGES:
                    check_stop(stop_file)
                    if metadata["stages"][stage].get("reused"):
                        continue
                    metadata.update(status="running", current_stage=stage)
                    metadata["stages"][stage]["status"] = "running"
                    publish("stage_started")
                    runner(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "collm.train",
                            stage,
                            "--config",
                            str(config_path),
                        ],
                        env,
                        child_line,
                    )
                    metadata["stages"][stage] = stage_outcome(cfg, stage, context)
                    publish("stage_finished")
                check_stop(stop_file)
                metadata.update(status="evaluating", current_stage="evaluate")
                publish("evaluation_started")
                runner(
                    [sys.executable, "-u", "-m", "collm.evaluate", "--config", str(config_path)],
                    env,
                    child_line,
                )
                check_stop(stop_file)
                if read_json(results_path(cfg) / "status.json").get("status") != "complete":
                    raise RuntimeError("Evaluation did not produce a complete result set")
                metadata.update(status="finalizing", current_stage="finalize")
                publish("finalizing")
                if finalizer is None:
                    from .delivery import finalize_run

                    finalizer = finalize_run
                metadata.update(finalizer(cfg, directory, metadata))
                if metadata.get("completion_status") not in {
                    "complete",
                    "time_limit",
                } or not metadata.get("source_zip"):
                    raise RuntimeError(
                        "Finalization did not produce the promised result/source archive metadata"
                    )
                partial = any(
                    value["status"] == "time_limit" for value in metadata["stages"].values()
                )
                metadata["status"] = (
                    "time_limit"
                    if partial or metadata.get("completion_status") == "time_limit"
                    else "complete"
                )
                metadata.update(current_stage=None, finished_at=now_iso())
                publish("finished")
        except JobAlreadyStarted:
            raise
        except JobAlreadyRunning as error:
            # A duplicate invocation of the same active job must not overwrite
            # job.json while its owner is still publishing progress.
            if metadata_path.exists() and read_json(metadata_path).get("started_at"):
                raise
            metadata.update(status="failed", error=str(error), finished_at=now_iso())
            publish("failed")
        except KeyboardInterrupt as error:
            stage = metadata["current_stage"]
            if stage in STAGES and metadata["stages"][stage]["status"] == "running":
                metadata["stages"][stage]["status"] = "stopped"
            metadata.update(
                status="stopped", error=str(error) or "Stopped by user", finished_at=now_iso()
            )
            publish("stopped")
        except Exception as error:
            stage = metadata["current_stage"]
            if stage in STAGES and metadata["stages"][stage]["status"] == "running":
                metadata["stages"][stage]["status"] = "failed"
            metadata.update(
                status="failed", error=f"{type(error).__name__}: {error}", finished_at=now_iso()
            )
            publish("failed")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir", required=True, help="Prepared job directory under .cache/jobs/"
    )
    parser.add_argument(
        "--reuse-from", help="Verified completed stages from a previous outputs/ run"
    )
    args = parser.parse_args()
    metadata = run_job(args.job_dir, args.reuse_from)
    return (
        0
        if metadata["status"] in {"complete", "time_limit"}
        else 130
        if metadata["status"] == "stopped"
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
