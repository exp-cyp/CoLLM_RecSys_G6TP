"""Desktop interface for training and evaluating CoLLM experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import threading
import tkinter as tk
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/local.yaml"
LATEST_JOB = "latest.json"
ACTIVE_STATES = {"preparing", "running", "evaluating", "finalizing"}
REUSABLE_STATES = ACTIVE_STATES | {"stopped", "failed", "interrupted", "time_limit"}
STAGE_NAMES = {
    "mf": "ฝึก MF — ขั้นที่ 1/4",
    "stage1": "ฝึก TALLRec (LoRA) — ขั้นที่ 2/4",
    "stage2": "ฝึก CoLLM-MF (MLP) — ขั้นที่ 3/4",
    "evaluate": "ประเมินผล — ขั้นที่ 4/4",
    "evaluation": "ประเมินผล — ขั้นที่ 4/4",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_settings(path):
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("ไฟล์ตั้งค่าต้องเป็น YAML mapping")
    return cfg


def settings_signature(cfg):
    """Output location changes for every click; the experiment settings do not."""
    settings = {key: value for key, value in cfg.items() if key != "output_dir"}
    canonical = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def scoped_path(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("ตำแหน่งไฟล์อยู่นอกโฟลเดอร์ที่กำหนด")
    return path


def reusable_job(cfg, root=ROOT):
    """Find the latest compatible unfinished job without executing anything."""
    jobs = Path(root) / ".cache/jobs"
    try:
        pointer = read_json(jobs / LATEST_JOB)
        directory = scoped_path(Path(root) / pointer["job_dir"], jobs)
        previous = load_settings(directory / "config.yaml")
        state = read_json(directory / "job.json")
        if state.get("status") not in REUSABLE_STATES:
            return None
        if settings_signature(previous) != settings_signature(cfg):
            return None
        output = scoped_path(Path(root) / previous["output_dir"], Path(root) / "outputs")
        if not (output / "run.json").is_file():
            return None
        return {
            "job_dir": directory,
            "output_dir": previous["output_dir"],
            "status": state["status"],
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def create_job(cfg, root=ROOT):
    """Create only an ignored job snapshot; outputs/results remain untouched."""
    import yaml

    root = Path(root).resolve()
    stamp = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y%m%d_%H%M%S")
    name = f"run_{stamp}_{uuid4().hex[:8]}"
    directory = root / ".cache/jobs" / name
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = deepcopy(cfg)
    snapshot["output_dir"] = f"outputs/{name}"
    (directory / "config.yaml").write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    write_json(
        root / ".cache/jobs" / LATEST_JOB, {"job_dir": directory.relative_to(root).as_posix()}
    )
    return directory, snapshot


def progress_text(state):
    """Translate pipeline job/event metadata into a short user-facing status."""
    status = state.get("status", "")
    stage = state.get("current_stage") or state.get("stage") or ""
    stage = stage if isinstance(stage, str) else ""
    if not status and stage:
        status = "running"
    if status not in ACTIVE_STATES:
        return None
    text = STAGE_NAMES.get(
        stage,
        {
            "preparing": "กำลังตรวจและเตรียมรอบทดลอง",
            "evaluating": "กำลังประเมินผล — ขั้นที่ 4/4",
            "finalizing": "กำลังสร้างรายงานและ ZIP",
        }.get(status, "กำลังทำงาน"),
    )
    stages = state.get("stages")
    details = stages.get(stage, {}) if isinstance(stages, dict) else state
    if isinstance(details, dict):
        step, maximum = details.get("step"), details.get("max_steps")
        if isinstance(step, int) and isinstance(maximum, int) and maximum > 0:
            text += f" — อัปเดต {step}/{maximum}"
    return text


def artifact_paths(state, cfg, root=ROOT):
    """Prefer the human HTML report and only expose existing in-project artifacts."""
    root = Path(root).resolve()
    delivery = state.get("delivery") if isinstance(state.get("delivery"), dict) else {}

    def safe(value):
        try:
            path = scoped_path(root / value, root)
            return path if path.exists() else None
        except (TypeError, ValueError):
            return None

    default = "results/" + Path(cfg["output_dir"]).relative_to("outputs").as_posix()
    directory = safe(state.get("results_dir") or delivery.get("results_dir") or default)
    report_path = directory
    if directory is not None and directory.is_dir() and (directory / "report.html").is_file():
        report_path = directory / "report.html"
    archive = safe(state.get("source_zip") or delivery.get("source_zip"))
    return report_path, archive


def stream_command(command, on_line, cwd):
    """Run without a console window and relay stdout/stderr to the GUI queue."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    with subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=flags,
        env=env,
    ) as process:
        for line in process.stdout:
            on_line(line.rstrip("\r\n"))
        return process.wait()


class JobController:
    """Worker-side orchestration; it never calls Tk widgets."""

    def __init__(self, emit, root=ROOT, command_runner=stream_command):
        self.emit = emit
        self.root = Path(root).resolve()
        self.command_runner = command_runner
        self.python = str(self.root / ".venv/Scripts/python.exe")
        self.job_dir = None
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()

    def prepare_start(self):
        self.job_dir = None
        self._stop_requested.clear()

    def _line(self, line):
        self.emit("log", line)
        if line.startswith("COLLM_EVENT "):
            try:
                event = json.loads(line.removeprefix("COLLM_EVENT "))
                if isinstance(event, dict):
                    self.emit("progress", event)
            except json.JSONDecodeError:
                pass

    def check_ready(self, config_path):
        commands = [
            [self.python, "-u", "-m", "collm.git_state", "--check"],
            [self.python, "-u", "-m", "collm.run", "--check", "--config", str(config_path)],
        ]
        for command in commands:
            if self.command_runner(command, self._line, self.root) != 0:
                raise RuntimeError("ตรวจความพร้อมไม่ผ่าน อ่านรายละเอียดในบันทึก แล้วกดตรวจอีกครั้ง")

    def run_training(self, config_path):
        if not self._lock.acquire(blocking=False):
            self.emit("error", "มีงานกำลังทำอยู่แล้ว")
            return
        try:
            self.emit("checking", None)
            cfg = load_settings(config_path)
            self.check_ready(config_path)  # Recheck on the final Train click.
            if self._stop_requested.is_set():
                self.emit("finished", {"status": "stopped", "message": "ยกเลิกก่อนเริ่มเทรน"})
                return
            previous = reusable_job(cfg, self.root)
            self.job_dir, _ = create_job(cfg, self.root)
            if self._stop_requested.is_set():
                self.request_stop()
                self.emit("finished", {"status": "stopped", "message": "ยกเลิกก่อนเริ่มเทรน"})
                return
            command = [
                self.python,
                "-u",
                "-m",
                "collm.pipeline",
                "--job-dir",
                self.job_dir.relative_to(self.root).as_posix(),
            ]
            if previous is not None:
                command += ["--reuse-from", previous["output_dir"]]
            self.emit("started", str(self.job_dir))
            code = self.command_runner(command, self._line, self.root)
            try:
                state = read_json(self.job_dir / "job.json")
            except (OSError, ValueError):
                state = {"status": "failed", "message": f"โปรแกรมจบด้วยรหัส {code}; อ่าน pipeline.log"}
            if code and state.get("status") == "complete":
                state = {**state, "status": "failed", "message": f"โปรแกรมจบด้วยรหัส {code}"}
            self.emit("finished", state)
        except Exception as error:
            self.emit("error", str(error))
        finally:
            self._lock.release()

    def request_stop(self):
        self._stop_requested.set()
        if self.job_dir is None:
            return True
        directory = scoped_path(self.job_dir, self.root / ".cache/jobs")
        (directory / "STOP").write_text("User requested a safe stop.\n", encoding="utf-8")
        return True


class TrainWindow:
    def __init__(self, window, config_path=DEFAULT_CONFIG, preview=False):
        self.window = window
        self.config_path = Path(config_path).resolve()
        self.preview = preview
        self.events = queue.Queue()
        self.controller = JobController(lambda kind, value: self.events.put((kind, value)))
        self.busy = False
        self.ready = False
        self.close_after_stop = False
        self.closed = False
        self.latest_state = {}
        self.result_path = None
        self.zip_path = None
        window.title("CoLLM — Train")
        window.geometry("940x720")
        window.minsize(760, 560)
        window.protocol("WM_DELETE_WINDOW", self.on_close)
        frame = ttk.Frame(window, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="CoLLM: การทดลอง", font=("Segoe UI", 19, "bold")).pack(anchor="w")
        ttk.Label(frame, text="ฝึกและประเมิน MF, TALLRec และ CoLLM-MF").pack(anchor="w", pady=(4, 12))
        self.settings_text = tk.StringVar()
        ttk.Label(frame, textvariable=self.settings_text, justify="left", wraplength=880).pack(
            anchor="w"
        )
        self.status_text = tk.StringVar(value="กำลังเริ่มต้นโปรแกรม")
        ttk.Label(
            frame, textvariable=self.status_text, font=("Segoe UI", 11, "bold"), wraplength=880
        ).pack(anchor="w", pady=(14, 4))
        self.reuse_text = tk.StringVar()
        ttk.Label(frame, textvariable=self.reuse_text, wraplength=880).pack(anchor="w", pady=(0, 8))
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(0, 10))
        self.train_button = ttk.Button(
            buttons, text="Train", command=self.on_train, state="disabled"
        )
        self.train_button.pack(side="left")
        self.check_button = ttk.Button(buttons, text="ตรวจอีกครั้ง", command=self.check_readiness)
        self.check_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(
            buttons, text="หยุดอย่างปลอดภัย", command=self.on_stop, state="disabled"
        )
        self.stop_button.pack(side="left")
        self.results_button = ttk.Button(
            buttons,
            text="เปิดผลลัพธ์",
            command=lambda: self.open_path(self.result_path),
            state="disabled",
        )
        self.results_button.pack(side="right")
        self.zip_button = ttk.Button(
            buttons, text="เปิด ZIP", command=lambda: self.open_path(self.zip_path), state="disabled"
        )
        self.zip_button.pack(side="right", padx=8)
        ttk.Label(frame, text="บันทึกการฝึกและประเมินผล").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(
            frame, height=18, wrap="word", state="disabled", font=("Consolas", 10)
        )
        self.log.pack(fill="both", expand=True, pady=(4, 0))
        self.refresh_settings()
        window.after(100, self.poll_events)
        if preview:
            self.status_text.set("โหมดแสดงตัวอย่าง — ปิดการฝึกและตรวจความพร้อม")
            self.check_button.configure(state="disabled")
        else:
            window.after(150, self.check_readiness)

    def refresh_settings(self):
        try:
            cfg = load_settings(self.config_path)
            data = cfg["data"]
            self.settings_text.set(
                f"โมเดล: {cfg['model']['id']}  |  อุปกรณ์: {cfg['device'].upper()}\n"
                f"ข้อมูล: train {data['train_limit']:,} / validation {data['valid_limit']:,} / test {data['test_limit']:,}\n"
                f"MF → TALLRec → CoLLM-MF → AUC / Ranking / รายงาน / ZIP\n"
                f"เวลาสูงสุด {cfg['budget_hours']:g} ชั่วโมงต่อขั้น"
            )
            previous = reusable_job(cfg)
            self.reuse_text.set(
                "พบการทดลองที่ยังไม่จบ: ใช้ checkpoint ของขั้นที่เสร็จแล้ว และเริ่มขั้นที่ค้างใหม่"
                if previous
                else "กด Train เพื่อเริ่มการทดลองใหม่ ผลลัพธ์แต่ละรอบเก็บแยกโฟลเดอร์"
            )
        except Exception as error:
            self.settings_text.set(f"อ่านไฟล์ตั้งค่าไม่ได้: {error}")

    def set_busy(self, value):
        self.busy = value
        self.check_button.configure(state="disabled" if value or self.preview else "normal")
        self.train_button.configure(
            state="normal" if self.ready and not value and not self.preview else "disabled"
        )

    def check_readiness(self):
        if self.busy or self.preview:
            return
        self.ready = False
        self.controller.prepare_start()
        self.set_busy(True)
        self.status_text.set("กำลังตรวจข้อมูล โมเดล และเวอร์ชันโค้ด")

        def worker():
            try:
                self.controller.check_ready(self.config_path)
                self.events.put(("ready", None))
            except Exception as error:
                self.events.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def on_train(self):
        if self.busy or not self.ready or self.preview:
            return
        self.set_busy(True)
        self.controller.prepare_start()
        self.status_text.set("กำลังตรวจซ้ำก่อนเริ่มเทรน")
        threading.Thread(
            target=self.controller.run_training, args=(self.config_path,), daemon=True
        ).start()

    def on_stop(self):
        if self.controller.request_stop():
            self.status_text.set("กำลังหยุดและบันทึก checkpoint")
            self.stop_button.configure(state="disabled")

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 1600:
            self.log.delete("1.0", "300.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def update_job(self, state):
        self.latest_state = state
        text = progress_text(state)
        if text is not None:
            self.status_text.set(text)

    def finish(self, state):
        self.latest_state = state
        self.stop_button.configure(state="disabled")
        self.set_busy(False)
        status = state.get("status", "failed")
        delivery = state.get("delivery") if isinstance(state.get("delivery"), dict) else {}
        completion = state.get("completion_status", delivery.get("completion_status"))
        success = status in {"complete", "time_limit"}
        if success:
            self.status_text.set(
                "ได้ผลลัพธ์บางส่วนตามเพดานเวลา — ตรวจสถานะในรายงาน"
                if status == "time_limit" or completion == "time_limit"
                else "การทดลองเสร็จสมบูรณ์ — เปิดผลลัพธ์ได้"
            )
        elif status in {"stopped", "interrupted"}:
            self.status_text.set("หยุดและบันทึกแล้ว — กด Train เพื่อใช้ขั้นที่เสร็จแล้วต่อ")
        else:
            self.status_text.set("การทดลองไม่สำเร็จ — ตรวจรายละเอียดในบันทึก")
            self.append_log(str(state.get("message") or state.get("error") or "ตรวจ pipeline.log"))
        if self.controller.job_dir is not None:
            cfg = load_settings(self.controller.job_dir / "config.yaml")
            self.result_path, self.zip_path = artifact_paths(state, cfg)
            self.results_button.configure(
                state="normal" if success and self.result_path else "disabled"
            )
            self.zip_button.configure(state="normal" if success and self.zip_path else "disabled")
        self.refresh_settings()
        if self.close_after_stop:
            self.destroy()

    def open_path(self, path):
        if path is None or not path.exists():
            messagebox.showerror("CoLLM", "ยังไม่พบไฟล์ผลลัพธ์")
        elif os.name == "nt":
            os.startfile(path)

    def poll_events(self):
        try:
            for _ in range(200):
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.append_log(value)
                elif kind == "ready":
                    if self.close_after_stop:
                        self.destroy()
                        return
                    self.ready = True
                    self.status_text.set("พร้อมเริ่มการทดลอง")
                    self.set_busy(False)
                elif kind == "checking":
                    self.status_text.set("กำลังตรวจซ้ำก่อนเริ่มเทรน")
                elif kind == "started":
                    self.stop_button.configure(state="normal")
                    self.append_log(f"โฟลเดอร์งาน: {value}")
                elif kind == "progress":
                    self.update_job(value)
                elif kind == "finished":
                    self.finish(value)
                elif kind == "error":
                    self.ready = False
                    self.set_busy(False)
                    self.stop_button.configure(state="disabled")
                    self.status_text.set("ตรวจความพร้อมไม่ผ่าน — ตรวจรายละเอียดในบันทึก")
                    self.append_log(value)
                    if self.close_after_stop:
                        self.destroy()
                if self.closed:
                    return
        except queue.Empty:
            pass
        if self.busy and self.controller.job_dir:
            try:
                self.update_job(read_json(self.controller.job_dir / "job.json"))
            except (OSError, ValueError):
                pass
        if not self.closed:
            self.window.after(150, self.poll_events)

    def destroy(self):
        self.closed = True
        self.window.destroy()

    def on_close(self):
        if not self.busy:
            self.destroy()
        elif messagebox.askyesno(
            "CoLLM", "ขอให้โปรแกรมหยุดอย่างปลอดภัย แล้วปิดหน้าต่างเมื่อเก็บงานเสร็จหรือไม่?"
        ):
            self.close_after_stop = True
            self.controller.request_stop()
            self.status_text.set("กำลังขอหยุดและเก็บงานก่อนปิด")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--preview", action="store_true", help="Show UI with Train disabled and no checks/jobs"
    )
    args = parser.parse_args()
    window = tk.Tk()
    TrainWindow(window, args.config, preview=args.preview)
    window.mainloop()


if __name__ == "__main__":
    main()
