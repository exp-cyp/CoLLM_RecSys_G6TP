from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))


def write_json(path, value):
    """Replace JSON atomically, so interruption cannot leave half a metadata file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def project_path(value, folder):
    """Resolve a configured path inside a named project directory."""
    path = (ROOT / value).resolve()
    allowed = (ROOT / folder).resolve()
    try:
        relative = path.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Path must stay inside {allowed}: {value}") from error
    if not relative.parts:
        raise ValueError(f"Choose a subdirectory inside {folder}, not {folder} itself")
    return path


def results_path(cfg):
    relative = project_path(cfg["output_dir"], "outputs").relative_to(ROOT / "outputs")
    return ROOT / "results" / relative


def validate_config(cfg):
    """Fail before model loading when an experiment configuration is invalid."""
    import re

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    required = {
        "seed",
        "run_kind",
        "device",
        "threads",
        "budget_hours",
        "output_dir",
        "model",
        "data",
        "mf",
        "stage1",
        "stage2",
    }
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    if cfg["device"] not in {"cpu", "xpu"}:
        raise ValueError("device must be cpu or xpu")
    if cfg["run_kind"] not in {"smoke", "experiment"}:
        raise ValueError("run_kind must be smoke or experiment")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")

    def positive_int(value, name):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    def positive_number(value, name, allow_zero=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")

    positive_int(cfg["threads"], "threads")
    positive_number(cfg["budget_hours"], "budget_hours")
    if cfg["budget_hours"] > 6:
        raise ValueError("budget_hours cannot exceed the agreed six-hour limit")
    cfg["output_dir"] = project_path(cfg["output_dir"], "outputs").relative_to(ROOT).as_posix()
    model = cfg["model"]
    for key in ("id", "revision", "path", "max_length"):
        if key not in model:
            raise ValueError(f"Missing model.{key}")
    if not isinstance(model["id"], str) or "/" not in model["id"]:
        raise ValueError("model.id must be a Hugging Face model ID")
    if not isinstance(model["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", model["revision"]
    ):
        raise ValueError("model.revision must be a pinned 40-character commit SHA")
    model["path"] = project_path(model["path"], "models").relative_to(ROOT).as_posix()
    positive_int(model["max_length"], "model.max_length")
    for split in ("train", "valid", "test"):
        value = cfg["data"].get(f"{split}_limit")
        if value is not None:
            positive_int(value, f"data.{split}_limit")
    mf = cfg["mf"]
    for key in ("embedding_dim", "epochs", "patience", "batch_size"):
        positive_int(mf[key], f"mf.{key}")
    if not isinstance(mf["learning_rates"], list) or not mf["learning_rates"]:
        raise ValueError("mf.learning_rates must be a non-empty list")
    for lr in mf["learning_rates"]:
        positive_number(lr, "mf.learning_rates")
    positive_number(mf["weight_decay"], "mf.weight_decay", allow_zero=True)
    for stage in ("stage1", "stage2"):
        spec = cfg[stage]
        for key in ("batch_size", "accumulation", "max_steps", "eval_every", "eval_batch_size"):
            positive_int(spec[key], f"{stage}.{key}")
        positive_number(spec["learning_rate"], f"{stage}.learning_rate")
        positive_number(spec["weight_decay"], f"{stage}.weight_decay", allow_zero=True)


def seed_all(seed, threads=6):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync(device):
    if str(device).startswith("xpu"):
        torch.xpu.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


class StopRequested(KeyboardInterrupt):
    """A user stop request handled by the normal checkpoint-saving path."""


class Budget:
    """A wall-clock budget; callers check between batches and save completed work."""

    def __init__(self, hours):
        self.start = time.monotonic()
        self.seconds = float(hours) * 3600
        stop_file = os.environ.get("COLLM_STOP_FILE")
        self.stop_file = Path(stop_file) if stop_file else None

    def check_stop(self):
        if self.stop_file is not None and self.stop_file.exists():
            raise StopRequested("User requested a safe stop")

    @property
    def elapsed(self):
        return time.monotonic() - self.start

    @property
    def remaining(self):
        self.check_stop()
        return max(0.0, self.seconds - self.elapsed)

    def expired(self, reserve=0):
        self.check_stop()
        return self.elapsed >= max(0, self.seconds - reserve)
