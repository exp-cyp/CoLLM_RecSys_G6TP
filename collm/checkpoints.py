"""Checkpoint identity, parent binding, and safe stage initialization.

Each selected checkpoint gets its own directory. Only best.json is replaced, after
all weights are written and hashed. Incomplete checkpoint writes are never selected.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

from .assets import verify_local_model
from .common import ROOT, project_path, read_json, sha256, write_json
from .data import manifest

SCHEMA_VERSION = 2
ARTIFACT_FILES = {
    "mf": ("model.pt",),
    "stage1": ("adapter_config.json", "adapter_model.safetensors"),
    "stage2": ("mapper.pt",),
}


def json_digest(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def run_context(cfg, rows):
    source = verify_local_model(
        ROOT / cfg["model"]["path"], cfg["model"]["id"], cfg["model"]["revision"]
    )
    # Runtime/output settings do not change checkpoint compatibility. Training and
    # prompt settings, selected data content, and model bytes do.
    semantic_config = {key: cfg[key] for key in ("seed", "model", "data", "mf", "stage1", "stage2")}
    semantic_config["model"] = {key: value for key, value in cfg["model"].items() if key != "path"}
    implementation = {}
    for filename in ("models.py", "data.py", "train.py"):
        # AST ignores comments/formatting but detects changed model/training logic.
        tree = ast.parse((ROOT / "collm" / filename).read_text(encoding="utf-8"))
        implementation[filename] = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    contract = {
        "config": semantic_config,
        "data": manifest(rows),
        "model_source": source,
        "implementation": implementation,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": json_digest(contract),
        "data_manifest": contract["data"],
        "model_source": source,
        "implementation": implementation,
        "config": cfg,
        "git_commit": os.environ.get("COLLM_GIT_COMMIT"),
    }


def stage_path(cfg, stage):
    if stage not in ARTIFACT_FILES:
        raise ValueError(f"Unknown stage: {stage}")
    return project_path(cfg["output_dir"], "outputs") / stage


def prepare_stage(cfg, stage, context, parents):
    root = project_path(cfg["output_dir"], "outputs")
    run_file = root / "run.json"
    if run_file.exists():
        previous = read_json(run_file)
        if previous.get("contract_sha256") != context["contract_sha256"]:
            raise ValueError(
                "This output_dir belongs to a different experiment. Choose a new output_dir."
            )
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "Existing output_dir has no compatible run manifest. Choose a new output_dir."
        )
    path = stage_path(cfg, stage)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite {path}. Use a fresh output_dir for a new experiment."
        )
    path.mkdir(parents=True, exist_ok=True)
    write_json(run_file, context)
    write_json(path / "status.json", {"stage": stage, "status": "running", "parents": parents})
    return path


def artifact_digest(stage, directory):
    return json_digest({name: sha256(Path(directory) / name) for name in ARTIFACT_FILES[stage]})


def select_checkpoint(root, stage, directory, context, parents, step, valid_auc):
    metadata = {
        **context,
        "stage": stage,
        "parents": parents,
        "step": step,
        "valid_auc": valid_auc,
        "checkpoint": Path(directory).relative_to(root).as_posix(),
        "artifact_sha256": artifact_digest(stage, directory),
    }
    write_json(Path(root) / "best.json", metadata)
    return metadata


def load_checkpoint(cfg, stage, context, parents=None):
    root = stage_path(cfg, stage)
    metadata = read_json(root / "best.json")
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("stage") != stage:
        raise ValueError(f"Unsupported {stage} checkpoint; use checkpoints from the current code")
    if metadata.get("contract_sha256") != context["contract_sha256"]:
        raise ValueError(
            f"{stage} checkpoint model/config/data content does not match this experiment"
        )
    if parents is not None and metadata.get("parents") != parents:
        raise ValueError(f"{stage} checkpoint has different MF/LoRA parents")
    directory = (root / metadata["checkpoint"]).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("Checkpoint path escapes its stage directory")
    if artifact_digest(stage, directory) != metadata["artifact_sha256"]:
        raise ValueError(f"{stage} checkpoint files have changed since selection")
    return directory, metadata
