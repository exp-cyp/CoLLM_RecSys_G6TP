"""Download public assets without executing code or loading remote pickle files."""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path

from .common import ROOT, load_config, sha256, write_json

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATASET_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
MODEL_MANIFEST = "collm_source.json"
MODEL_METADATA_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
}


def verify_dataset_archive(path):
    """Reject altered/incomplete copies of the pinned official MovieLens archive."""
    actual = sha256(path)
    if actual != DATASET_SHA256:
        raise ValueError(
            f"MovieLens archive checksum mismatch: {path}. "
            f"Expected {DATASET_SHA256}, got {actual}. "
            "Restore the official archive before preparing data."
        )
    return actual


def _model_files(directory):
    directory = Path(directory)
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and (path.name in MODEL_METADATA_FILES or path.suffix == ".safetensors")
    )
    names = {path.name for path in files}
    if "config.json" not in names or "tokenizer_config.json" not in names:
        raise ValueError(f"Missing model/tokenizer configuration in {directory}")
    if not ({"tokenizer.json", "tokenizer.model"} & names or {"vocab.json", "merges.txt"} <= names):
        raise ValueError(f"Missing tokenizer files in {directory}")
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"Missing safetensors model weights in {directory}")
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        missing = set(index["weight_map"].values()) - names
        if missing:
            raise ValueError(f"Missing model weight shards in {directory}: {sorted(missing)}")
    return files


def _check_identity(model_id, revision):
    if not isinstance(model_id, str) or "/" not in model_id:
        raise ValueError("model.id must identify the Hugging Face model repository")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("model.revision must be a pinned 40-character commit SHA")


def record_local_model_source(directory, model_id, revision):
    """Record hashes after a download of the specified pinned model revision."""
    _check_identity(model_id, revision)
    directory = Path(directory)
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "model_revision": revision,
        "files": {path.name: sha256(path) for path in _model_files(directory)},
    }
    write_json(directory / MODEL_MANIFEST, manifest)
    return manifest


def verify_local_model(directory, model_id, revision):
    """Verify pinned identity and all local model inputs without any network calls."""
    _check_identity(model_id, revision)
    directory = Path(directory)
    manifest_path = directory / MODEL_MANIFEST
    if not manifest_path.exists():
        raise ValueError(
            f"Missing model source manifest: {manifest_path}; prepare pinned assets first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported model source manifest: {manifest_path}")
    if (manifest.get("model_id"), manifest.get("model_revision")) != (model_id, revision):
        raise ValueError("Local model identity/revision differs from the configured pinned model")
    files = _model_files(directory)
    expected = manifest.get("files")
    if not isinstance(expected, dict) or set(expected) != {path.name for path in files}:
        raise ValueError("Local model file list differs from its source manifest")
    for path in files:
        if sha256(path) != expected[path.name]:
            raise ValueError(f"Local model checksum mismatch: {path.name}")
    return manifest


def download_assets(config_path="configs/local.yaml"):
    cfg = load_config(config_path)
    model_id = cfg["model"]["id"]
    revision = cfg["model"]["revision"]
    _check_identity(model_id, revision)
    local_dir = (ROOT / cfg["model"]["path"]).resolve()
    if not local_dir.is_relative_to(ROOT.resolve()) or local_dir == ROOT.resolve():
        raise ValueError("model.path must be a dedicated directory inside this project")
    raw = ROOT / "data/raw/ml-1m.zip"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        import requests

        print("Downloading official MovieLens-1M...", flush=True)
        partial = raw.with_suffix(".partial")
        with requests.get(DATASET_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
        verify_dataset_archive(partial)
        partial.replace(raw)
    archive_hash = verify_dataset_archive(raw)
    if (local_dir / MODEL_MANIFEST).exists():
        verify_local_model(local_dir, model_id, revision)
    else:
        from huggingface_hub import snapshot_download

        print(f"Preparing pinned model {model_id}@{revision}...", flush=True)
        snapshot_download(
            model_id,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "tokenizer.model",
                "merges.txt",
                "README.md",
                "LICENSE*",
            ],
            max_workers=2,
        )
        record_local_model_source(local_dir, model_id, revision)
    provenance = {
        "dataset_url": DATASET_URL,
        "dataset_sha256": archive_hash,
        "model_id": model_id,
        "model_revision": revision,
        "model_path": str(local_dir.relative_to(ROOT)),
        "upstream_repository": "https://github.com/zyang1580/CoLLM",
        "upstream_commit": "df1a24dcc083ce9568565b72def58c7af80cd8c9",
    }
    write_json(ROOT / "results/provenance.json", provenance)
    print("Assets ready:", provenance, flush=True)


def probe():
    import psutil
    import torch

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "available_ram_gib": round(psutil.virtual_memory().available / 2**30, 2),
        "xpu_available": torch.xpu.is_available(),
        "devices": {},
    }
    devices = ["cpu"] + (["xpu"] if info["xpu_available"] else [])
    for device in devices:
        try:
            a = torch.randn(128, 128, device=device, requires_grad=True)
            started = time.perf_counter()
            (a @ a.T).square().mean().backward()
            if device == "xpu":
                torch.xpu.synchronize()
            info["devices"][device] = {
                "backward_ok": bool(torch.isfinite(a.grad).all()),
                "seconds_including_first_kernel": time.perf_counter() - started,
                "name": torch.xpu.get_device_name() if device == "xpu" else "CPU",
            }
        except Exception as error:
            info["devices"][device] = {"error": repr(error)}
    write_json(ROOT / "results/hardware.json", info)
    print(info, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "probe"])
    parser.add_argument("--config", default="configs/local.yaml")
    args = parser.parse_args()
    download_assets(args.config) if args.command == "download" else probe()
