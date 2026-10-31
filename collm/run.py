"""Run each stage in a fresh process, freeing LLM memory between stages."""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import torch
import yaml

from .checkpoints import run_context
from .common import ROOT, load_config, project_path
from .data import experiment_data


def preflight(cfg, allow_existing=False):
    """Check files, selected data, device, and output location without training."""
    rows = experiment_data(cfg)
    context = run_context(cfg, rows)
    if cfg["device"] == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is unavailable; select device: cpu or fix the environment")
    root = project_path(cfg["output_dir"], "outputs")
    if root.exists() and any(root.iterdir()):
        if not allow_existing:
            raise FileExistsError(f"Full pipeline needs a fresh output_dir: {root}")
        from .common import read_json

        if read_json(root / "run.json").get("contract_sha256") != context["contract_sha256"]:
            raise ValueError(
                "Existing output_dir belongs to a different model/data/training contract"
            )
    from transformers import AutoTokenizer

    from .models import PromptDataset

    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["model"]["path"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer.encode("Yes", add_special_tokens=False)) != 1:
        raise ValueError("The scoring method requires Yes to be one token")
    # Tokenization allocates no model weights and performs no optimizer updates.
    PromptDataset(rows["train"], tokenizer, cfg["model"]["max_length"], collaborative=True)
    print(
        "Preflight passed: pinned model/data verified; prompts, device, and output_dir are ready.",
        flush=True,
    )
    print(
        {split: values["count"] for split, values in context["data_manifest"].items()}, flush=True
    )
    return context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument(
        "--check", action="store_true", help="Validate readiness only; do not train"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check:
        preflight(cfg)
        return 0
    from .pipeline import run_job

    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S")
    directory = ROOT / ".cache/jobs" / f"cli_{stamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    metadata = run_job(directory)
    return (
        0
        if metadata["status"] in {"complete", "time_limit"}
        else 130
        if metadata["status"] == "stopped"
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
