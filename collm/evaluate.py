"""Compare held-out scores and show candidate ranking under one fixed history."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter

import pandas as pd
import torch

from .checkpoints import load_checkpoint, run_context, stage_path
from .common import (
    Budget,
    load_config,
    project_path,
    read_json,
    results_path,
    seed_all,
    write_json,
)
from .data import experiment_data
from .models import build_model
from .train import auc, llm_scores, load_mf, make_loader, mf_scores


def group_auc(rows, scores, warm):
    indices = [index for index, row in enumerate(rows) if row["warm"] == warm]
    chosen = [rows[index] for index in indices]
    if len({row["label"] for row in chosen}) != 2:
        return None
    return auc(chosen, scores[indices])


def ranking_candidates(rows):
    counts = Counter(row["uid"] for row in rows)
    user = max(counts, key=counts.get)
    candidates = sorted(
        [row for row in rows if row["uid"] == user], key=lambda row: row["timestamp"]
    )
    context = candidates[0]
    seen = set()
    result = []
    for row in candidates:
        if row["iid"] in seen:
            continue
        seen.add(row["iid"])
        candidate = dict(row)
        for field in ("history_titles", "history_timestamps", "timestamp"):
            candidate[field] = context[field]
        result.append(candidate)
        if len(result) == 20:
            break
    return result


def evaluate(cfg, output_dir=None):
    seed_all(cfg["seed"], cfg["threads"])
    budget = Budget(cfg["budget_hours"])
    rows = experiment_data(cfg)
    context = run_context(cfg, rows)
    mf, mf_metadata = load_mf(cfg, rows, context)
    adapter_path, adapter_metadata = load_checkpoint(cfg, "stage1", context, parents={})
    parents = {"mf": mf_metadata["artifact_sha256"], "stage1": adapter_metadata["artifact_sha256"]}
    mapper_path, mapper_metadata = load_checkpoint(cfg, "stage2", context, parents)
    root = project_path(output_dir, "results") if output_dir else results_path(cfg)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Results already exist at {root}; choose --results-dir for a new evaluation"
        )
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "status.json", {"status": "running"})
    try:
        candidates = ranking_candidates(rows["test"])
        predictions = {"MF": mf_scores(mf, rows["test"])}
        ranking = {"MF": mf_scores(mf, candidates)}
        details = {
            "MF": {
                "checkpoint_step": mf_metadata["step"],
                "best_valid_auc": mf_metadata["valid_auc"],
            }
        }
        selected = {"stage1": adapter_metadata, "stage2": mapper_metadata}
        for stage, name in (("stage1", "TALLRec"), ("stage2", "CoLLM-MF")):
            model, tokenizer = build_model(cfg, mf, stage, adapter_path=adapter_path)
            try:
                if stage == "stage2":
                    model.mapper.load_state_dict(
                        torch.load(mapper_path / "mapper.pt", map_location="cpu", weights_only=True)
                    )
                predictions[name] = llm_scores(
                    model, make_loader(rows["test"], tokenizer, cfg, stage), budget
                )
                ranking[name] = llm_scores(
                    model, make_loader(candidates, tokenizer, cfg, stage), budget
                )
                summary = read_json(stage_path(cfg, stage) / "summary.json")
                details[name] = {
                    "checkpoint_step": selected[stage]["step"],
                    "best_valid_auc": selected[stage]["valid_auc"],
                    "training_steps_completed": summary["steps_completed"],
                    "training_status": summary["status"],
                }
                print(f"{name}: test AUC={auc(rows['test'], predictions[name]):.5f}", flush=True)
            finally:
                del model, tokenizer
                mf.cpu()
                gc.collect()
                if cfg["device"] == "xpu":
                    torch.xpu.empty_cache()
        results = {
            "run_kind": cfg["run_kind"],
            "model": cfg["model"]["id"],
            "model_revision": cfg["model"]["revision"],
            "device": cfg["device"],
            "config": cfg,
            "data_manifest": context["data_manifest"],
            "parents": parents,
            "contract_sha256": context["contract_sha256"],
            "elapsed_seconds": budget.elapsed,
            "evaluation_protocol": "ROC AUC over observed held-out ratings; not full-catalog top-k evaluation",
            "score": "raw Yes logit, matching upstream Qwen BCE; no vocabulary softmax",
            "test_used_for_selection": False,
            "metrics": {
                name: {
                    "auc": auc(rows["test"], scores),
                    "warm_auc": group_auc(rows["test"], scores, True),
                    "cold_auc": group_auc(rows["test"], scores, False),
                    **details[name],
                }
                for name, scores in predictions.items()
            },
        }
        write_json(root / "metrics.json", results)
        frame = pd.DataFrame(
            [
                {key: row[key] for key in ("record_id", "uid", "iid", "label", "warm", "timestamp")}
                for row in rows["test"]
            ]
        )
        for name, scores in predictions.items():
            frame[name] = scores
        frame.to_csv(root / "predictions.csv", index=False)
        demo = pd.DataFrame(
            [
                {key: row[key] for key in ("uid", "raw_uid", "iid", "title", "timestamp")}
                for row in candidates
            ]
        )
        for name, scores in ranking.items():
            demo[name] = scores
            demo[name + "_rank"] = (
                pd.Series(scores).rank(ascending=False, method="first").astype(int)
            )
        demo.sort_values("CoLLM-MF_rank").to_csv(root / "ranking_example.csv", index=False)
        write_json(
            root / "ranking_context.json",
            {
                "uid": candidates[0]["uid"],
                "timestamp": candidates[0]["timestamp"],
                "history_titles": candidates[0]["history_titles"],
                "candidate_set": "up to 20 observed test movies for this user; fixed historical context",
            },
        )
        report(results, root)
        write_json(root / "status.json", {"status": "complete", "elapsed_seconds": budget.elapsed})
        print(json.dumps(results["metrics"], indent=2), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        write_json(
            root / "status.json",
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error": str(error),
                "elapsed_seconds": budget.elapsed,
            },
        )
        raise


def report(results, root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results["metrics"])
    scores = [results["metrics"][name]["auc"] for name in names]
    smoke = results["run_kind"] == "smoke"
    title = (
        "SMOKE TEST ONLY - not research results"
        if smoke
        else "MovieLens-1M: resource-constrained experiment"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, scores, color=["#64748b", "#3b82f6", "#0d9488"])
    ax.set(ylim=(0, 1), ylabel="ROC AUC", title=title)
    ax.axhline(0.5, linestyle="--", color="#94a3b8", linewidth=1)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    fig.tight_layout()
    fig.savefig(root / "auc_comparison.png", dpi=160)
    plt.close(fig)
    counts = {split: value["count"] for split, value in results["data_manifest"].items()}
    cfg = results["config"]
    lines = [
        "# "
        + ("Smoke test only - not research results" if smoke else "Measured experiment results"),
        "",
    ]
    if smoke:
        lines += [
            "**Pipeline verification only. Do not use these scores as term-project findings.**",
            "",
        ]
    lines += [
        f"Backbone: `{results['model']}` at `{results['model_revision']}`. Device: `{results['device']}`.",
        "",
        f"Records: train={counts['train']}, validation={counts['valid']}, test={counts['test']}.",
        "",
        "| Model | Test AUC | Warm AUC | Cold AUC | Selected step |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in results["metrics"].items():
        cells = [
            f"{values[key]:.4f}" if values[key] is not None else "N/A"
            for key in ("auc", "warm_auc", "cold_auc")
        ]
        lines.append(f"| {name} | {' | '.join(cells)} | {values['checkpoint_step']} |")
    lines += [
        "",
        "All models use identical record selections, including full input/label content fingerprints.",
        "Checkpoints are selected by validation AUC; test data is not used for tuning.",
        "MF, Stage 1, and Stage 2 weights and their exact dependencies were verified before loading.",
        "",
        "## Actual training",
        "",
    ]
    for name in ("TALLRec", "CoLLM-MF"):
        values = results["metrics"][name]
        lines.append(
            f"- {name}: {values['training_steps_completed']} optimizer steps; status `{values['training_status']}`."
        )
    lines += [
        "",
        "## Scope and limitations",
        "",
        "This implements the core method with resource adjustments; it is not an exact reproduction of the paper's tables.",
        f"MF dimension: {cfg['mf']['embedding_dim']}; maximum prompt length: {cfg['model']['max_length']} tokens.",
        "History contains only strictly earlier positive observations. Earlier held-out feedback may appear in later prompts under the documented rolling-history protocol.",
        "Unseen IDs have zero MF features before the mapper, whose biases may produce a constant nonzero feature.",
        "AUC is measured on observed held-out ratings; the fixed-context ranking CSV is an illustration, not full-catalog evaluation.",
        "",
        "## Config and data identity",
        "",
        "```json",
        json.dumps({"config": cfg, "data_manifest": results["data_manifest"]}, indent=2),
        "```",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument(
        "--results-dir", help="New directory under results/; existing results are never overwritten"
    )
    args = parser.parse_args()
    evaluate(load_config(args.config), args.results_dir)
