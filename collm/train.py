"""MF pretraining, text-only LoRA tuning, mapping tuning, and short benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import psutil
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint, prepare_stage, run_context, select_checkpoint
from .common import ROOT, Budget, StopRequested, load_config, seed_all, sync, write_json
from .data import experiment_data
from .models import Collator, MatrixFactorization, PromptDataset, build_model


def require_training_source(cfg):
    if cfg["run_kind"] == "experiment":
        from .git_state import require_published

        published = require_published()
        os.environ["COLLM_GIT_COMMIT"] = published["commit"]


def move_batch(batch, device):
    """Keep ground-truth labels separate from model inputs."""
    features = {key: value.to(device) for key, value in batch.items() if key != "label"}
    return features, batch["label"].to(device)


def make_mf(cfg, rows):
    # IDs are mapped over the retained dataset; unseen IDs remain masked out.
    from .common import read_json

    metadata = read_json(ROOT / "data/processed/metadata.json")
    model = MatrixFactorization(
        metadata["num_users"], metadata["num_items"], cfg["mf"]["embedding_dim"]
    )
    model.set_seen(rows["train"])
    return model


def load_mf(cfg, rows, context):
    directory, metadata = load_checkpoint(cfg, "mf", context, parents={})
    model = make_mf(cfg, rows)
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    return model, metadata


def mf_scores(model, rows):
    device = next(model.parameters()).device
    with torch.no_grad():
        users = torch.tensor([row["uid"] for row in rows], device=device)
        items = torch.tensor([row["iid"] for row in rows], device=device)
        return model(users, items).float().cpu().numpy()


def auc(rows, scores):
    return float(roc_auc_score([row["label"] for row in rows], scores))


def checkpoint_directory(root, name):
    directory = root / "checkpoints" / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def save_weights(model, stage, directory):
    directory.mkdir(parents=True, exist_ok=True)
    if stage == "mf":
        torch.save(model.state_dict(), directory / "model.pt")
    elif stage == "stage1":
        model.lm.save_pretrained(directory, safe_serialization=True)
    else:
        torch.save(model.mapper.state_dict(), directory / "mapper.pt")


def train_mf(cfg):
    """Fit each learning-rate trial on CPU; select only by validation AUC."""
    require_training_source(cfg)
    budget = Budget(cfg["budget_hours"])
    seed_all(cfg["seed"], cfg["threads"])
    rows = experiment_data(cfg)
    context = run_context(cfg, rows)
    root = prepare_stage(cfg, "mf", context, parents={})
    spec = cfg["mf"]
    users = torch.tensor([row["uid"] for row in rows["train"]])
    items = torch.tensor([row["iid"] for row in rows["train"]])
    labels = torch.tensor([row["label"] for row in rows["train"]], dtype=torch.float32)
    best_auc = None
    completed = 0
    trials = []
    model = None
    status = "complete"
    failure = None
    current_trial_steps = 0
    try:
        for trial, learning_rate in enumerate(spec["learning_rates"], start=1):
            if budget.expired():
                status = "time_limit"
                break
            seed_all(cfg["seed"], cfg["threads"])
            model = make_mf(cfg, rows)
            current_trial_steps = 0
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=spec["weight_decay"]
            )
            trial_best, stale = None, 0
            epochs_completed = 0
            for epoch in range(1, spec["epochs"] + 1):
                model.train()
                for indices in torch.randperm(len(users)).split(spec["batch_size"]):
                    if budget.expired():
                        status = "time_limit"
                        break
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(users[indices], items[indices]), labels[indices]
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite MF loss")
                    loss.backward()
                    optimizer.step()
                    completed += 1
                    current_trial_steps += 1
                if status == "time_limit":
                    break
                epochs_completed = epoch
                model.eval()
                valid_auc = auc(rows["valid"], mf_scores(model, rows["valid"]))
                if best_auc is None or valid_auc > best_auc:
                    directory = checkpoint_directory(root, f"trial_{trial:02d}_epoch_{epoch:04d}")
                    save_weights(model, "mf", directory)
                    select_checkpoint(root, "mf", directory, context, {}, completed, valid_auc)
                    best_auc = valid_auc
                improved = trial_best is None or valid_auc > trial_best
                stale = 0 if improved else stale + 1
                trial_best = valid_auc if improved else trial_best
                if epoch == 1 or epoch % 10 == 0:
                    print(
                        f"MF lr={learning_rate} epoch={epoch} valid_auc={valid_auc:.5f}", flush=True
                    )
                if stale >= spec["patience"]:
                    break
            trials.append(
                {
                    "learning_rate": learning_rate,
                    "epochs_completed": epochs_completed,
                    "best_valid_auc": trial_best,
                }
            )
            if status == "time_limit":
                break
    except (Exception, KeyboardInterrupt) as error:
        failure = error
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    finally:
        save_error = None
        if model is not None and current_trial_steps:
            try:
                save_weights(model, "mf", root / "last")
            except Exception as error:
                save_error = error
                status = "failed" if failure is None else status
        write_json(
            root / "summary.json",
            {
                "status": status,
                "steps_completed": completed,
                "best_valid_auc": best_auc,
                "elapsed_seconds": budget.elapsed,
                "trials": trials,
                "config": cfg,
                "data_manifest": context["data_manifest"],
                "error": str(failure) if failure else None,
                "last_checkpoint_error": str(save_error) if save_error else None,
            },
        )
        write_json(root / "status.json", {"stage": "mf", "status": status})
    if failure:
        raise failure
    if save_error:
        raise save_error
    if best_auc is None:
        raise RuntimeError(
            "MF stopped before a checkpoint could be validated; choose a new output_dir and sufficient time"
        )


def make_loader(rows, tokenizer, cfg, stage, training=False):
    dataset = PromptDataset(
        rows, tokenizer, cfg["model"]["max_length"], collaborative=stage == "stage2"
    )
    spec = cfg[stage]
    return DataLoader(
        dataset,
        batch_size=spec["batch_size"] if training else spec["eval_batch_size"],
        shuffle=training,
        collate_fn=Collator(tokenizer.pad_token_id),
        num_workers=0,
    )


def llm_scores(model, loader, budget=None):
    """Return complete continuous scores, or fail rather than report partial AUC."""
    model.eval()
    scores = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            if budget is not None and budget.expired():
                raise TimeoutError(
                    "Evaluation reached the wall-clock limit; partial AUC was not reported"
                )
            features, _ = move_batch(batch, device)
            scores.extend(model(**features).float().cpu().tolist())
    return np.asarray(scores)


def train_stage(cfg, stage):
    """Train LoRA or mapping only; preserve last completed-step weights on a stop."""
    require_training_source(cfg)
    budget = Budget(cfg["budget_hours"])
    seed_all(cfg["seed"], cfg["threads"])
    rows = experiment_data(cfg)
    context = run_context(cfg, rows)
    mf, mf_metadata = load_mf(cfg, rows, context)
    adapter_path, parents = None, {}
    if stage == "stage2":
        adapter_path, adapter_metadata = load_checkpoint(cfg, "stage1", context, parents={})
        parents = {
            "mf": mf_metadata["artifact_sha256"],
            "stage1": adapter_metadata["artifact_sha256"],
        }
    root = prepare_stage(cfg, stage, context, parents)
    spec = cfg[stage]
    model = None
    completed = 0
    best_auc = None
    initial_auc = None
    status = "complete"
    failure = None
    validation_seconds = 0.0
    last_entry = None
    trainable_names = []
    try:
        model, tokenizer = build_model(cfg, mf, stage, adapter_path=adapter_path)
        training = make_loader(rows["train"], tokenizer, cfg, stage, training=True)
        validation = make_loader(rows["valid"], tokenizer, cfg, stage)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        expected = "lora_" if stage == "stage1" else "mapper."
        if not trainable_names or any(expected not in name for name in trainable_names):
            raise RuntimeError("Unexpected trainable parameter group")
        optimizer = torch.optim.AdamW(
            parameters, lr=spec["learning_rate"], weight_decay=spec["weight_decay"]
        )
        print(
            f"{stage}: {sum(parameter.numel() for parameter in parameters):,} trainable parameters on {cfg['device']}",
            flush=True,
        )

        started = time.monotonic()
        initial_auc = auc(rows["valid"], llm_scores(model, validation, budget))
        validation_seconds = time.monotonic() - started
        print(f"{stage} initial valid_auc={initial_auc:.5f}", flush=True)
        iterator = iter(training)
        optimizer.zero_grad(set_to_none=True)
        for step in range(1, spec["max_steps"] + 1):
            # Measured validation time replaces the old fixed five-minute reserve.
            reserve = validation_seconds * 1.2 + 5
            estimated_step = last_entry["step_seconds"] if last_entry else 0
            if budget.remaining <= reserve + estimated_step:
                status = "time_limit"
                break
            model.train()
            losses = []
            started = time.monotonic()
            for _ in range(spec["accumulation"]):
                if budget.expired():
                    raise TimeoutError("Time limit reached during gradient accumulation")
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(training)
                    batch = next(iterator)
                features, labels = move_batch(batch, cfg["device"])
                loss = nn.functional.binary_cross_entropy_with_logits(model(**features), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                (loss / spec["accumulation"]).backward()
                losses.append(float(loss.detach().cpu()))
            norm = nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
            completed = step
            optimizer.zero_grad(set_to_none=True)
            sync(cfg["device"])
            entry = {
                "step": step,
                "train_loss": float(np.mean(losses)),
                "gradient_norm": float(norm.detach().cpu()),
                "step_seconds": time.monotonic() - started,
                "elapsed_seconds": budget.elapsed,
            }
            if step % spec["eval_every"] == 0 or step == spec["max_steps"]:
                started = time.monotonic()
                valid_auc = auc(rows["valid"], llm_scores(model, validation, budget))
                validation_seconds = time.monotonic() - started
                entry["valid_auc"] = valid_auc
                if best_auc is None or valid_auc > best_auc:
                    directory = checkpoint_directory(root, f"step_{step:06d}")
                    save_weights(model, stage, directory)
                    select_checkpoint(root, stage, directory, context, parents, step, valid_auc)
                    best_auc = valid_auc
            last_entry = entry
            with (root / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            if step == 1 or step % 10 == 0 or "valid_auc" in entry:
                print(json.dumps({"stage": stage, **entry}), flush=True)

        if completed and (last_entry is None or "valid_auc" not in last_entry):
            if budget.remaining > validation_seconds * 1.1 + 2:
                valid_auc = auc(rows["valid"], llm_scores(model, validation, budget))
                if best_auc is None or valid_auc > best_auc:
                    directory = checkpoint_directory(root, f"step_{completed:06d}")
                    save_weights(model, stage, directory)
                    select_checkpoint(
                        root, stage, directory, context, parents, completed, valid_auc
                    )
                    best_auc = valid_auc
            else:
                status = "time_limit"
    except (Exception, KeyboardInterrupt) as error:
        failure = error
        if isinstance(error, TimeoutError):
            status = "time_limit"
        else:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    finally:
        # Save before leaving even if validation was interrupted or timed out.
        save_error = None
        if model is not None and completed:
            try:
                sync(cfg["device"])
                save_weights(model, stage, root / "last")
            except Exception as error:
                save_error = error
                status = "failed" if failure is None else status
        summary = {
            "status": status,
            "steps_completed": completed,
            "best_valid_auc": best_auc,
            "initial_valid_auc": initial_auc,
            "elapsed_seconds": budget.elapsed,
            "config": cfg,
            "data_manifest": context["data_manifest"],
            "parents": parents,
            "trainable_names": trainable_names,
            "test_used_for_selection": False,
            "checkpoint_selection": "highest complete validation AUC among trained checkpoints",
            "error": str(failure) if failure else None,
            "last_checkpoint_error": str(save_error) if save_error else None,
        }
        write_json(root / "summary.json", summary)
        write_json(root / "status.json", {"stage": stage, "status": status})
    if failure is not None and not isinstance(failure, TimeoutError):
        raise failure
    if save_error:
        raise save_error
    if best_auc is None:
        raise RuntimeError(
            "No trained checkpoint could be validated; last completed-step weights and status are preserved"
        )


def benchmark(cfg, device, steps=5):
    """Short timing/gradient check only; no experiment checkpoints are produced."""
    if steps < 1:
        raise ValueError("Benchmark steps must be positive")
    seed_all(cfg["seed"], cfg["threads"])
    cfg["device"] = device
    rows = experiment_data(cfg)
    model, tokenizer = build_model(cfg, make_mf(cfg, rows), "stage1")
    needed = max(cfg[stage]["batch_size"] for stage in ("stage1", "stage2"))
    examples = sorted(
        rows["train"], key=lambda row: len(" ".join(row["history_titles"])), reverse=True
    )[:needed]
    result = {
        "device": device,
        "steps_measured": steps,
        "max_length": cfg["model"]["max_length"],
        "stages": {},
    }
    for stage in ("stage1", "stage2"):
        model.set_stage(stage)
        dataset = PromptDataset(
            examples, tokenizer, cfg["model"]["max_length"], collaborative=stage == "stage2"
        )
        # No shuffle: CPU/XPU benchmarks receive exactly the same examples.
        loader = DataLoader(
            dataset,
            batch_size=cfg[stage]["batch_size"],
            collate_fn=Collator(tokenizer.pad_token_id),
        )
        features, labels = move_batch(next(iter(loader)), device)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0.0001)
        durations = []
        for step in range(steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            sync(device)
            started = time.perf_counter()
            loss = nn.functional.binary_cross_entropy_with_logits(model(**features), labels)
            loss.backward()
            gradient = sum(
                float(parameter.grad.detach().float().abs().sum().cpu())
                for parameter in parameters
                if parameter.grad is not None
            )
            if not np.isfinite(gradient) or gradient == 0:
                raise RuntimeError(f"No finite gradient reaches {stage}")
            optimizer.step()
            sync(device)
            if step:
                durations.append(time.perf_counter() - started)
        result["stages"][stage] = {
            "seconds_per_microbatch": float(np.median(durations)),
            "batch_size": int(labels.shape[0]),
            "tokens": int(features["attention_mask"].sum().cpu()),
            "finite_nonzero_gradient": True,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        }
        del optimizer
        gc.collect()
    result["process_ram_gib"] = psutil.Process().memory_info().rss / 2**30
    if device == "xpu":
        result["xpu_peak_allocated_gib"] = torch.xpu.max_memory_allocated() / 2**30
    result["batch_size"] = result["stages"]["stage1"]["batch_size"]
    write_json(ROOT / f"results/benchmark_{device}_batch{result['batch_size']}.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["mf", "stage1", "stage2", "benchmark"])
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--device", choices=["cpu", "xpu"])
    parser.add_argument("--steps", type=int, help="Measured steps for benchmark only")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.steps is not None and args.command != "benchmark":
        parser.error("--steps is for benchmark; set stage.max_steps in the YAML for training")
    if args.device:
        cfg["device"] = args.device
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        for stage in ("stage1", "stage2"):
            cfg[stage]["batch_size"] = args.batch_size
    if args.command == "benchmark":
        benchmark(cfg, cfg["device"], args.steps if args.steps is not None else 5)
    elif args.command == "mf":
        train_mf(cfg)
    else:
        train_stage(cfg, args.command)


if __name__ == "__main__":
    try:
        main()
    except StopRequested:
        print("Stopped safely; completed-step weights and status were preserved.", flush=True)
        sys.exit(130)
