"""Exercise short budgets and cancellation without loading an LLM or using XPU."""

import numpy as np
import pytest
import torch
from torch import nn

from collm import train
from collm.common import load_config, read_json
from collm.evaluate import report


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = nn.Parameter(torch.tensor(0.1))

    def save_pretrained(self, directory, safe_serialization=True):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        # Tests only check persistence/identity; this file is never loaded as an LLM.
        (directory / "adapter_model.safetensors").write_bytes(
            self.lora_weight.detach().numpy().tobytes()
        )


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = TinyAdapter()

    def forward(self, value):
        return self.lm.lora_weight * value


@pytest.fixture
def training_fixture(tmp_path, monkeypatch):
    cfg = load_config("configs/smoke.yaml")
    cfg["device"] = "cpu"
    cfg["budget_hours"] = 0.02  # Under five minutes: must still permit short runs.
    cfg["stage1"].update(max_steps=2, eval_every=1, accumulation=1)
    rows = {split: [{"label": 0}, {"label": 1}] for split in ("train", "valid", "test")}
    context = {
        "schema_version": 2,
        "contract_sha256": "fixture",
        "data_manifest": {},
        "config": cfg,
    }
    root = tmp_path / "stage1"
    root.mkdir()
    model = TinyModel()
    monkeypatch.setattr(train, "experiment_data", lambda config: rows)
    monkeypatch.setattr(train, "run_context", lambda config, selected: context)
    monkeypatch.setattr(train, "load_mf", lambda *args: (None, {"artifact_sha256": "mf"}))
    monkeypatch.setattr(train, "prepare_stage", lambda *args: root)
    monkeypatch.setattr(train, "build_model", lambda *args, **kwargs: (model, None))
    batches = [{"value": torch.tensor([-1.0, 1.0]), "label": torch.tensor([0.0, 1.0])}]
    monkeypatch.setattr(train, "make_loader", lambda *args, **kwargs: batches)
    monkeypatch.setattr(train, "llm_scores", lambda *args, **kwargs: np.array([0.2, 0.8]))
    return cfg, root, model


def test_short_budget_allows_optimizer_updates(training_fixture):
    cfg, root, model = training_fixture
    train.train_stage(cfg, "stage1")
    summary = read_json(root / "summary.json")
    assert summary["status"] == "complete"
    assert summary["steps_completed"] == 2
    assert model.lm.lora_weight.item() != pytest.approx(0.1)
    assert (root / "last/adapter_model.safetensors").exists()


@pytest.mark.parametrize(
    "error,expected_status",
    [(TimeoutError("deadline"), "time_limit"), (KeyboardInterrupt(), "interrupted")],
)
def test_stopped_validation_preserves_last_completed_step(
    training_fixture, monkeypatch, error, expected_status
):
    cfg, root, _ = training_fixture
    calls = 0

    def scores(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        return np.array([0.2, 0.8])

    monkeypatch.setattr(train, "llm_scores", scores)
    expected_error = RuntimeError if isinstance(error, TimeoutError) else KeyboardInterrupt
    with pytest.raises(expected_error):
        train.train_stage(cfg, "stage1")
    summary = read_json(root / "summary.json")
    assert summary["status"] == expected_status
    assert summary["steps_completed"] == 1
    assert summary["best_valid_auc"] is None
    assert (root / "last/adapter_model.safetensors").exists()
    assert not (root / "best.json").exists()


def test_uncommitted_checkpoint_is_never_reported_as_best(training_fixture, monkeypatch):
    cfg, root, _ = training_fixture
    original = train.save_weights

    def save(model, stage, directory):
        if directory.parent.name == "checkpoints":
            raise OSError("simulated interrupted checkpoint write")
        original(model, stage, directory)

    monkeypatch.setattr(train, "save_weights", save)
    with pytest.raises(OSError, match="checkpoint write"):
        train.train_stage(cfg, "stage1")
    assert read_json(root / "summary.json")["best_valid_auc"] is None
    assert not (root / "best.json").exists()
    assert (root / "last/adapter_model.safetensors").exists()


def test_report_uses_actual_config_and_training_counts(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    cfg["mf"]["embedding_dim"] = 128
    metrics = {
        name: {
            "auc": 0.6,
            "warm_auc": None,
            "cold_auc": 0.6,
            "checkpoint_step": 3,
            "training_steps_completed": 7,
            "training_status": "complete",
        }
        for name in ("MF", "TALLRec", "CoLLM-MF")
    }
    results = {
        "run_kind": "smoke",
        "model": cfg["model"]["id"],
        "model_revision": cfg["model"]["revision"],
        "device": "cpu",
        "config": cfg,
        "data_manifest": {
            split: {"count": count} for split, count in (("train", 48), ("valid", 20), ("test", 24))
        },
        "metrics": metrics,
    }
    report(results, tmp_path)
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "train=48, validation=20, test=24" in text
    assert "128" in text
    assert "7 optimizer steps" in text
    assert "32 training" not in text
