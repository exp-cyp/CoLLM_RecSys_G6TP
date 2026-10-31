"""Check scientific failure modes: leakage, frozen stages, and scoring equivalence."""

import json

import pandas as pd
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from collm.common import ROOT
from collm.data import make_records
from collm.models import CoLLM, MatrixFactorization, PromptDataset

torch.set_num_threads(2)


def tiny_model(stage):
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    base = Qwen2ForCausalLM(config)
    lm = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    mf = MatrixFactorization(4, 5, 4)
    mf.set_seen([{"uid": 1, "iid": 1}, {"uid": 2, "iid": 2}])
    return CoLLM(lm, mf, yes_id=10, stage=stage)


def batch():
    return {
        "input_ids": torch.tensor([[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "uid": torch.tensor([1, 2]),
        "iid": torch.tensor([1, 2]),
        "user_pos": torch.tensor([1, 1]),
        "item_pos": torch.tensor([3, 3]),
    }


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_only_the_intended_module_receives_gradients(stage):
    model = tiny_model(stage).train()
    data = batch()
    labels = torch.tensor([0.0, 1.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**data), labels)
    loss.backward()
    wanted = "lora_" if stage == "stage1" else "mapper."
    changed = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert changed and all(wanted in name for name in changed)
    assert all(p.grad is None for p in model.mf.parameters())
    if stage == "stage2":
        assert not model.lm.training
        assert all(p.grad is None for p in model.lm.parameters())


def test_yes_projection_matches_full_lm_logits():
    model = tiny_model("stage1").eval()
    data = batch()
    with torch.no_grad():
        optimized = model(**data)
        full = model.lm(
            input_ids=data["input_ids"], attention_mask=data["attention_mask"], use_cache=False
        ).logits[:, -1]
    torch.testing.assert_close(optimized, full[:, 10])


def test_only_yes_head_row_is_projected(monkeypatch):
    """The scoring path must not allocate logits for discarded vocabulary rows."""
    model = tiny_model("stage1").eval()
    original_linear = torch.nn.functional.linear
    projected_shapes = []

    def capture_projection(inputs, weight, bias=None):
        projected_shapes.append(tuple(weight.shape))
        return original_linear(inputs, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", capture_projection)
    with torch.no_grad():
        scores = model(**batch())
    assert scores.shape == (2,)
    assert projected_shapes[-1] == (1, model.lm.config.hidden_size)


def test_frozen_lm_preserves_gradient_to_collaborative_inputs():
    """Stage 2 must differentiate through the frozen decoder back to the mapper."""
    model = tiny_model("stage2").train()
    inputs = batch()
    captured = {}

    def retain_input_gradient(module, args, kwargs):
        embeddings = kwargs["inputs_embeds"]
        embeddings.retain_grad()
        captured["embeddings"] = embeddings

    decoder = model.lm.get_base_model().model
    hook = decoder.register_forward_pre_hook(retain_input_gradient, with_kwargs=True)
    try:
        labels = torch.tensor([0.0, 1.0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**inputs), labels)
        loss.backward()
    finally:
        hook.remove()

    gradient = captured["embeddings"].grad
    indices = torch.arange(inputs["uid"].shape[0])
    assert torch.isfinite(gradient).all()
    assert gradient[indices, inputs["user_pos"]].abs().sum() > 0
    assert gradient[indices, inputs["item_pos"]].abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in model.lm.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.mapper.parameters())


def test_left_padding_does_not_change_scores_or_feature_positions():
    model = tiny_model("stage2").eval()
    original = batch()
    padded = {k: v.clone() for k, v in original.items()}
    padded["input_ids"] = torch.cat(
        [torch.zeros(2, 3, dtype=torch.long), original["input_ids"]], dim=1
    )
    padded["attention_mask"] = torch.cat(
        [torch.zeros(2, 3, dtype=torch.long), original["attention_mask"]], dim=1
    )
    padded["user_pos"] += 3
    padded["item_pos"] += 3
    with torch.no_grad():
        torch.testing.assert_close(model(**original), model(**padded), atol=2e-6, rtol=2e-5)


def test_unseen_collaborative_ids_do_not_use_random_embeddings():
    mf = tiny_model("stage2").mf
    u, i = mf.features(torch.tensor([3]), torch.tensor([4]))
    assert torch.count_nonzero(u) == torch.count_nonzero(i) == 0


def test_history_has_no_current_or_equal_time_label_leakage():
    frame = pd.DataFrame(
        [
            (1, 1, 5, 10, "A", "train"),
            (1, 2, 5, 10, "B", "train"),
            (1, 3, 2, 20, "C", "valid"),
            (1, 4, 5, 30, "D", "test"),
        ],
        columns=["uid", "iid", "rating", "timestamp", "title", "split"],
    )
    rows, _, _ = make_records(frame)
    assert all(r["history_titles"] == [] for r in rows["train"])
    assert rows["valid"][0]["label"] == 0
    assert rows["valid"][0]["history_titles"] == ["A", "B"]
    assert rows["test"][0]["history_titles"] == ["A", "B"]
    assert all(t < rows["test"][0]["timestamp"] for t in rows["test"][0]["history_timestamps"])


class WordTokenizer:
    eos_token_id = 1
    pad_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + len(word) % 60 for word in text.split()]


def test_long_history_cannot_truncate_target_placeholders():
    row = {
        "uid": 1,
        "iid": 2,
        "label": 1,
        "title": "Target Film",
        "history_titles": ["long history movie " * 100] * 10,
    }
    dataset = PromptDataset([row], WordTokenizer(), max_length=128, collaborative=True)
    example = dataset[0]
    assert len(example["input_ids"]) == 128
    assert example["input_ids"][example["user_pos"]] == 1
    assert example["input_ids"][example["item_pos"]] == 1
    assert example["user_pos"] < example["item_pos"] < 127


@pytest.mark.skipif(
    not (ROOT / "data/processed/metadata.json").exists(), reason="run data preparation first"
)
def test_real_data_split_and_history_audit():
    splits = {}
    for name, expected in (("train", 33891), ("valid", 10401), ("test", 7331)):
        rows = [
            json.loads(line)
            for line in (ROOT / f"data/processed/{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == expected
        assert all(all(t < r["timestamp"] for t in r["history_timestamps"]) for r in rows)
        assert all(len(r["history_titles"]) <= 10 for r in rows)
        splits[name] = rows
    assert max(r["timestamp"] for r in splits["train"]) < min(
        r["timestamp"] for r in splits["valid"]
    )
    assert max(r["timestamp"] for r in splits["valid"]) < min(
        r["timestamp"] for r in splits["test"]
    )
    ids = [r["record_id"] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids))
