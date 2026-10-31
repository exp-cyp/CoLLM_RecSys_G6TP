"""Reproducibility checks that do not load models, download assets, or train."""

import json
import socket
from copy import deepcopy

import pytest

from collm import assets
from collm.data import load_rows, manifest

MODEL_ID = "Qwen/Qwen2-0.5B"
REVISION = "a" * 40


def example_rows():
    return {
        "train": [
            {
                "record_id": "1:2:10",
                "uid": 1,
                "iid": 2,
                "timestamp": 10,
                "label": 1,
                "title": "Movie A",
                "history_titles": [],
                "warm": False,
            },
            {
                "record_id": "1:3:20",
                "uid": 1,
                "iid": 3,
                "timestamp": 20,
                "label": 0,
                "title": "Movie B",
                "history_titles": ["Movie A"],
                "warm": False,
            },
        ]
    }


def local_model(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"fixture weights; never loaded")
    return directory


@pytest.mark.parametrize(
    "field,value",
    [
        ("title", "Different Movie"),
        ("history_titles", ["Different Earlier Movie"]),
        ("uid", 9),
    ],
)
def test_content_fingerprint_detects_changed_inputs(field, value):
    original = example_rows()
    changed = deepcopy(original)
    changed["train"][1][field] = value
    before, after = manifest(original)["train"], manifest(changed)["train"]
    assert before["record_ids_sha256"] == after["record_ids_sha256"]
    assert before["content_sha256"] != after["content_sha256"]


def test_content_fingerprint_detects_label_swap_with_same_positive_rate():
    original = example_rows()
    changed = deepcopy(original)
    changed["train"][0]["label"], changed["train"][1]["label"] = 0, 1
    before, after = manifest(original)["train"], manifest(changed)["train"]
    assert before["positive_rate"] == after["positive_rate"]
    assert before["record_ids_sha256"] == after["record_ids_sha256"]
    assert before["content_sha256"] != after["content_sha256"]


def test_content_fingerprint_is_independent_of_dictionary_key_order():
    original = example_rows()
    reordered = {"train": [dict(reversed(list(row.items()))) for row in original["train"]]}
    assert manifest(original) == manifest(reordered)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_sample_limit_cannot_silently_select_the_full_split(limit):
    with pytest.raises(ValueError, match="positive integer"):
        load_rows("train", limit=limit)


def test_model_verification_is_offline_and_checks_pinned_identity(tmp_path, monkeypatch):
    directory = local_model(tmp_path)
    expected = assets.record_local_model_source(directory, MODEL_ID, REVISION)

    def network_forbidden(*args, **kwargs):
        raise AssertionError("Asset verification must not open a network connection")

    monkeypatch.setattr(socket, "socket", network_forbidden)
    assert assets.verify_local_model(directory, MODEL_ID, REVISION) == expected
    with pytest.raises(ValueError, match="identity/revision"):
        assets.verify_local_model(directory, MODEL_ID, "b" * 40)


@pytest.mark.parametrize("filename", ["config.json", "tokenizer.json", "model.safetensors"])
def test_model_verification_rejects_altered_local_files(tmp_path, filename):
    directory = local_model(tmp_path)
    assets.record_local_model_source(directory, MODEL_ID, REVISION)
    with (directory / filename).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        assets.verify_local_model(directory, MODEL_ID, REVISION)


def test_model_verification_rejects_missing_weight_shards(tmp_path):
    directory = local_model(tmp_path)
    index = {"weight_map": {"weight": "model-00002-of-00002.safetensors"}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing model weight shards"):
        assets.record_local_model_source(directory, MODEL_ID, REVISION)


def test_bad_existing_dataset_archive_is_rejected(tmp_path):
    archive = tmp_path / "ml-1m.zip"
    archive.write_bytes(b"incorrect or incomplete dataset")
    with pytest.raises(ValueError, match="MovieLens archive checksum mismatch"):
        assets.verify_dataset_archive(archive)
