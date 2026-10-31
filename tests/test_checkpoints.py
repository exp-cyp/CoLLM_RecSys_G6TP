"""Protect experiment identity and checkpoint lineage with tiny local fixtures."""

from copy import deepcopy

import pytest

from collm import checkpoints, common


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(checkpoints, "ROOT", tmp_path)
    source_dir = tmp_path / "collm"
    source_dir.mkdir()
    for filename in ("models.py", "data.py", "train.py"):
        (source_dir / filename).write_text("VALUE = 1\n", encoding="utf-8")
    source = {
        "schema_version": 1,
        "model_id": "Qwen/fixture",
        "model_revision": "a" * 40,
        "files": {"config.json": "config-hash", "model.safetensors": "weights-hash"},
    }
    monkeypatch.setattr(checkpoints, "verify_local_model", lambda *_: deepcopy(source))
    stage = {
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "batch_size": 1,
        "accumulation": 1,
        "max_steps": 2,
        "eval_every": 1,
        "eval_batch_size": 1,
    }
    cfg = {
        "seed": 42,
        "run_kind": "experiment",
        "device": "cpu",
        "threads": 2,
        "budget_hours": 1,
        "output_dir": "outputs/run",
        "model": {
            "id": "Qwen/fixture",
            "revision": "a" * 40,
            "path": "models/fixture",
            "max_length": 128,
        },
        "data": {"train_limit": 2, "valid_limit": 2, "test_limit": 2},
        "mf": {
            "embedding_dim": 4,
            "learning_rates": [0.001],
            "weight_decay": 0.0001,
            "epochs": 2,
            "patience": 1,
            "batch_size": 2,
        },
        "stage1": deepcopy(stage),
        "stage2": deepcopy(stage),
    }
    rows = {
        split: [
            {
                "record_id": f"{split}:1",
                "uid": 1,
                "iid": 1,
                "label": 1,
                "title": "Movie A",
                "history_titles": [],
            },
            {
                "record_id": f"{split}:2",
                "uid": 1,
                "iid": 2,
                "label": 0,
                "title": "Movie B",
                "history_titles": ["Movie A"],
            },
        ]
        for split in ("train", "valid", "test")
    }
    return {"root": tmp_path, "cfg": cfg, "rows": rows, "source": source}


def selected_checkpoint(project, stage="mf", parents=None):
    cfg, rows = project["cfg"], project["rows"]
    parents = parents or {}
    context = checkpoints.run_context(cfg, rows)
    root = checkpoints.prepare_stage(cfg, stage, context, parents)
    directory = root / "checkpoint-000001"
    directory.mkdir()
    for name in checkpoints.ARTIFACT_FILES[stage]:
        (directory / name).write_bytes(f"fixture {stage} {name}".encode())
    metadata = checkpoints.select_checkpoint(root, stage, directory, context, parents, 1, 0.6)
    return directory, metadata, context


@pytest.mark.parametrize("path", ["outputs", "outputs/../README.md", "../outside/run"])
def test_output_paths_cannot_escape_or_replace_the_output_root(project, path):
    with pytest.raises(ValueError):
        common.project_path(path, "outputs")


def test_absolute_outside_output_path_is_rejected(project):
    outside = project["root"].parent / "unrelated-output"
    with pytest.raises(ValueError, match="inside"):
        common.project_path(outside, "outputs")


def test_nested_results_preserve_the_full_run_path(project):
    cfg = project["cfg"]
    cfg["output_dir"] = "outputs/group-a/run"
    first = common.results_path(cfg)
    cfg["output_dir"] = "outputs/group-b/run"
    second = common.results_path(cfg)
    assert first == project["root"] / "results/group-a/run"
    assert second == project["root"] / "results/group-b/run"
    assert first != second


@pytest.mark.parametrize(
    "keys,value",
    [
        (("seed",), -1),
        (("seed",), True),
        (("device",), "cuda"),
        (("threads",), 0),
        (("budget_hours",), 0),
        (("budget_hours",), 6.1),
        (("budget_hours",), float("nan")),
        (("data", "train_limit"), 0),
        (("data", "valid_limit"), -1),
        (("model", "revision"), "main"),
        (("model", "max_length"), 0),
        (("mf", "epochs"), 0),
        (("mf", "learning_rates"), []),
        (("mf", "learning_rates"), [-0.001]),
        (("stage1", "accumulation"), 0),
        (("stage2", "max_steps"), 0),
        (("stage2", "learning_rate"), float("inf")),
        (("stage2", "weight_decay"), -0.1),
    ],
)
def test_invalid_config_is_rejected_before_loading_models(project, keys, value):
    cfg = deepcopy(project["cfg"])
    container = cfg
    for key in keys[:-1]:
        container = container[key]
    container[keys[-1]] = value
    with pytest.raises(ValueError):
        common.validate_config(cfg)


def test_full_split_null_limit_and_zero_weight_decay_are_valid(project):
    cfg = deepcopy(project["cfg"])
    cfg["data"]["train_limit"] = None
    cfg["stage1"]["weight_decay"] = 0
    common.validate_config(cfg)


def test_populated_stage_is_never_silently_overwritten(project):
    _, _, context = selected_checkpoint(project)
    with pytest.raises(FileExistsError, match="overwrite"):
        checkpoints.prepare_stage(project["cfg"], "mf", context, {})


def test_legacy_nonempty_output_without_manifest_is_refused(project):
    directory = project["root"] / "outputs/run"
    directory.mkdir(parents=True)
    (directory / "old-checkpoint.pt").write_bytes(b"old artifact")
    context = checkpoints.run_context(project["cfg"], project["rows"])
    with pytest.raises(FileExistsError, match="no compatible run manifest"):
        checkpoints.prepare_stage(project["cfg"], "mf", context, {})


def test_same_output_directory_cannot_be_reused_for_another_contract(project):
    selected_checkpoint(project)
    cfg = deepcopy(project["cfg"])
    cfg["seed"] += 1
    context = checkpoints.run_context(cfg, project["rows"])
    with pytest.raises(ValueError, match="different experiment"):
        checkpoints.prepare_stage(cfg, "stage1", context, {})


@pytest.mark.parametrize("change", ["model", "config", "labels", "history", "model_bytes"])
def test_checkpoint_rejects_model_config_or_data_contract_changes(project, change):
    selected_checkpoint(project)
    cfg, rows = deepcopy(project["cfg"]), deepcopy(project["rows"])
    if change == "model":
        cfg["model"]["revision"] = "b" * 40
    elif change == "config":
        cfg["stage1"]["learning_rate"] *= 2
    elif change == "labels":
        rows["train"][0]["label"], rows["train"][1]["label"] = 0, 1
    elif change == "history":
        rows["valid"][1]["history_titles"] = ["A different earlier movie"]
    else:
        project["source"]["files"]["model.safetensors"] = "changed-weights"
    context = checkpoints.run_context(cfg, rows)
    with pytest.raises(ValueError, match="model/config/data content"):
        checkpoints.load_checkpoint(cfg, "mf", context)


@pytest.mark.parametrize("stage", ["mf", "stage1", "stage2"])
def test_changed_artifact_files_are_rejected(project, stage):
    directory, _, context = selected_checkpoint(project, stage)
    artifact = directory / checkpoints.ARTIFACT_FILES[stage][0]
    with artifact.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="files have changed"):
        checkpoints.load_checkpoint(project["cfg"], stage, context)


@pytest.mark.parametrize("parent", ["mf", "stage1"])
def test_stage2_is_bound_to_exact_mf_and_stage1_parents(project, parent):
    parents = {"mf": "mf-checkpoint-A", "stage1": "adapter-checkpoint-A"}
    _, _, context = selected_checkpoint(project, "stage2", parents)
    other_parents = dict(parents)
    other_parents[parent] = "a-different-checkpoint"
    with pytest.raises(ValueError, match="different MF/LoRA parents"):
        checkpoints.load_checkpoint(project["cfg"], "stage2", context, other_parents)
    checkpoints.load_checkpoint(project["cfg"], "stage2", context, parents)


def test_checkpoint_directory_cannot_escape_its_stage(project):
    _, metadata, context = selected_checkpoint(project)
    metadata["checkpoint"] = "../../outside"
    common.write_json(checkpoints.stage_path(project["cfg"], "mf") / "best.json", metadata)
    with pytest.raises(ValueError, match="escapes"):
        checkpoints.load_checkpoint(project["cfg"], "mf", context)


def test_runtime_only_changes_remain_checkpoint_compatible(project):
    directory, _, original = selected_checkpoint(project)
    cfg = deepcopy(project["cfg"])
    cfg.update({"device": "xpu", "threads": 4, "budget_hours": 2})
    cfg["model"]["path"] = "models/relocated-identical-model"
    updated = checkpoints.run_context(cfg, project["rows"])
    assert original["contract_sha256"] == updated["contract_sha256"]
    loaded, _ = checkpoints.load_checkpoint(cfg, "mf", updated)
    assert loaded == directory
    cfg["output_dir"] = "outputs/another-run"
    moved = checkpoints.run_context(cfg, project["rows"])
    assert moved["contract_sha256"] == original["contract_sha256"]


def test_implementation_hash_ignores_formatting_but_detects_logic_changes(project):
    cfg, rows = project["cfg"], project["rows"]
    original = checkpoints.run_context(cfg, rows)
    source = project["root"] / "collm/models.py"
    source.write_text("# formatting change\nVALUE=1\n", encoding="utf-8")
    formatted = checkpoints.run_context(cfg, rows)
    assert formatted["contract_sha256"] == original["contract_sha256"]
    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = checkpoints.run_context(cfg, rows)
    assert changed["contract_sha256"] != original["contract_sha256"]
