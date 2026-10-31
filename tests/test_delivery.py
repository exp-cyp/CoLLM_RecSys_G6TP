"""Test final reports and ZIP contents using tiny result fixtures only."""

import json
import zipfile
from copy import deepcopy

import pytest

from collm import common, delivery
from scripts.package_submission import package_project


@pytest.fixture
def result_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(delivery, "ROOT", tmp_path)
    cfg = {
        "run_kind": "smoke",
        "output_dir": "outputs/run01",
        "device": "cpu",
        "model": {"id": "Qwen/Qwen2-0.5B", "revision": "a" * 40},
        "stage1": {"max_steps": 2},
        "stage2": {"max_steps": 2},
    }
    metrics = {
        "model": cfg["model"]["id"],
        "elapsed_seconds": 4,
        "evaluation_protocol": "observed ratings, rolling history",
        "data_manifest": {
            split: {"count": count} for split, count in (("train", 32), ("valid", 16), ("test", 16))
        },
        "metrics": {
            name: {"auc": value, "warm_auc": None, "cold_auc": value}
            for name, value in (("MF", 0.5), ("TALLRec", 0.62), ("CoLLM-MF", 0.58))
        },
    }
    stages = {
        stage: {
            "status": "complete",
            "steps_completed": 2,
            "elapsed_seconds": 3,
            "best_valid_auc": 0.6,
        }
        for stage in ("mf", "stage1", "stage2")
    }
    results = tmp_path / "results/run01"
    results.mkdir(parents=True)
    (results / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (results / "predictions.csv").write_text("label,score\n1,0.8\n", encoding="utf-8")
    (results / "REPORT.md").write_text("# Existing evaluation report", encoding="utf-8")
    for stage, summary in stages.items():
        output = tmp_path / "outputs/run01" / stage
        output.mkdir(parents=True)
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    job = tmp_path / ".cache/jobs/run01"
    job.mkdir(parents=True)
    (tmp_path / "README.md").write_text("# Source", encoding="utf-8")
    (tmp_path / "START_TRAIN.cmd").write_text("@echo off", encoding="utf-8")
    return cfg, metrics, stages, results, job


def test_summary_reports_worse_result_honestly(result_fixture):
    cfg, metrics, stages, _, _ = result_fixture
    summary = delivery.build_summary(
        cfg, metrics, stages, {"elapsed_seconds": 20, "git_commit": "f" * 40}
    )
    text = delivery.thai_summary(summary)
    assert summary["collm_auc_delta"]["TALLRec"] == pytest.approx(-0.04)
    assert "ต่ำกว่า TALLRec 0.0400" in text
    assert "สูงกว่า MF 0.0800" in text
    assert "ไม่ใช่ผลวิจัย" in text
    assert summary["pipeline_elapsed_seconds"] == 20
    assert summary["training_elapsed_seconds"] == 9


@pytest.mark.parametrize("reason", ["time_limit", "steps_short"])
def test_incomplete_training_is_not_labeled_complete(result_fixture, reason):
    cfg, metrics, stages, _, _ = result_fixture
    stages = deepcopy(stages)
    if reason == "time_limit":
        stages["stage2"]["status"] = "time_limit"
    else:
        stages["stage2"]["steps_completed"] = 1
    summary = delivery.build_summary(cfg, metrics, stages, {})
    assert summary["completion_status"] == "time_limit"
    assert summary["partial_stages"] == ["stage2"]
    assert "ทำไม่ครบ" in delivery.thai_summary(summary)


def test_failed_stage_cannot_be_finalized(result_fixture):
    cfg, metrics, stages, _, _ = result_fixture
    stages["stage1"]["status"] = "failed"
    with pytest.raises(ValueError, match="Cannot finalize"):
        delivery.build_summary(cfg, metrics, stages, {})


def test_missing_stage_cannot_be_finalized(result_fixture):
    cfg, metrics, stages, _, _ = result_fixture
    del stages["stage2"]
    with pytest.raises(ValueError, match="requires MF"):
        delivery.build_summary(cfg, metrics, stages, {})


def test_reused_training_time_is_separate_from_current_run(result_fixture):
    cfg, metrics, stages, _, _ = result_fixture
    summary = delivery.build_summary(
        cfg, metrics, stages, {"stages": {"mf": {"reused": True}, "stage1": {"reused": True}}}
    )
    assert summary["reused_stages"] == ["mf", "stage1"]
    assert summary["training_elapsed_seconds"] == 9
    assert summary["current_training_elapsed_seconds"] == 3


@pytest.mark.parametrize("value", [float("nan"), 1.2, None])
def test_invalid_or_missing_test_scores_are_rejected(result_fixture, value):
    cfg, metrics, stages, _, _ = result_fixture
    metrics["metrics"]["CoLLM-MF"]["auc"] = value
    with pytest.raises(ValueError, match="invalid completed test AUC"):
        delivery.build_summary(cfg, metrics, stages, {})


def test_finalize_generates_all_deliverables_without_model_loading(result_fixture):
    cfg, _, _, results, job = result_fixture
    value = delivery.finalize_run(cfg, job, {"elapsed_seconds": 15, "git_commit": "f" * 40})
    assert value["completion_status"] == "complete"
    assert value["results_dir"] == "results/run01"
    for filename in ("SUMMARY_TH.md", "report.html", "pipeline_summary.json", "config_used.yaml"):
        assert (results / filename).is_file()
    for stage in ("mf", "stage1", "stage2"):
        assert (results / f"{stage}_summary.json").is_file()
    page = (results / "report.html").read_text(encoding="utf-8")
    assert "Smoke test" in page
    assert "ต่ำกว่า TALLRec" in page
    assert '<html lang="th">' in page
    archive = results.parent.parent / value["source_zip"]
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        names = zipped.namelist()
        assert "START_TRAIN.cmd" in names
        assert "results/run01/report.html" in names
        assert "results/run01/config_used.yaml" in names
        assert not any(name.endswith(".zip") for name in names)
        assert not any(name.startswith(("outputs/", ".cache/")) for name in names)


def test_per_run_archive_excludes_unrelated_results_and_large_assets(result_fixture):
    _, _, _, results, _ = result_fixture
    root = results.parent.parent
    for name in (
        "models/weights.safetensors",
        "data/raw.csv",
        "outputs/run01/model.pt",
        ".venv/library.py",
        ".cache/private.txt",
        "results/other/old.csv",
        "results/run01/old.zip",
        "collm/accidental.pt",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"excluded")
    code = root / "collm/model.py"
    code.write_text("value = 1\n", encoding="utf-8")
    archive = package_project(root, "results/run01/bundle.zip", "results/run01")
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert "collm/model.py" in names
        assert "results/run01/predictions.csv" in names
        assert not any(
            name.startswith(("models/", "data/", "outputs/", ".venv/", ".cache/", "results/other/"))
            for name in names
        )
        assert not any(name.endswith((".pt", ".zip", ".safetensors")) for name in names)
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        package_project(root, "results/run01/bundle.zip", "results/run01")


def test_archive_paths_cannot_escape_the_project(result_fixture):
    _, _, _, results, _ = result_fixture
    root = results.parent.parent
    with pytest.raises(ValueError, match="Archive target"):
        package_project(root, root.parent / "outside.zip", results)
    with pytest.raises(ValueError, match="Choose one run directory"):
        package_project(root, "submission/source.zip", root / "data")
