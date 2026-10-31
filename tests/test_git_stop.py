"""Real Train is gated on published source, and Stop remains cooperative."""

import pytest

from collm import git_state
from collm.common import Budget, StopRequested


def git_fixture(monkeypatch, *, dirty="", commit="abc", upstream_commit="abc"):
    values = {
        ("status", "--porcelain", "--untracked-files=normal"): dirty,
        ("symbolic-ref", "--short", "HEAD"): "main",
        ("rev-parse", "--abbrev-ref", "@{upstream}"): "origin/main",
        ("rev-parse", "HEAD"): commit,
        ("rev-parse", "@{upstream}"): upstream_commit,
    }
    monkeypatch.setattr(git_state, "git_output", lambda *arguments, **kwargs: values[arguments])


def test_real_train_requires_clean_source(monkeypatch):
    git_fixture(monkeypatch, dirty=" M collm/models.py")
    with pytest.raises(RuntimeError, match="commit"):
        git_state.require_published()


def test_real_train_requires_current_commit_to_be_pushed(monkeypatch):
    git_fixture(monkeypatch, commit="new", upstream_commit="old")
    with pytest.raises(RuntimeError, match="upstream"):
        git_state.require_published()


def test_published_source_records_exact_commit(monkeypatch):
    git_fixture(monkeypatch)
    assert git_state.require_published() == {
        "commit": "abc",
        "branch": "main",
        "upstream": "origin/main",
    }


def test_stop_file_is_checked_between_operations(tmp_path, monkeypatch):
    stop = tmp_path / "STOP"
    monkeypatch.setenv("COLLM_STOP_FILE", str(stop))
    budget = Budget(1)
    assert not budget.expired()
    stop.write_text("stop", encoding="utf-8")
    with pytest.raises(StopRequested):
        budget.expired()
    with pytest.raises(StopRequested):
        _ = budget.remaining
