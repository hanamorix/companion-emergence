"""Tests for brain.paths — platformdirs-aware path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import paths


def test_get_home_returns_a_path(clean_env: None) -> None:
    """get_home() returns a pathlib.Path."""
    result = paths.get_home()
    assert isinstance(result, Path)


def test_get_home_respects_env_override(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NELLBRAIN_HOME env var fully overrides the platformdirs default."""
    override = tmp_path / "custom_home"
    monkeypatch.setenv("NELLBRAIN_HOME", str(override))
    result = paths.get_home()
    assert result == override.resolve()


def test_get_home_falls_back_to_platformdirs(clean_env: None) -> None:
    """Without NELLBRAIN_HOME, get_home() returns a concrete Path."""
    result = paths.get_home()
    assert isinstance(result, Path)
    assert result.is_absolute()


def test_get_persona_dir_nests_under_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """get_persona_dir('nell') returns <home>/personas/nell."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    result = paths.get_persona_dir("nell")
    assert result == tmp_path.resolve() / "personas" / "nell"


def test_get_persona_dir_handles_multiple_personas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Different persona names resolve to different dirs under /personas/."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    nell = paths.get_persona_dir("nell")
    sage = paths.get_persona_dir("sage")
    assert nell != sage
    assert nell.parent == sage.parent


def test_get_cache_dir_is_absolute_path(clean_env: None) -> None:
    """get_cache_dir() returns an absolute Path."""
    result = paths.get_cache_dir()
    assert isinstance(result, Path)
    assert result.is_absolute()


def test_get_log_dir_is_absolute_path(clean_env: None) -> None:
    """get_log_dir() returns an absolute Path."""
    result = paths.get_log_dir()
    assert isinstance(result, Path)
    assert result.is_absolute()
