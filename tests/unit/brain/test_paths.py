"""Tests for brain.paths — platformdirs-aware path resolution."""

from __future__ import annotations

import warnings
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
    """KINDLED_HOME env var fully overrides the platformdirs default."""
    override = tmp_path / "custom_home"
    monkeypatch.setenv("KINDLED_HOME", str(override))
    result = paths.get_home()
    assert result == override.resolve()


def test_get_home_falls_back_to_platformdirs(clean_env: None) -> None:
    """Without NELLBRAIN_HOME, get_home() returns a concrete Path."""
    result = paths.get_home()
    assert isinstance(result, Path)
    assert result.is_absolute()


def test_get_persona_dir_nests_under_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """get_persona_dir('nell') returns <home>/personas/nell."""
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    result = paths.get_persona_dir("nell")
    assert result == tmp_path.resolve() / "personas" / "nell"


def test_get_persona_dir_handles_multiple_personas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Different persona names resolve to different dirs under /personas/."""
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
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


def test_get_log_dir_respects_env_override(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """KINDLED_HOME redirects get_log_dir to <HOME>/logs."""
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    result = paths.get_log_dir()
    assert result == (tmp_path / "logs").resolve()


def test_get_log_dir_falls_back_to_platformdirs(clean_env: None) -> None:
    """Without NELLBRAIN_HOME, get_log_dir() returns the platformdirs path.

    Asserts the resolved path contains the project app name — platformdirs
    always nests under the appname on every supported OS.
    """
    result = paths.get_log_dir()
    assert isinstance(result, Path)
    assert result.is_absolute()
    assert "companion-emergence" in str(result).lower()


def test_get_cache_dir_respects_env_override(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """KINDLED_HOME redirects get_cache_dir to <HOME>/cache."""
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    result = paths.get_cache_dir()
    assert result == (tmp_path / "cache").resolve()


def test_get_cache_dir_falls_back_to_platformdirs(clean_env: None) -> None:
    """Without NELLBRAIN_HOME, get_cache_dir() returns the platformdirs path."""
    result = paths.get_cache_dir()
    assert isinstance(result, Path)
    assert result.is_absolute()
    assert "companion-emergence" in str(result).lower()


def test_get_persona_dir_rejects_path_traversal() -> None:
    with pytest.raises(ValueError):
        paths.get_persona_dir("../etc/passwd")


def test_get_persona_dir_rejects_forward_slash() -> None:
    with pytest.raises(ValueError):
        paths.get_persona_dir("a/b")


def test_get_persona_dir_rejects_dot_name() -> None:
    with pytest.raises(ValueError):
        paths.get_persona_dir("..")


def test_get_persona_dir_rejects_brace_chars() -> None:
    """Persona name with literal '{' or '}' would break str.format_map
    prompt rendering used by reflex/research engines."""
    with pytest.raises(ValueError):
        paths.get_persona_dir("evil{persona_name}")
    with pytest.raises(ValueError):
        paths.get_persona_dir("evil}persona{")


def test_get_persona_dir_rejects_empty() -> None:
    with pytest.raises(ValueError):
        paths.get_persona_dir("")


def test_get_persona_dir_accepts_valid_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    result = paths.get_persona_dir("nell_sandbox")
    assert result == tmp_path / "personas" / "nell_sandbox"


def test_get_persona_dir_rejects_names_with_spaces() -> None:
    """get_persona_dir must enforce the same grammar as validate_persona_name —
    no spaces, even though spaces aren't path-traversal chars."""
    with pytest.raises(ValueError, match=r"\[A-Za-z0-9_-\]"):
        paths.get_persona_dir("Nell Smith")


def test_get_persona_dir_rejects_names_with_special_chars() -> None:
    with pytest.raises(ValueError, match=r"\[A-Za-z0-9_-\]"):
        paths.get_persona_dir("nell@home.com")


def test_get_persona_dir_rejects_oversize_names() -> None:
    with pytest.raises(ValueError):
        paths.get_persona_dir("n" * 41)  # 41 chars; strict regex is {1,40}


def test_kindled_home_takes_priority_over_nellbrain_home(tmp_path, monkeypatch):
    """KINDLED_HOME is the canonical var as of v0.0.13.

    Both vars are set; the canonical path must win AND must not emit the
    NELLBRAIN_HOME deprecation warning (regression guard against
    accidentally walking into the fallback branch when KINDLED_HOME is
    already set).
    """
    kindled = tmp_path / "kindled"
    nell = tmp_path / "nell"
    kindled.mkdir()
    nell.mkdir()
    monkeypatch.setenv("KINDLED_HOME", str(kindled))
    monkeypatch.setenv("NELLBRAIN_HOME", str(nell))
    from brain.paths import get_home

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = get_home()
    assert result == kindled.resolve()
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings == [], (
        f"KINDLED_HOME path must not emit DeprecationWarning; got: "
        f"{[str(w.message) for w in deprecation_warnings]}"
    )


def test_nellbrain_home_still_honored_with_deprecation_warning(tmp_path, monkeypatch):
    """Backwards-compat fallback: NELLBRAIN_HOME works but warns.
    Fallback is removed in v0.0.14."""
    nell = tmp_path / "nell"
    nell.mkdir()
    monkeypatch.delenv("KINDLED_HOME", raising=False)
    monkeypatch.setenv("NELLBRAIN_HOME", str(nell))
    from brain.paths import get_home

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = get_home()
    assert result == nell.resolve()
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1
    assert "NELLBRAIN_HOME is deprecated" in str(deprecation_warnings[0].message)
    assert "KINDLED_HOME" in str(deprecation_warnings[0].message)
    assert "v0.0.14" in str(deprecation_warnings[0].message)


def test_neither_env_var_falls_back_to_platformdirs(tmp_path, monkeypatch):
    """Neither set → platformdirs default (unchanged from prior behaviour)."""
    monkeypatch.delenv("KINDLED_HOME", raising=False)
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    from brain.paths import get_home

    result = get_home()
    # Result is the OS-appropriate user_data_path — just verify it's a
    # real absolute path that isn't either env-var value.
    assert result.is_absolute()
    assert "companion-emergence" in str(result)


def test_kindled_home_propagates_to_cache_dir(tmp_path, monkeypatch):
    kindled = tmp_path / "k"
    kindled.mkdir()
    monkeypatch.setenv("KINDLED_HOME", str(kindled))
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    from brain.paths import get_cache_dir

    assert get_cache_dir() == (kindled / "cache").resolve()


def test_kindled_home_propagates_to_log_dir(tmp_path, monkeypatch):
    kindled = tmp_path / "k"
    kindled.mkdir()
    monkeypatch.setenv("KINDLED_HOME", str(kindled))
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    from brain.paths import get_log_dir

    assert get_log_dir() == (kindled / "logs").resolve()
