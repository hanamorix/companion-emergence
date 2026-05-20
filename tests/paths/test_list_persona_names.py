"""list_persona_names() scans <home>/personas/ deterministically."""

from pathlib import Path
import pytest

from brain.paths import list_persona_names


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    return tmp_path


def test_empty_returns_empty_list(tmp_home):
    (tmp_home / "personas").mkdir()
    assert list_persona_names() == []


def test_returns_sorted_names(tmp_home):
    for name in ["mira", "alex", "nell"]:
        (tmp_home / "personas" / name).mkdir(parents=True)
    assert list_persona_names() == ["alex", "mira", "nell"]


def test_skips_non_directories(tmp_home):
    (tmp_home / "personas").mkdir()
    (tmp_home / "personas" / "nell").mkdir()
    (tmp_home / "personas" / "stray.txt").write_text("ignore me")
    assert list_persona_names() == ["nell"]


def test_skips_invalid_grammar(tmp_home):
    (tmp_home / "personas").mkdir()
    (tmp_home / "personas" / "nell").mkdir()
    (tmp_home / "personas" / ".hidden").mkdir()
    (tmp_home / "personas" / "has space").mkdir()
    assert list_persona_names() == ["nell"]


def test_personas_root_missing_returns_empty(tmp_home):
    assert list_persona_names() == []
