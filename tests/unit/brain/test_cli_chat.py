"""Tests for `nell chat` CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.cli import main


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    """Minimal persona directory wired to the fake provider.

    get_home() will return tmp_path, so get_persona_dir("nell")
    resolves to tmp_path/personas/nell.
    """
    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop"}),
        encoding="utf-8",
    )
    # Minimal emotion_vocabulary.json so load_persona_vocabulary doesn't fail
    (d / "emotion_vocabulary.json").write_text(
        json.dumps({"version": 1, "emotions": []}),
        encoding="utf-8",
    )
    return d


@pytest.fixture(autouse=True)
def _patch_persona_dir(persona_dir: Path, monkeypatch):
    """Route get_persona_dir("nell") to our tmp persona_dir.

    persona_dir = tmp_path/personas/nell, so get_home() = tmp_path.
    """
    from brain import paths

    # persona_dir.parent = tmp_path/personas; persona_dir.parent.parent = tmp_path
    monkeypatch.setattr(paths, "get_home", lambda: persona_dir.parent.parent)


@pytest.fixture(autouse=True)
def _reset_sessions():
    from brain.chat.session import reset_registry

    reset_registry()
    yield
    reset_registry()


# ── One-shot mode ─────────────────────────────────────────────────────────────


def test_chat_one_shot_prints_response(capsys: pytest.CaptureFixture) -> None:
    """nell chat --persona nell 'hello' prints a response and exits 0."""
    exit_code = main(["chat", "--persona", "nell", "hello"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FAKE_CHAT" in captured.out


def test_chat_one_shot_exits_zero() -> None:
    exit_code = main(["chat", "--persona", "nell", "hello"])
    assert exit_code == 0


# ── Interactive REPL ──────────────────────────────────────────────────────────


def test_chat_repl_handles_exit_command(capsys: pytest.CaptureFixture) -> None:
    """Typing 'exit' closes the REPL cleanly."""
    with patch("builtins.input", side_effect=["exit"]):
        exit_code = main(["chat", "--persona", "nell"])
    assert exit_code == 0
    captured = capsys.readouterr()
    # Summary line printed
    assert "Session ended" in captured.out


def test_chat_repl_handles_eof(capsys: pytest.CaptureFixture) -> None:
    """EOF (Ctrl+D) closes the REPL cleanly."""
    with patch("builtins.input", side_effect=EOFError()):
        exit_code = main(["chat", "--persona", "nell"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Session ended" in captured.out


def test_chat_repl_produces_response_for_one_turn(capsys: pytest.CaptureFixture) -> None:
    """One turn produces a FAKE_CHAT response before exit."""
    with patch("builtins.input", side_effect=["hello there", "exit"]):
        exit_code = main(["chat", "--persona", "nell"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "FAKE_CHAT" in captured.out


# ── Missing persona dir ───────────────────────────────────────────────────────


def test_chat_missing_persona_dir_raises_file_not_found(tmp_path: Path, monkeypatch) -> None:
    """Missing persona directory raises FileNotFoundError.

    Override the autouse fixture by pointing get_home to a directory that
    has no personas/ghost subdirectory.
    """
    from brain import paths

    # Point home at empty tmp_path — personas/ghost won't exist.
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr(paths, "get_home", lambda: empty_home)
    with pytest.raises(FileNotFoundError):
        main(["chat", "--persona", "ghost"])
