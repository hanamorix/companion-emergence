"""Wizard model prompt — _prompt_choice helper + model persisted to persona_config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain import cli


def _run_init(monkeypatch, tmp_path: Path, argv: list[str], inputs: list[str] | None = None) -> int:
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    if inputs is not None:
        it = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(it))
    return cli.main(["init", *argv])


# ---------------------------------------------------------------------------
# _prompt_choice unit tests
# ---------------------------------------------------------------------------


def test_prompt_choice_returns_valid_choice(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "opus")
    result = cli._prompt_choice("model", choices=["sonnet", "opus", "haiku"], default="sonnet")
    assert result == "opus"


def test_prompt_choice_accepts_default_on_empty(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "")
    result = cli._prompt_choice("model", choices=["sonnet", "opus", "haiku"], default="sonnet")
    assert result == "sonnet"


def test_prompt_choice_retries_on_invalid_then_accepts(monkeypatch):
    answers = iter(["gpt4", "opus"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))
    result = cli._prompt_choice("model", choices=["sonnet", "opus", "haiku"], default="sonnet")
    assert result == "opus"


def test_prompt_choice_eof_returns_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **kw: (_ for _ in ()).throw(EOFError))
    result = cli._prompt_choice("model", choices=["sonnet", "opus", "haiku"], default="sonnet")
    assert result == "sonnet"


# ---------------------------------------------------------------------------
# Wizard integration: model flag → persona_config.json
# ---------------------------------------------------------------------------


def test_init_flag_model_opus_persists(monkeypatch, tmp_path: Path):
    """--model opus writes model=opus to persona_config.json."""
    rc = _run_init(
        monkeypatch,
        tmp_path,
        [
            "--persona",
            "mira",
            "--user-name",
            "Hana",
            "--fresh",
            "--voice-template",
            "default",
            "--model",
            "opus",
        ],
    )
    assert rc == 0
    cfg = json.loads((tmp_path / "personas" / "mira" / "persona_config.json").read_text())
    assert cfg["model"] == "opus"


def test_init_interactive_model_haiku(monkeypatch, tmp_path: Path):
    """Interactive wizard: user picks haiku → persisted."""
    inputs = [
        "mira",  # persona name
        "Hana",  # user name
        "n",  # migrate? no
        "default",  # voice template
        "haiku",  # model
    ]
    rc = _run_init(monkeypatch, tmp_path, [], inputs=inputs)
    assert rc == 0
    cfg = json.loads((tmp_path / "personas" / "mira" / "persona_config.json").read_text())
    assert cfg["model"] == "haiku"


def test_init_flag_model_unknown_rejected_by_argparse(monkeypatch, tmp_path: Path, capsys):
    """argparse rejects unknown --model values at parse time."""
    with pytest.raises(SystemExit) as excinfo:
        _run_init(
            monkeypatch,
            tmp_path,
            [
                "--persona",
                "mira",
                "--user-name",
                "Hana",
                "--fresh",
                "--voice-template",
                "default",
                "--model",
                "gpt4",
            ],
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "gpt4" in err
