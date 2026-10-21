"""`nell init --model` writes the install-wide chat model setting (#441, #529: C12). No Claude CLI call at init.

A WORD follows the one write rule (the word that means the chat default removes the line or writes nothing, opus
writes the Opus id, haiku writes the alias); a FULL ID is always written exactly as given, even when it equals the
default. Omitted, blank or EOF writes nothing. The persona file no longer holds a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from brain import cli, tunables

_KEY = "model.interactive_chat"


def _init(monkeypatch, tmp_path: Path, model: str | None, extra: list[str] | None = None, inputs: list[str] | None = None):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    if inputs is not None:
        it = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(it))
    cli._build_parser()  # import the lazily-imported service modules first: they probe PATH at import time
    argv = ["init", "--persona", "siren", "--user-name", "Ada", "--fresh", "--voice-template", "default"]
    if model is not None:
        argv += ["--model", model]
    with patch("subprocess.run", side_effect=AssertionError("no CLI call at init")), patch(
        "subprocess.Popen", side_effect=AssertionError("no CLI call at init")
    ), patch("shutil.which", side_effect=AssertionError("init must not even look for claude")):
        return cli.main(argv + (extra or []))


def _cfg(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "personas" / "siren" / "persona_config.json").read_text())


@pytest.mark.parametrize("model_id", ["claude-sonnet-5", "claude-opus-4-8", "claude-3-5-haiku-20241022"])
def test_a_full_id_is_written_verbatim_even_when_it_equals_the_default(monkeypatch, tmp_path: Path, capsys, model_id):
    assert _init(monkeypatch, tmp_path, model_id) == 0
    assert tunables.get_override(_KEY) == model_id
    cfg = _cfg(tmp_path)
    assert "model" not in cfg and "tier_models" not in cfg and "derived_unavailable" not in cfg
    assert model_id in capsys.readouterr().out


def test_the_word_for_the_default_writes_nothing(monkeypatch, tmp_path: Path):
    assert _init(monkeypatch, tmp_path, "sonnet") == 0
    assert tunables.get_override(_KEY) is None
    assert not (tmp_path / "tunables.json").exists()


def test_the_word_for_the_default_removes_an_existing_line(monkeypatch, tmp_path: Path):
    (tmp_path).mkdir(exist_ok=True)
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables.set_overrides({_KEY: "claude-sonnet-4-6"})
    assert _init(monkeypatch, tmp_path, "sonnet") == 0
    assert tunables.get_override(_KEY) is None


@pytest.mark.parametrize(("word", "written"), [("opus", "claude-opus-5"), ("haiku", "haiku")])
def test_opus_and_haiku_write_their_meaning(monkeypatch, tmp_path: Path, word, written):
    assert _init(monkeypatch, tmp_path, word) == 0
    assert tunables.get_override(_KEY) == written


def test_the_persona_file_has_exactly_the_keys_it_had_without_a_model(monkeypatch, tmp_path: Path):
    assert _init(monkeypatch, tmp_path, "opus") == 0
    assert set(_cfg(tmp_path)) == {
        "provider", "searcher", "mcp_audit_log_level", "user_name", "last_opened_at",
        "user_pronouns", "notes_enabled", "notes_folder", "kindled_link_enabled", "kindled_relay_url",
    }


@pytest.mark.parametrize("bad", ["gpt4", "Claude-Sonnet-5", "claude-", "claude-sonnet-5\n", "claude-" + "a" * 58])
def test_invalid_model_is_refused_and_nothing_is_written(monkeypatch, tmp_path: Path, capsys, bad):
    with pytest.raises(SystemExit) as exc:  # argparse rejects it with the rule in its message
        _init(monkeypatch, tmp_path, bad)
    assert exc.value.code == 2
    assert "claude-" in capsys.readouterr().err
    assert not (tmp_path / "personas" / "siren").exists()
    assert not (tmp_path / "tunables.json").exists()


def test_omitted_blank_or_eof_writes_nothing(monkeypatch, tmp_path: Path):
    answers = ["siren", "Ada", "n", "default", ""]  # the model answer is blank
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    cli._build_parser()
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(it))
    assert cli.main(["init"]) == 0
    assert tunables.get_override(_KEY) is None and not (tmp_path / "tunables.json").exists()


def test_the_interactive_prompt_shows_the_current_effective_chat_id(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables.set_overrides({_KEY: "claude-sonnet-4-6"})
    cli._build_parser()
    seen: list[str] = []
    answers = iter(["siren", "Ada", "n", "default", ""])

    def fake_input(prompt=""):
        seen.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.main(["init"]) == 0
    assert any("blank keeps claude-sonnet-4-6" in p for p in seen)


def test_interactive_prompt_accepts_a_typed_full_id(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    cli._build_parser()
    answers = iter(["siren", "Ada", "n", "default", "bogus!", "claude-sonnet-5"])  # the first model answer is invalid
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))
    with patch("subprocess.run", side_effect=AssertionError), patch("subprocess.Popen", side_effect=AssertionError):
        assert cli.main(["init"]) == 0
    assert tunables.get_override(_KEY) == "claude-sonnet-5"


def test_a_settings_file_that_cannot_be_changed_is_reported_and_exits_non_zero(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    (tmp_path / "tunables.json").write_text("{not json")
    assert _init(monkeypatch, tmp_path, "opus") == 1
    assert "could not be saved" in capsys.readouterr().err
    assert (tmp_path / "tunables.json").read_text() == "{not json"
