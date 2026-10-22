"""`nell init --model` writes the install-wide chat model setting (#441, #529: C12). No Claude CLI call at init.

The pick is WRITTEN as the entry's value (sonnet is the chat default id, opus the Opus id, haiku the alias, a full id exactly
as given); nothing is ever removed, and a value that already equals the pick writes nothing. A setting equal to the shipped
default follows later default changes (owner decision). Omitted, blank or EOF writes nothing. The persona file no longer
holds a model.
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


@pytest.mark.parametrize("model_id", ["claude-opus-4-8", "claude-3-5-haiku-20241022"])
def test_a_full_id_is_written_verbatim(monkeypatch, tmp_path: Path, capsys, model_id):
    assert _init(monkeypatch, tmp_path, model_id) == 0
    assert tunables.get_raw(_KEY) == model_id
    cfg = _cfg(tmp_path)
    assert "model" not in cfg and "tier_models" not in cfg and "derived_unavailable" not in cfg
    assert model_id in capsys.readouterr().out


def test_the_word_for_the_default_writes_nothing_when_the_value_already_is_the_default(monkeypatch, tmp_path: Path):
    assert _init(monkeypatch, tmp_path, "sonnet") == 0
    assert tunables.get_raw(_KEY) is None  # no entry yet: the boot pass creates it with the default
    assert not (tmp_path / "tunables.toml").exists()


def test_a_full_id_equal_to_the_default_writes_nothing_and_the_chat_runs_it(monkeypatch, tmp_path: Path, capsys):
    from brain.bridge.model_tier import TIER_INTERACTIVE_CHAT, model_for_tier

    assert _init(monkeypatch, tmp_path, "claude-sonnet-5") == 0
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == "claude-sonnet-5"
    assert not (tmp_path / "tunables.toml").exists()  # equal to the default: nothing to write, it follows the default
    assert "claude-sonnet-5" in capsys.readouterr().out


def test_the_word_for_the_default_writes_the_default_id_over_another_value_and_keeps_the_entry(monkeypatch, tmp_path: Path):
    (tmp_path).mkdir(exist_ok=True)
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables.set_values({_KEY: "claude-sonnet-4-6"})
    assert _init(monkeypatch, tmp_path, "sonnet") == 0
    assert tunables.get_raw(_KEY) == "claude-sonnet-5"  # the entry stays and holds the id the word means


@pytest.mark.parametrize(("word", "written"), [("opus", "claude-opus-5"), ("haiku", "haiku")])
def test_opus_and_haiku_write_their_meaning(monkeypatch, tmp_path: Path, word, written):
    assert _init(monkeypatch, tmp_path, word) == 0
    assert tunables.get_raw(_KEY) == written


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
    assert not (tmp_path / "tunables.toml").exists()


def test_omitted_blank_or_eof_writes_nothing(monkeypatch, tmp_path: Path):
    answers = ["siren", "Ada", "n", "default", ""]  # the model answer is blank
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    cli._build_parser()
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(it))
    assert cli.main(["init"]) == 0
    assert tunables.get_raw(_KEY) is None and not (tmp_path / "tunables.toml").exists()


def test_the_interactive_prompt_shows_the_current_effective_chat_id(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables.set_values({_KEY: "claude-sonnet-4-6"})
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
    answers = iter(["siren", "Ada", "n", "default", "bogus!", "claude-opus-4-8"])  # the first model answer is invalid
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))
    with patch("subprocess.run", side_effect=AssertionError), patch("subprocess.Popen", side_effect=AssertionError):
        assert cli.main(["init"]) == 0
    assert tunables.get_raw(_KEY) == "claude-opus-4-8"


def test_a_settings_file_that_cannot_be_changed_is_reported_and_exits_non_zero(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    (tmp_path / "tunables.toml").write_text("a = = 1\n")
    assert _init(monkeypatch, tmp_path, "opus") == 1
    err = capsys.readouterr().err
    assert "could not be saved" in err and "line 1" in err  # the reason names the failing line
    assert (tmp_path / "tunables.toml").read_text() == "a = = 1\n"
