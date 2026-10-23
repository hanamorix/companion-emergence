"""#446 - a model setting this app does not support (for example ``fable``) is NOT used and is reported, never erased.

Since #529 the chat model is the ``model.interactive_chat`` line of the tunables file. A value that fails the model
grammar runs the shipped default with one warning, the owner's text stays in the file untouched (the runtime never
edits it), and `/persona/state` and `nell model show` say what text is being replaced. Real files, the real bridge
routes and the real `brain.cli` entry; the Claude CLI is never called.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from brain import cli, tunables
from brain.bridge.model_tier import TIER_INTERACTIVE_CHAT, model_for_tier
from brain.bridge.persona_state import build_persona_state
from brain.utils import log_throttle

_KEY = "model.interactive_chat"
BAD_VALUES = ["fable", "sonnet[1m]", "Sonnet", "claude-", "gpt4", " claude-sonnet-5", "claude-sonnet-5\n"]
GOOD_VALUES = ["sonnet", "opus", "haiku", "claude-sonnet-5", "claude-3-5-haiku-20241022"]


@pytest.fixture(autouse=True)
def _fresh_throttle():
    log_throttle._last.clear()
    yield
    log_throttle._last.clear()


def _persona(tmp: Path, provider="claude-cli") -> Path:
    d = tmp / "persona"
    d.mkdir(exist_ok=True)
    (d / "persona_config.json").write_text(json.dumps({"provider": provider, "searcher": "noop"}))
    return d


def _raw() -> bytes:
    return tunables._file_path().read_bytes()


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_an_unsupported_value_runs_the_default_and_is_remembered_in_the_file(bad: str, caplog):
    tunables.set_values({_KEY: bad})
    before = _raw()
    with caplog.at_level(logging.WARNING):
        assert model_for_tier(TIER_INTERACTIVE_CHAT) == "claude-sonnet-5"
    assert any("not a usable model" in r.getMessage() for r in caplog.records)
    assert _raw() == before  # the runtime never erases the owner's text


@pytest.mark.parametrize("good", GOOD_VALUES)
def test_a_supported_value_has_no_unsupported_marker(good: str, tmp_path: Path, caplog):
    tunables.set_values({_KEY: good})
    with caplog.at_level(logging.WARNING):
        model_for_tier(TIER_INTERACTIVE_CHAT)
    assert not [r for r in caplog.records if "not a usable model" in r.getMessage()]
    assert build_persona_state(_persona(tmp_path))["model_unsupported"] is None


def test_an_empty_or_non_string_value_runs_the_default():
    for raw in ("", 5, None, ["x"]):
        tunables.set_values({_KEY: raw})
        assert model_for_tier(TIER_INTERACTIVE_CHAT) == "claude-sonnet-5"


def test_the_warning_is_logged_once_per_value_not_per_call(caplog):
    tunables.set_values({_KEY: "fable"})
    with caplog.at_level(logging.WARNING):
        for _ in range(10):
            model_for_tier(TIER_INTERACTIVE_CHAT)
    assert len([r for r in caplog.records if "not a usable model" in r.getMessage()]) == 1


def test_the_bridge_startup_sequence_keeps_the_raw_value_on_disk(tmp_path: Path):
    from fastapi.testclient import TestClient

    from brain.bridge.server import build_app

    d = _persona(tmp_path)
    tunables.set_values({_KEY: "fable"})
    with TestClient(build_app(persona_dir=d, client_origin="tests")):
        pass  # the lifespan runs the settings boot pass and touches the persona file
    assert tunables.get_raw(_KEY) == "fable"


def test_a_writer_of_another_key_keeps_the_raw_value():
    tunables.set_values({_KEY: "fable"})
    tunables.set_values({"model.compaction": "claude-haiku-4-5"})
    assert tunables.get_raw(_KEY) == "fable"


def test_nell_model_set_tier_and_clear_tier_of_another_job_keep_the_raw_value(monkeypatch):
    tunables.set_values({_KEY: "fable"})
    cli._build_parser()
    assert cli.main(["model", "set-tier", "compaction", "claude-haiku-4-5"]) == 0
    assert cli.main(["model", "clear-tier", "compaction"]) == 0
    assert tunables.get_raw(_KEY) == "fable"


@pytest.mark.parametrize("applied", ["opus", "claude-sonnet-5", "sonnet"])
def test_a_new_choice_replaces_the_unsupported_text(applied: str, tmp_path: Path):
    import asyncio

    from brain.bridge import server

    tunables.set_values({_KEY: "fable"})
    status, _body = asyncio.run(server._apply_chat_model(applied, check=False, lock=asyncio.Lock()))
    assert status == 200
    assert tunables.get_raw(_KEY) in (None, "claude-opus-5", "claude-sonnet-5")  # never "fable" any more
    assert build_persona_state(_persona(tmp_path))["model_unsupported"] is None


def test_nell_init_with_a_model_replaces_the_unsupported_text(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables.set_values({_KEY: "fable"})
    cli._build_parser()
    argv = ["init", "--persona", "bob", "--user-name", "Ada", "--fresh", "--voice-template", "default", "--model", "sonnet"]
    with patch("subprocess.run", side_effect=AssertionError("no CLI call")), patch(
        "subprocess.Popen", side_effect=AssertionError("no CLI call")
    ):
        assert cli.main(argv) == 0
    assert tunables.get_raw(_KEY) == "claude-sonnet-5"  # the word writes the default id over the unsupported text


def test_nell_init_without_a_model_keeps_the_raw_value(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables.set_values({_KEY: "fable"})
    cli._build_parser()
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    assert cli.main(["init", "--persona", "bob", "--user-name", "Ada", "--fresh", "--voice-template", "default"]) == 0
    assert tunables.get_raw(_KEY) == "fable"


def test_hand_fixing_the_file_clears_the_warning(tmp_path: Path):
    d = _persona(tmp_path)
    tunables.set_values({_KEY: "fable"})
    assert build_persona_state(d)["model_unsupported"] == "fable"
    tunables.set_values({_KEY: "claude-opus-4-9"})
    assert build_persona_state(d)["model_unsupported"] is None


# --- /persona/state -------------------------------------------------------------------------------------


def test_state_reports_the_unsupported_value_for_claude_cli(tmp_path: Path):
    tunables.set_values({_KEY: "fable"})
    state = build_persona_state(_persona(tmp_path))
    assert state["model_unsupported"] == "fable"
    assert state["connection"]["model"] == "claude-sonnet-5"


def test_state_is_null_for_supported_and_for_other_providers(tmp_path: Path):
    tunables.set_values({_KEY: "opus"})
    assert build_persona_state(_persona(tmp_path))["model_unsupported"] is None
    tunables.set_values({_KEY: "fable"})
    for prov in ("ollama", "fake"):
        assert build_persona_state(_persona(tmp_path, provider=prov))["model_unsupported"] is None


def test_state_display_value_is_cut_and_cleaned(tmp_path: Path):
    d = _persona(tmp_path)
    tunables.set_values({_KEY: "x" * 200})
    assert build_persona_state(d)["model_unsupported"] == "x" * 64 + "…"
    tunables.set_values({_KEY: "bad\x1b[31m\nvalue"})
    shown = build_persona_state(d)["model_unsupported"]
    assert shown == "bad[31mvalue" and "\x1b" not in shown and "\n" not in shown


def test_state_display_of_a_value_with_only_non_printing_characters_is_not_empty(tmp_path: Path):
    tunables.set_values({_KEY: "​\x07"})
    assert build_persona_state(_persona(tmp_path))["model_unsupported"] == "(non-printing characters)"


def test_state_keeps_the_keys_the_app_reads(tmp_path: Path):
    keys = set(build_persona_state(_persona(tmp_path)))
    for k in ("persona", "emotions", "body", "interior", "connection", "model_rejections", "model_fallbacks", "model_unconfirmed"):
        assert k in keys


# --- nell model show ------------------------------------------------------------------------------------


def test_show_warns_for_an_unsupported_value(capsys):
    tunables.set_values({_KEY: "fable"})
    cli._build_parser()
    assert cli.main(["model", "show"]) == 0
    out = capsys.readouterr().out
    warn = [ln for ln in out.splitlines() if ln.startswith("warning:")]
    assert len(warn) == 1
    assert "fable" in warn[0] and "not a usable model" in warn[0] and "claude-sonnet-5" in warn[0]
    assert "—" not in warn[0]


def test_show_is_silent_for_supported_values(capsys):
    tunables.set_values({_KEY: "opus"})
    cli._build_parser()
    assert cli.main(["model", "show"]) == 0
    assert "warning:" not in capsys.readouterr().out
