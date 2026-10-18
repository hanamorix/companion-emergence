"""#446 - a model value this app does not support (for example ``fable``) falls back to sonnet WITH a visible warning.

Real PersonaConfig files, the real bridge routes and the real ``brain.cli`` entry; the Claude CLI is never called.
The owner's raw value stays in the file (``save()`` writes it back while the config still holds the fallback), so
the warning can be derived from the file by any process (the bridge's own startup load -> save would otherwise
erase it). Every writer that SETS the model goes through ``PersonaConfig.set_model``, which clears it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain import cli
from brain import persona_config as pc
from brain.bridge.persona_state import build_persona_state
from brain.bridge.server import build_app
from brain.persona_config import PersonaConfig
from brain.setup import write_persona_config

BAD_VALUES = ["fable", "sonnet[1m]", "Sonnet", "claude-", "gpt4", " claude-sonnet-5", "claude-sonnet-5\n"]
GOOD_VALUES = ["sonnet", "opus", "haiku", "claude-sonnet-5", "claude-3-5-haiku-20241022"]


@pytest.fixture(autouse=True)
def _fresh_warning_set(monkeypatch):
    monkeypatch.setattr(pc, "_warned_bad_values", set())


def _write(tmp: Path, model="fable", provider="claude-cli", **extra) -> Path:
    p = tmp / "persona_config.json"
    p.write_text(json.dumps({"provider": provider, "searcher": "noop", "model": model, **extra}))
    return p


def _file(p: Path) -> dict:
    return json.loads(p.read_text())


# --- MA1: parse ------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_unsupported_value_falls_back_and_is_remembered(tmp_path: Path, caplog, bad: str):
    p = _write(tmp_path, bad)
    with caplog.at_level(logging.WARNING):
        cfg = PersonaConfig.load(p)
    assert cfg.model == "sonnet"
    assert cfg.model_unsupported == bad
    assert any("unknown model" in r.message.lower() and repr(bad) in r.message for r in caplog.records)


@pytest.mark.parametrize("good", GOOD_VALUES)
def test_supported_value_has_no_unsupported_marker(tmp_path: Path, caplog, good: str):
    p = _write(tmp_path, good)
    with caplog.at_level(logging.WARNING):
        cfg = PersonaConfig.load(p)
    assert cfg.model == good and cfg.model_unsupported is None
    assert not [r for r in caplog.records if "unknown model" in r.message.lower()]


@pytest.mark.parametrize("raw", ["", 5, None, ["claude-sonnet-5"]])
def test_empty_or_non_string_model_is_unchanged_behaviour(tmp_path: Path, caplog, raw):
    p = _write(tmp_path, raw)
    with caplog.at_level(logging.WARNING):
        cfg = PersonaConfig.load(p)
    assert cfg.model == "sonnet" and cfg.model_unsupported is None
    assert not [r for r in caplog.records if "unknown model" in r.message.lower()]


def test_absent_model_key_is_unchanged_behaviour(tmp_path: Path):
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop"}))
    cfg = PersonaConfig.load(p)
    assert cfg.model == "sonnet" and cfg.model_unsupported is None


# --- MA2: once per value ---------------------------------------------------------------------------


def test_warning_is_logged_once_per_value_not_per_load(tmp_path: Path, caplog):
    p = _write(tmp_path, "fable")
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            PersonaConfig.load(p)
    assert len([r for r in caplog.records if "unknown model" in r.message.lower()]) == 1
    p2 = _write(tmp_path, "gpt4")
    with caplog.at_level(logging.WARNING):
        PersonaConfig.load(p2)
        PersonaConfig.load(p2)
    assert len([r for r in caplog.records if "unknown model" in r.message.lower()]) == 2


# --- MA3: other writers keep the owner's value ------------------------------------------------------


def test_bridge_startup_sequence_keeps_the_raw_value_on_disk(tmp_path: Path):
    p = _write(tmp_path, "fable", user_name="Bob")
    cfg = PersonaConfig.load(p)  # server.py lifespan: load -> touch_last_opened -> save
    cfg.touch_last_opened()
    cfg.save(p)
    saved = _file(p)
    assert saved["model"] == "fable" and saved["user_name"] == "Bob" and saved["last_opened_at"]
    again = PersonaConfig.load(p)
    assert again.model == "sonnet" and again.model_unsupported == "fable"


def test_a_writer_that_changes_another_field_keeps_the_raw_value(tmp_path: Path):
    p = _write(tmp_path, "fable")
    replace(PersonaConfig.load(p), user_name="Bob").save(p)
    assert _file(p)["model"] == "fable"
    assert PersonaConfig.load(p).model_unsupported == "fable"


def test_nell_model_set_tier_and_clear_tier_keep_the_raw_value(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    d = tmp_path / "personas" / "bob"
    d.mkdir(parents=True)
    p = _write(d, "fable")
    cli._build_parser()
    with patch("subprocess.run", side_effect=AssertionError("no CLI call")):
        assert cli.main(["model", "set-tier", "compaction", "claude-haiku-4-5", "--persona", "bob"]) == 0
        assert _file(p)["model"] == "fable"
        assert cli.main(["model", "clear-tier", "compaction", "--persona", "bob"]) == 0
    assert _file(p)["model"] == "fable"


# --- MA4: every writer that SETS the model clears it ----------------------------------------------------


def test_set_model_helper_sets_and_clears(tmp_path: Path):
    cfg = PersonaConfig.load(_write(tmp_path, "fable"))
    assert cfg.model_unsupported == "fable"
    cfg.set_model("opus")
    assert cfg.model == "opus" and cfg.model_unsupported is None


def _client(persona_dir: Path) -> TestClient:
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests"))


@pytest.mark.parametrize("applied", ["opus", "claude-sonnet-5", "sonnet"])
def test_endpoint_apply_clears_the_marker_including_the_fallback_value_itself(persona_dir: Path, applied: str):
    p = _write(persona_dir, "fable", provider="fake", searcher="fake")
    with _client(persona_dir) as c:
        assert c.get("/persona/state").json()["model_unsupported"] is None  # provider is not claude-cli (m1)
        r = c.post("/persona/config/model", json={"model": applied})
        assert r.status_code == 200, r.text
    assert _file(p)["model"] == applied
    assert PersonaConfig.load(p).model_unsupported is None


@pytest.mark.parametrize("applied", ["opus", "sonnet"])
def test_nell_init_writer_clears_the_marker(tmp_path: Path, applied: str):
    p = _write(tmp_path, "fable")
    write_persona_config(tmp_path, user_name="Bob", model=applied)
    assert _file(p)["model"] == applied
    assert PersonaConfig.load(p).model_unsupported is None


def test_nell_init_without_a_model_keeps_the_raw_value(tmp_path: Path):
    p = _write(tmp_path, "fable")
    write_persona_config(tmp_path, user_name="Bob")
    assert _file(p)["model"] == "fable"


def test_nell_init_cli_with_model_sonnet_clears_the_marker(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    d = tmp_path / "personas" / "siren"
    d.mkdir(parents=True)
    p = _write(d, "fable")
    cli._build_parser()
    argv = ["init", "--persona", "siren", "--user-name", "Hana", "--fresh", "--voice-template", "default", "--model", "sonnet"]
    with patch("subprocess.run", side_effect=AssertionError("no CLI call at init")), patch(
        "subprocess.Popen", side_effect=AssertionError("no CLI call at init")
    ):
        rc = cli.main(argv)
        if rc != 0:  # an existing persona needs --force
            rc = cli.main([*argv, "--force"])
    assert rc == 0
    assert _file(p)["model"] == "sonnet"
    assert PersonaConfig.load(p).model_unsupported is None


def test_hand_fixing_the_file_clears_the_warning(tmp_path: Path, caplog):
    p = _write(tmp_path, "fable")
    p.write_text(json.dumps({**_file(p), "model": "opus"}))
    with caplog.at_level(logging.WARNING):
        cfg = PersonaConfig.load(p)
    assert cfg.model == "opus" and cfg.model_unsupported is None
    assert not [r for r in caplog.records if "unknown model" in r.message.lower()]


# --- MA5: byte identity for supported configs ------------------------------------------------------------

# LITERAL golden payloads (the key set and values `save()` wrote before this change), not a re-implementation.
_GOLDEN_KEYS = {
    "provider", "searcher", "mcp_audit_log_level", "user_name", "model", "last_opened_at",
    "user_pronouns", "notes_enabled", "notes_folder", "kindled_link_enabled", "kindled_relay_url",
}


@pytest.mark.parametrize("good", GOOD_VALUES)
def test_supported_configs_save_exactly_todays_payload(tmp_path: Path, good: str):
    p = _write(tmp_path, good, user_name="Bob")
    PersonaConfig.load(p).save(p)
    saved = _file(p)
    assert saved == {
        "provider": "claude-cli",
        "searcher": "noop",
        "mcp_audit_log_level": pc.DEFAULT_MCP_AUDIT_LOG_LEVEL,
        "user_name": "Bob",
        "model": good,
        "last_opened_at": None,
        "user_pronouns": None,
        "notes_enabled": False,
        "notes_folder": None,
        "kindled_link_enabled": False,
        "kindled_relay_url": None,
    }
    assert set(saved) == _GOLDEN_KEYS
    assert "model_unsupported" not in saved


# --- MA6: /persona/state ------------------------------------------------------------------------------


def test_state_reports_the_unsupported_value_for_claude_cli(tmp_path: Path):
    _write(tmp_path, "fable")
    state = build_persona_state(tmp_path)
    assert state["model_unsupported"] == "fable"
    assert state["connection"]["model"] == "sonnet"


def test_state_is_null_for_supported_and_for_other_providers(tmp_path: Path):
    _write(tmp_path, "opus")
    assert build_persona_state(tmp_path)["model_unsupported"] is None
    for prov in ("ollama", "fake"):
        _write(tmp_path, "fable", provider=prov)
        assert build_persona_state(tmp_path)["model_unsupported"] is None


def test_state_display_value_is_cut_and_cleaned(tmp_path: Path):
    _write(tmp_path, "x" * 200)
    assert build_persona_state(tmp_path)["model_unsupported"] == "x" * 64 + "…"
    _write(tmp_path, "bad\x1b[31m\nvalue")
    shown = build_persona_state(tmp_path)["model_unsupported"]
    assert shown == "bad[31mvalue" and "\x1b" not in shown and "\n" not in shown


def test_state_keeps_every_previous_key(tmp_path: Path):
    _write(tmp_path, "opus")
    keys = set(build_persona_state(tmp_path))
    for k in ("persona", "emotions", "body", "interior", "connection", "model_rejections", "tier_fallbacks", "tier_unconfirmed", "model_unconfirmed"):
        assert k in keys


# --- MA7: nell model show ---------------------------------------------------------------------------------


def _show(monkeypatch, tmp_path: Path, model, provider="claude-cli", capsys=None):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    d = tmp_path / "personas" / "bob"
    d.mkdir(parents=True, exist_ok=True)
    _write(d, model, provider=provider)
    cli._build_parser()
    with patch("subprocess.run", side_effect=AssertionError("no CLI call")), patch(
        "subprocess.Popen", side_effect=AssertionError("no CLI call")
    ):
        assert cli.main(["model", "show", "--persona", "bob"]) == 0
    return capsys.readouterr().out


def test_show_warns_for_an_unsupported_value(monkeypatch, tmp_path: Path, capsys):
    out = _show(monkeypatch, tmp_path, "fable", capsys=capsys)
    warn = [ln for ln in out.splitlines() if ln.startswith("warning:")]
    assert len(warn) == 1
    assert "fable" in warn[0] and "not supported" in warn[0] and "sonnet" in warn[0]
    assert "chat model: sonnet" in out and "background models" in out
    assert "—" not in warn[0]


def test_show_is_silent_for_supported_and_for_other_providers(monkeypatch, tmp_path: Path, capsys):
    assert "warning:" not in _show(monkeypatch, tmp_path, "opus", capsys=capsys)
    assert "warning:" not in _show(monkeypatch, tmp_path, "fable", provider="ollama", capsys=capsys)


def test_show_warns_after_the_bridge_started(monkeypatch, tmp_path: Path, capsys):
    """The file still says fable after a real bridge start, so a separate process still sees the warning."""
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    d = tmp_path / "personas" / "bob"
    d.mkdir(parents=True)
    _write(d, "fable", provider="claude-cli")
    cfg = PersonaConfig.load(d / "persona_config.json")
    cfg.touch_last_opened()
    cfg.save(d / "persona_config.json")
    cli._build_parser()
    assert cli.main(["model", "show", "--persona", "bob"]) == 0
    assert "warning:" in capsys.readouterr().out
