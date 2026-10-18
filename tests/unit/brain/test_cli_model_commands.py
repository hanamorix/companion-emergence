"""#441 scope addition: `nell model show | set-tier | clear-tier`. Syntax-only: no Claude CLI call is ever made."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from brain import cli
from brain.bridge.model_tier import (
    ROUTED_CLAUDE_TIERS,
    TIER_BACKGROUND_CLASSIFIER,
    TIER_COMPACTION,
    resolve_tier_model,
)


@pytest.fixture
def persona_dir(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    d = tmp_path / "personas" / "bob"
    d.mkdir(parents=True)
    return d


def _write(persona_dir: Path, **fields) -> Path:
    p = persona_dir / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop", "model": "claude-sonnet-5", "user_name": "Bob", **fields}))
    return p


def _run(argv: list[str]) -> int:
    cli._build_parser()  # warm the lazily-imported service modules before the CLI-call guards go on
    with patch("subprocess.run", side_effect=AssertionError("no CLI call allowed")), patch(
        "subprocess.Popen", side_effect=AssertionError("no CLI call allowed")
    ):
        return cli.main(["model", *argv, "--persona", "bob"])


def test_show_lists_the_chat_model_and_each_tier_with_its_source(persona_dir: Path, capsys):
    _write(persona_dir, tier_models={TIER_COMPACTION: "claude-haiku-4-5"}, derived_unavailable=["claude-haiku-5"])
    assert _run(["show"]) == 0
    out = capsys.readouterr().out
    assert "claude-sonnet-5" in out
    for tier in ROUTED_CLAUDE_TIERS:
        assert tier in out
    assert f"{TIER_COMPACTION}: claude-haiku-4-5 (override)" in out
    assert f"{TIER_BACKGROUND_CLASSIFIER}: haiku (fallback)" in out
    assert "background-generative: claude-sonnet-5 (derived)" in out
    assert "again when the chat model is re-applied in the picker" in out


def test_show_with_an_alias_chat_model_lists_todays_defaults(persona_dir: Path, capsys):
    _write(persona_dir, model="sonnet")
    assert _run(["show"]) == 0
    out = capsys.readouterr().out
    assert "attunement-detector: claude-haiku-4-5-20251001 (default)" in out
    assert "re-applied" not in out


def test_set_tier_persists_validates_and_beats_derivation(persona_dir: Path, capsys):
    path = _write(persona_dir)
    assert _run(["set-tier", TIER_COMPACTION, "claude-haiku-4-5"]) == 0
    assert "checked when first used" in capsys.readouterr().out
    saved = json.loads(path.read_text())
    assert saved["tier_models"] == {TIER_COMPACTION: "claude-haiku-4-5"}
    assert saved["user_name"] == "Bob" and saved["model"] == "claude-sonnet-5"
    assert resolve_tier_model(persona_dir, TIER_COMPACTION) == ("claude-haiku-4-5", "override")
    assert resolve_tier_model(persona_dir, TIER_BACKGROUND_CLASSIFIER)[1] == "derived"
    assert _run(["set-tier", TIER_COMPACTION, "opus"]) == 0  # an alias is fine; replaces the value
    assert json.loads(path.read_text())["tier_models"] == {TIER_COMPACTION: "opus"}


@pytest.mark.parametrize("bad", ["gpt4", "Claude-Haiku-5", "claude-", "claude-haiku-5\n", "claude-" + "a" * 58])
def test_set_tier_refuses_invalid_ids_and_leaves_the_file_alone(persona_dir: Path, capsys, bad):
    path = _write(persona_dir)
    before = path.read_bytes()
    assert _run(["set-tier", TIER_COMPACTION, bad]) != 0
    assert "claude-" in capsys.readouterr().err  # states the rule
    assert path.read_bytes() == before


def test_unknown_tier_is_rejected(persona_dir: Path):
    path = _write(persona_dir)
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        _run(["set-tier", "not-a-tier", "haiku"])
    assert path.read_bytes() == before


def test_clear_tier_removes_the_override_and_the_key_when_empty(persona_dir: Path):
    path = _write(persona_dir, tier_models={TIER_COMPACTION: "claude-haiku-4-5"})
    assert _run(["clear-tier", TIER_COMPACTION]) == 0
    saved = json.loads(path.read_text())
    assert "tier_models" not in saved
    assert resolve_tier_model(persona_dir, TIER_COMPACTION) == ("claude-haiku-5", "derived")
    assert _run(["clear-tier", TIER_COMPACTION]) == 0  # clearing nothing is fine


def test_set_tier_reads_the_config_right_before_writing(persona_dir: Path, monkeypatch):
    # a concurrent write by the bridge (here: a changed model) between launch and write must survive
    path = _write(persona_dir)
    from brain import persona_config as pc

    original = pc.PersonaConfig.load.__func__

    calls = {"n": 0}

    def _load(cls, p):
        calls["n"] += 1
        if calls["n"] == 1:  # the bridge saves a new chat model before the CLI's own load
            d = json.loads(Path(p).read_text())
            d["model"] = "claude-opus-4-8"
            Path(p).write_text(json.dumps(d))
        return original(cls, p)

    monkeypatch.setattr(pc.PersonaConfig, "load", classmethod(_load))
    assert _run(["set-tier", TIER_COMPACTION, "haiku"]) == 0
    assert json.loads(path.read_text())["model"] == "claude-opus-4-8"
