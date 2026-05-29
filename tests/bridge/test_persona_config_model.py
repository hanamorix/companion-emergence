"""PersonaConfig.model — allowlist with friendly fallback."""

import json
import logging
from pathlib import Path

from brain.persona_config import DEFAULT_MODEL, KNOWN_MODELS, PersonaConfig


def test_default_model_is_sonnet():
    assert DEFAULT_MODEL == "sonnet"
    assert "sonnet" in KNOWN_MODELS
    assert "opus" in KNOWN_MODELS
    assert "haiku" in KNOWN_MODELS


def test_load_missing_field_falls_to_default(tmp_path: Path):
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop"}))
    config = PersonaConfig.load(p)
    assert config.model == "sonnet"


def test_load_round_trips_opus(tmp_path: Path):
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop", "model": "opus"}))
    config = PersonaConfig.load(p)
    assert config.model == "opus"


def test_load_unknown_model_falls_to_default(tmp_path: Path, caplog):
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop", "model": "gpt4"}))
    with caplog.at_level(logging.WARNING):
        config = PersonaConfig.load(p)
    assert config.model == "sonnet"
    assert any("unknown model" in r.message.lower() for r in caplog.records)


def test_save_round_trips_model(tmp_path: Path):
    """Save then reload preserves the model field."""
    from dataclasses import replace

    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop", "model": "opus"}))
    config = PersonaConfig.load(p)
    assert config.model == "opus"
    updated = replace(config, model="haiku")
    updated.save(p)
    reloaded = PersonaConfig.load(p)
    assert reloaded.model == "haiku"


def test_thinking_budget_tokens_default_is_none(tmp_path: Path):
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop"}))
    config = PersonaConfig.load(p)
    assert config.thinking_budget_tokens is None


def test_thinking_budget_tokens_round_trips(tmp_path: Path):
    from dataclasses import replace
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop"}))
    config = PersonaConfig.load(p)
    updated = replace(config, thinking_budget_tokens=8000)
    updated.save(p)
    reloaded = PersonaConfig.load(p)
    assert reloaded.thinking_budget_tokens == 8000


def test_thinking_budget_tokens_none_persists(tmp_path: Path):
    from dataclasses import replace
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop", "thinking_budget_tokens": 8000}))
    config = PersonaConfig.load(p)
    assert config.thinking_budget_tokens == 8000
    updated = replace(config, thinking_budget_tokens=None)
    updated.save(p)
    reloaded = PersonaConfig.load(p)
    assert reloaded.thinking_budget_tokens is None


def test_thinking_budget_tokens_invalid_values_heal_to_none(tmp_path: Path):
    """Zero, negative, and non-int values all fall back to None."""
    for bad in [0, -1, "lots", True]:
        p = tmp_path / f"config_{bad}.json"
        p.write_text(json.dumps({"thinking_budget_tokens": bad}))
        config = PersonaConfig.load(p)
        assert config.thinking_budget_tokens is None, f"bad value {bad!r} should heal to None"
