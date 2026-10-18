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



# ---------------------------------------------------------------------------
# #440: a full model ID is accepted, stored and passed through as written.
# KEEP IN SYNC with app/src/lib/modelId.test.ts (the same accept/reject lists).
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from brain.persona_config import MODEL_ID_PATTERN, is_valid_model  # noqa: E402

FULL_IDS = ["claude-sonnet-5", "claude-opus-4-8", "claude-3-5-haiku-20241022", "claude-sonnet-4.5"]
BAD_IDS = [
    "gpt4",
    "Claude-Sonnet-5",
    "claude-",
    "claude-sonnet-5\n",
    " claude-sonnet-5",
    "claude-sonnet-5 ",
    "claude_sonnet_5",
    "claude-sonnet 5",
    "claude-sonnet-5;ls",
    "claude-Sonnet",
    "sonnet[1m]",
    "opusplan",
    "claude-sonnet-5[1m]",
]


@pytest.mark.parametrize("model_id", FULL_IDS)
def test_full_model_id_round_trips_verbatim(tmp_path: Path, model_id: str):
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop", "model": model_id}))
    config = PersonaConfig.load(p)
    assert config.model == model_id
    config.save(p)
    assert json.loads(p.read_text())["model"] == model_id
    assert f'"model": "{model_id}"' in p.read_text() or f'"model":"{model_id}"' in p.read_text()
    assert PersonaConfig.load(p).model == model_id


@pytest.mark.parametrize("alias", ["sonnet", "opus", "haiku"])
def test_aliases_still_round_trip(tmp_path: Path, alias: str):
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop", "model": alias}))
    config = PersonaConfig.load(p)
    config.save(p)
    assert PersonaConfig.load(p).model == alias


@pytest.mark.parametrize("bad", BAD_IDS)
def test_invalid_model_values_fall_back_with_warning(tmp_path: Path, caplog, bad: str):
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop", "model": bad}))
    with caplog.at_level(logging.WARNING):
        config = PersonaConfig.load(p)
    assert config.model == "sonnet"
    assert any("unknown model" in r.message.lower() for r in caplog.records)


@pytest.mark.parametrize("raw", ["", 5, None, ["claude-sonnet-5"]])
def test_empty_or_non_string_model_gets_the_default(tmp_path: Path, raw):
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop", "model": raw}))
    assert PersonaConfig.load(p).model == "sonnet"


@pytest.mark.parametrize("model_id", FULL_IDS + ["sonnet", "opus", "haiku"])
def test_is_valid_model_accepts(model_id: str):
    assert is_valid_model(model_id) is True


@pytest.mark.parametrize("bad", BAD_IDS + [None, 5, ""])
def test_is_valid_model_rejects(bad):
    assert is_valid_model(bad) is False


def test_pattern_uses_fullmatch_semantics():
    # `$` would accept a trailing newline; fullmatch must not.
    assert MODEL_ID_PATTERN.fullmatch("claude-sonnet-5\n") is None
