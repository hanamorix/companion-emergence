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
# KEEP IN SYNC with app/src/lib/modelId.test.ts (the same accept/reject lists, incl. the 64-character cap).
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from brain.persona_config import MODEL_ID_PATTERN, is_valid_model  # noqa: E402

FULL_IDS = [
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4.5",
    "claude-" + "a" * 57,  # exactly 64 characters: the longest accepted id
]
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
    "claude-" + "a" * 58,  # 65 characters: one over the cap
    "claude-" + "a" * 5000,
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


def test_length_cap_is_64_characters():
    from brain.persona_config import MODEL_ID_MAX_LENGTH

    assert MODEL_ID_MAX_LENGTH == 64
    assert len("claude-" + "a" * 57) == 64
    assert is_valid_model("claude-" + "a" * 57) is True
    assert is_valid_model("claude-" + "a" * 58) is False


# ---------------------------------------------------------------------------
# #441 scope addition: tier_models / derived_unavailable (C25)
# ---------------------------------------------------------------------------

OLD_KEYS = {
    "provider", "searcher", "mcp_audit_log_level", "user_name", "model", "last_opened_at",
    "user_pronouns", "notes_enabled", "notes_folder", "kindled_link_enabled", "kindled_relay_url",
}


def _load(tmp_path: Path, **extra) -> PersonaConfig:
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "searcher": "noop", "model": "claude-sonnet-5", **extra}))
    return PersonaConfig.load(p)


def test_new_keys_are_written_only_when_non_empty(tmp_path: Path):
    p = tmp_path / "persona_config.json"
    PersonaConfig().save(p)
    assert set(json.loads(p.read_text())) == OLD_KEYS  # byte-identical key set for a default / alias config
    cfg = PersonaConfig(model="claude-sonnet-5", tier_models={"compaction": "claude-haiku-4-5"}, derived_unavailable=["claude-haiku-5"])
    cfg.save(p)
    saved = json.loads(p.read_text())
    assert saved["tier_models"] == {"compaction": "claude-haiku-4-5"}
    assert saved["derived_unavailable"] == ["claude-haiku-5"]
    again = PersonaConfig.load(p)
    assert again.tier_models == {"compaction": "claude-haiku-4-5"} and again.derived_unavailable == ["claude-haiku-5"]
    again.tier_models = {}
    again.derived_unavailable = []
    again.save(p)
    assert set(json.loads(p.read_text())) == OLD_KEYS  # emptied again: keys vanish


def test_old_files_load_with_empty_defaults(tmp_path: Path):
    cfg = _load(tmp_path)
    assert cfg.tier_models == {} and cfg.derived_unavailable == []
    assert PersonaConfig().tier_models == {} and PersonaConfig().derived_unavailable == []


def test_bad_tier_entries_are_dropped_with_one_warning_per_value(tmp_path: Path, caplog, monkeypatch):
    import brain.persona_config as pc

    monkeypatch.setattr(pc, "_warned_bad_values", set())
    bad = {
        "compaction": "gpt4",  # invalid model
        "not-a-tier": "haiku",  # unknown tier
        "background-classifier": "claude-" + "a" * 58,  # over the cap
        "background-generative": "claude-haiku-4-5",  # good
        "self-model-articulate": 5,  # wrong type
    }
    with caplog.at_level(logging.WARNING):
        cfg = _load(tmp_path, tier_models=bad)
        first = len(caplog.records)
        _load(tmp_path, tier_models=bad)  # loaded again (per background call): no new warnings
    assert cfg.tier_models == {"background-generative": "claude-haiku-4-5"}
    assert first >= 4
    assert len(caplog.records) == first


@pytest.mark.parametrize("raw", ["x", 5, ["a"], None])
def test_non_dict_tier_models_become_empty(tmp_path: Path, raw):
    assert _load(tmp_path, tier_models=raw).tier_models == {}


def test_derived_unavailable_keeps_only_valid_unique_full_ids(tmp_path: Path):
    cfg = _load(tmp_path, derived_unavailable=["claude-haiku-5", "claude-haiku-5", "haiku", "Bad Id", 7, "claude-" + "a" * 58])
    assert cfg.derived_unavailable == ["claude-haiku-5"]
    assert _load(tmp_path, derived_unavailable="claude-haiku-5").derived_unavailable == []


def test_other_fields_survive_a_save_with_the_new_keys(tmp_path: Path):
    p = tmp_path / "persona_config.json"
    cfg = _load(tmp_path, user_name="Bob", notes_enabled=True)
    cfg.tier_models = {"compaction": "haiku"}
    cfg.save(p)
    back = PersonaConfig.load(p)
    assert back.user_name == "Bob" and back.notes_enabled is True and back.model == "claude-sonnet-5"
