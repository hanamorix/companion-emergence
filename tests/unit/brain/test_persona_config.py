"""Tests for brain.persona_config — provider + searcher routing file."""

from __future__ import annotations

from pathlib import Path

from brain.persona_config import DEFAULT_PROVIDER, DEFAULT_SEARCHER, PersonaConfig

# ---- Task 9: attempt_heal wiring ----


def test_persona_config_load_corrupt_file_quarantines_and_resets(tmp_path: Path) -> None:
    """Corrupt JSON → defaults returned + quarantine file present, original gone."""
    path = tmp_path / "persona_config.json"
    path.write_text("{this is not json", encoding="utf-8")

    cfg, anomaly = PersonaConfig.load_with_anomaly(path)

    assert cfg.provider == DEFAULT_PROVIDER
    assert cfg.searcher == DEFAULT_SEARCHER
    assert anomaly is not None
    assert anomaly.kind == "json_parse_error"
    # Original replaced by a quarantine file
    assert not path.exists() or path.read_text().strip().startswith("{")  # reset default written
    corrupt_files = list(tmp_path.glob("persona_config.json.corrupt-*"))
    assert len(corrupt_files) == 1


def test_persona_config_load_corrupt_file_restores_from_bak(tmp_path: Path) -> None:
    """A valid .bak1 + corrupt live file → .bak1 content returned."""
    path = tmp_path / "persona_config.json"
    bak1 = tmp_path / "persona_config.json.bak1"
    bak1.write_text('{"provider": "ollama", "searcher": "noop"}\n', encoding="utf-8")
    path.write_text("{corrupt", encoding="utf-8")

    cfg, anomaly = PersonaConfig.load_with_anomaly(path)

    assert cfg.provider == "ollama"
    assert cfg.searcher == "noop"
    assert anomaly is not None
    assert "bak1" in anomaly.action


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    """No persona_config.json → defaults."""
    cfg = PersonaConfig.load(tmp_path / "nope.json")
    assert cfg.provider == DEFAULT_PROVIDER
    assert cfg.searcher == DEFAULT_SEARCHER


def test_load_well_formed_file_returns_values(tmp_path: Path) -> None:
    path = tmp_path / "persona_config.json"
    path.write_text('{"provider": "ollama", "searcher": "noop"}\n', encoding="utf-8")
    cfg = PersonaConfig.load(path)
    assert cfg.provider == "ollama"
    assert cfg.searcher == "noop"


def test_load_corrupt_json_returns_defaults(tmp_path: Path) -> None:
    """Hand-corrupted JSON degrades to defaults rather than crashing."""
    path = tmp_path / "persona_config.json"
    path.write_text("{this is not json", encoding="utf-8")
    cfg = PersonaConfig.load(path)
    assert cfg.provider == DEFAULT_PROVIDER
    assert cfg.searcher == DEFAULT_SEARCHER


def test_load_non_object_payload_returns_defaults(tmp_path: Path) -> None:
    path = tmp_path / "persona_config.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    cfg = PersonaConfig.load(path)
    assert cfg.provider == DEFAULT_PROVIDER
    assert cfg.searcher == DEFAULT_SEARCHER


def test_load_wrong_field_types_falls_back_per_field(tmp_path: Path) -> None:
    path = tmp_path / "persona_config.json"
    path.write_text('{"provider": 42, "searcher": "noop"}', encoding="utf-8")
    cfg = PersonaConfig.load(path)
    assert cfg.provider == DEFAULT_PROVIDER
    assert cfg.searcher == "noop"


def test_save_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "persona_config.json"
    PersonaConfig(provider="ollama", searcher="noop").save(path)

    cfg = PersonaConfig.load(path)
    assert cfg.provider == "ollama"
    assert cfg.searcher == "noop"


def test_save_is_atomic(tmp_path: Path) -> None:
    """save() writes via .new + os.replace — no partial file on crash mid-write.

    Verified indirectly: after save() returns, no .new file remains.
    """
    path = tmp_path / "persona_config.json"
    PersonaConfig(provider="claude-cli", searcher="ddgs").save(path)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".new").exists()


# ---- Bug A (audit-3): user_name field ----


def test_persona_config_user_name_defaults_to_none(tmp_path: Path) -> None:
    """user_name is None when not set in the config file (backward-compat
    for forkers who haven't migrated yet)."""
    cfg = PersonaConfig.load(tmp_path / "persona_config.json")
    assert cfg.user_name is None


def test_persona_config_user_name_loads_from_file(tmp_path: Path) -> None:
    import json
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "user_name": "Hana"}))
    cfg = PersonaConfig.load(p)
    assert cfg.user_name == "Hana"


def test_persona_config_user_name_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "persona_config.json"
    PersonaConfig(user_name="Hana").save(p)
    loaded = PersonaConfig.load(p)
    assert loaded.user_name == "Hana"


def test_persona_config_user_name_strips_whitespace_and_treats_empty_as_none(
    tmp_path: Path,
) -> None:
    """Whitespace-only or empty user_name → None (treated as unset)."""
    import json
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"user_name": "  Hana  "}))
    assert PersonaConfig.load(p).user_name == "Hana"
    p.write_text(json.dumps({"user_name": "   "}))
    assert PersonaConfig.load(p).user_name is None
    p.write_text(json.dumps({"user_name": ""}))
    assert PersonaConfig.load(p).user_name is None
    p.write_text(json.dumps({"user_name": 42}))  # wrong type
    assert PersonaConfig.load(p).user_name is None
