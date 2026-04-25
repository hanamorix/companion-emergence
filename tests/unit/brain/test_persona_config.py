"""Tests for brain.persona_config — provider + searcher routing file."""

from __future__ import annotations

from pathlib import Path

from brain.persona_config import DEFAULT_PROVIDER, DEFAULT_SEARCHER, PersonaConfig


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
