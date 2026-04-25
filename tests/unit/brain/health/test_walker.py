"""Tests for brain.health.walker."""

from __future__ import annotations

from pathlib import Path

from brain.health.walker import walk_persona
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def _setup_persona(tmp_path: Path) -> Path:
    persona = tmp_path / "persona"
    persona.mkdir()
    MemoryStore(db_path=persona / "memories.db").close()
    HebbianMatrix(db_path=persona / "hebbian.db").close()
    return persona


def test_walk_clean_persona_no_anomalies(tmp_path: Path) -> None:
    persona = _setup_persona(tmp_path)
    anomalies = walk_persona(persona)
    assert anomalies == []


def test_walk_detects_corrupt_atomic_rewrite_file(tmp_path: Path) -> None:
    persona = _setup_persona(tmp_path)
    (persona / "user_preferences.json").write_text("{not json", encoding="utf-8")
    anomalies = walk_persona(persona)
    assert len(anomalies) == 1
    assert anomalies[0].file == "user_preferences.json"
    # File is healed (reset to default since no .bak)
    assert (persona / "user_preferences.json").exists()


def test_walk_detects_sqlite_corruption(tmp_path: Path) -> None:
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "memories.db").write_bytes(b"not sqlite")
    HebbianMatrix(db_path=persona / "hebbian.db").close()  # clean
    anomalies = walk_persona(persona)
    assert any(a.kind == "sqlite_integrity_fail" for a in anomalies)


def test_walk_returns_multiple_anomalies(tmp_path: Path) -> None:
    persona = _setup_persona(tmp_path)
    (persona / "user_preferences.json").write_text("{not json", encoding="utf-8")
    (persona / "persona_config.json").write_text("{also not json", encoding="utf-8")
    anomalies = walk_persona(persona)
    assert len(anomalies) >= 2
    files = {a.file for a in anomalies}
    assert "user_preferences.json" in files
    assert "persona_config.json" in files
