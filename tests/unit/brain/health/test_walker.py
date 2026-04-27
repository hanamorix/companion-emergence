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


def test_walk_detects_corrupt_crystallizations_db(tmp_path: Path) -> None:
    """Post-audit fix: walker now scans crystallizations.db (added in SP-5).

    Previously, soul corruption went undetected by `nell health check`.
    """
    persona = tmp_path / "persona"
    persona.mkdir()
    MemoryStore(db_path=persona / "memories.db").close()
    HebbianMatrix(db_path=persona / "hebbian.db").close()
    # Corrupt crystallizations.db with invalid SQLite header
    (persona / "crystallizations.db").write_bytes(b"not a sqlite database")

    anomalies = walk_persona(persona)
    soul_anomalies = [a for a in anomalies if a.file == "crystallizations.db"]
    assert len(soul_anomalies) == 1
    assert soul_anomalies[0].kind == "sqlite_integrity_fail"
    assert soul_anomalies[0].action == "alarmed_unrecoverable"


def test_walk_clean_persona_with_soul_db_no_anomalies(tmp_path: Path) -> None:
    """A persona with a clean crystallizations.db should NOT raise an anomaly."""
    from brain.soul.store import SoulStore

    persona = _setup_persona(tmp_path)
    SoulStore(db_path=persona / "crystallizations.db").close()  # init clean db

    anomalies = walk_persona(persona)
    assert anomalies == []


def test_walk_missing_soul_db_no_anomaly(tmp_path: Path) -> None:
    """A persona that has never crystallized has no soul db — that's expected,
    not an anomaly. Walker should skip silently."""
    persona = _setup_persona(tmp_path)
    # No crystallizations.db on disk
    anomalies = walk_persona(persona)
    assert all(a.file != "crystallizations.db" for a in anomalies)
