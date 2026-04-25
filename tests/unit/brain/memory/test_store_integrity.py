"""Tests for SQLite integrity check on store + hebbian open."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.health.anomaly import BrainIntegrityError
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def test_memory_store_clean_db_passes_integrity_check(tmp_path: Path) -> None:
    """A clean store opens without raising."""
    store = MemoryStore(db_path=tmp_path / "memories.db")
    store.close()
    # Re-open — integrity check runs again on fresh open
    store2 = MemoryStore(db_path=tmp_path / "memories.db")
    store2.close()


def test_memory_store_corrupt_db_raises_integrity_error(tmp_path: Path) -> None:
    """A file with bad SQLite header → BrainIntegrityError on open."""
    db = tmp_path / "memories.db"
    db.write_bytes(b"this is not a SQLite database")
    with pytest.raises(BrainIntegrityError):
        MemoryStore(db_path=db)


def test_hebbian_matrix_clean_db_passes(tmp_path: Path) -> None:
    h = HebbianMatrix(db_path=tmp_path / "hebbian.db")
    h.close()
    h2 = HebbianMatrix(db_path=tmp_path / "hebbian.db")
    h2.close()


def test_hebbian_matrix_corrupt_db_raises(tmp_path: Path) -> None:
    db = tmp_path / "hebbian.db"
    db.write_bytes(b"not sqlite")
    with pytest.raises(BrainIntegrityError):
        HebbianMatrix(db_path=db)
