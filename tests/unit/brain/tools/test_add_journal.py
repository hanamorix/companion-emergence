"""brain.tools.impls.add_journal — writes journal_entry memories with privacy metadata."""
from __future__ import annotations

import pytest

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.impls.add_journal import add_journal


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


@pytest.fixture
def hebbian(tmp_path):
    h = HebbianMatrix(tmp_path / "hebbian.db")
    yield h
    h.close()


def test_add_journal_writes_journal_entry_memory_type(tmp_path, store, hebbian):
    """memory_type must be 'journal_entry' (was 'journal' pre-spec)."""
    result = add_journal(
        "today felt heavy",
        store=store, hebbian=hebbian, persona_dir=tmp_path,
    )
    mem = store.get(result["created_id"])
    assert mem.memory_type == "journal_entry"
    assert result["memory_type"] == "journal_entry"


def test_add_journal_metadata_marks_private(tmp_path, store, hebbian):
    result = add_journal(
        "private thought",
        store=store, hebbian=hebbian, persona_dir=tmp_path,
    )
    mem = store.get(result["created_id"])
    assert mem.metadata.get("private") is True
    assert mem.metadata.get("source") == "brain_authored"
    assert mem.metadata.get("auto_generated") is False
    assert mem.metadata.get("reflex_arc_name") is None


def test_add_journal_emits_behavioral_log_entry(tmp_path, store, hebbian):
    from brain.behavioral.log import read_behavioral_log

    add_journal(
        "an entry",
        store=store, hebbian=hebbian, persona_dir=tmp_path,
    )
    entries = read_behavioral_log(tmp_path / "behavioral_log.jsonl")
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "journal_entry_added"
    assert e["source"] == "brain_authored"
    assert e["reflex_arc_name"] is None


def test_add_journal_returns_dict_shape(tmp_path, store, hebbian):
    """Return shape: {created_id, memory_type}."""
    result = add_journal(
        "x",
        store=store, hebbian=hebbian, persona_dir=tmp_path,
    )
    assert set(result.keys()) == {"created_id", "memory_type"}
    assert isinstance(result["created_id"], str)
    assert len(result["created_id"]) >= 32  # UUID-like


def test_add_journal_emotions_field_is_dict(tmp_path, store, hebbian):
    """V1 doesn't extract emotions at write-time (YAGNI per spec). Empty dict OK."""
    result = add_journal(
        "i am feeling grateful and tender today",
        store=store, hebbian=hebbian, persona_dir=tmp_path,
    )
    mem = store.get(result["created_id"])
    assert isinstance(mem.emotions, dict)
