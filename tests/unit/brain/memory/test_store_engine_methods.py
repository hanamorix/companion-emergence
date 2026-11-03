"""MemoryStore methods the emotion engine's callers use: `refresh_faded_emotions` (Engine 5, #580).

Naming: synthetic user = Bob, persona = Canary, model = Claude. No real persona or user names.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from brain.memory.store import Memory, MemoryStore

_WRITTEN_AT = datetime(2031, 5, 6, 7, 8, 9, tzinfo=UTC)

_COLUMNS = "created_at, emotions_json, peak_emotion_intensity, importance, score, recall_count, content"


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def _row(store, memory_id):
    return store._conn.execute(f"SELECT {_COLUMNS} FROM memories WHERE id = ?", (memory_id,)).fetchone()


def _make(store, emotions, *, importance=3.0):
    mem = Memory.create_new(
        content="Bob likes tea", memory_type="conversation", domain="us", emotions=emotions, importance=importance
    )
    store.create(mem, fold=False)
    return mem.id


def test_refresh_is_one_update_statement_and_leaves_no_transaction_open(store):
    memory_id = _make(store, {"joy": 8.0})
    statements: list[str] = []
    store._conn.set_trace_callback(statements.append)
    store.refresh_faded_emotions(memory_id, {"joy": 2.0}, _WRITTEN_AT)
    store._conn.set_trace_callback(None)
    updates = [s for s in statements if s.lstrip().upper().startswith("UPDATE MEMORIES")]
    assert len(updates) == 1  # ONE update of the row
    assert not [s for s in statements if s.lstrip().upper().startswith(("INSERT", "DELETE"))]
    assert store._conn.in_transaction is False  # committed


def test_refresh_writes_the_emotions_and_created_at_in_the_plus_zero_form(store):
    memory_id = _make(store, {"joy": 8.0, "trust": 4.0})
    store.refresh_faded_emotions(memory_id, {"joy": 2.5, "trust": 1.0}, _WRITTEN_AT)
    created_at, emotions_json, *_ = _row(store, memory_id)
    assert created_at == "2031-05-06T07:08:09+00:00"
    assert "Z" not in created_at
    got = store.get(memory_id, bump=False)
    assert got.emotions == {"joy": 2.5, "trust": 1.0}
    assert got.created_at == _WRITTEN_AT


def test_refresh_with_empty_emotions_keeps_them_empty_and_still_writes_created_at(store):
    memory_id = _make(store, {})
    store.refresh_faded_emotions(memory_id, {}, _WRITTEN_AT)
    created_at, emotions_json, peak, *_ = _row(store, memory_id)
    assert emotions_json == "{}"
    assert created_at == "2031-05-06T07:08:09+00:00"
    assert peak == 0.0


def test_refresh_never_touches_the_peak_in_either_direction(store):
    memory_id = _make(store, {"joy": 8.0})
    store.refresh_faded_emotions(memory_id, {"joy": 2.0}, _WRITTEN_AT)  # a lower (faded) value
    assert _row(store, memory_id)[2] == pytest.approx(8.0)
    store.refresh_faded_emotions(memory_id, {"joy": 9.5}, _WRITTEN_AT)  # a higher value does not raise it either
    assert _row(store, memory_id)[2] == pytest.approx(8.0)


def test_refresh_changes_only_emotions_and_created_at(store):
    memory_id = _make(store, {"joy": 8.0}, importance=4.5)
    before = _row(store, memory_id)
    store.refresh_faded_emotions(memory_id, {"joy": 2.0}, _WRITTEN_AT)
    after = _row(store, memory_id)
    # importance, score, recall_count, content (and the peak, above) are the row's own
    assert after[3:] == before[3:]
    assert (after[0], after[1]) != (before[0], before[1])


def test_refresh_of_a_missing_id_raises_key_error_and_creates_nothing(store):
    with pytest.raises(KeyError, match="no-such-id"):
        store.refresh_faded_emotions("no-such-id", {"joy": 1.0}, _WRITTEN_AT)
    assert store.count(active_only=False) == 0
    # sqlite3 opens a write transaction for an UPDATE that changes no row: the store must not be left holding it.
    assert store._conn.in_transaction is False
    other = sqlite3.connect(store.db_path, timeout=0.2)
    try:
        other.execute("BEGIN IMMEDIATE")  # would raise "database is locked" if the store still held a write lock
        other.rollback()
    finally:
        other.close()


def test_refresh_fires_no_hook(store, emotion_engine_stub):
    memory_id = _make(store, {"joy": 8.0})
    emotion_engine_stub.recorded_calls.clear()
    store.refresh_faded_emotions(memory_id, {"joy": 2.0}, _WRITTEN_AT)
    assert emotion_engine_stub.recorded_calls == []
    # The probe can see: a plain create records the committed hook on the same stub.
    store.create(Memory.create_new(content="control", memory_type="conversation", domain="us", emotions={"joy": 1.0}))
    assert [c[0] for c in emotion_engine_stub.recorded_calls] == ["on_memory_committed"]
