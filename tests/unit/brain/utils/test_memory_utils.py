"""Tests for brain.utils.memory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.memory.store import Memory, MemoryStore
from brain.utils.memory import days_since_human


def test_days_since_human_returns_999_when_no_conversations():
    store = MemoryStore(":memory:")
    try:
        result = days_since_human(store, datetime.now(UTC))
        assert result == 999.0
    finally:
        store.close()


def test_days_since_human_computes_delta():
    store = MemoryStore(":memory:")
    try:
        mem = Memory.create_new(
            content="x",
            memory_type="conversation",
            domain="us",
            emotions={},
        )
        store.create(mem)
        # Backdate 48h
        store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(hours=48)).isoformat(), mem.id),
        )
        store._conn.commit()  # type: ignore[attr-defined]
        result = days_since_human(store, datetime.now(UTC))
        assert 1.9 < result < 2.1  # ~2 days
    finally:
        store.close()
