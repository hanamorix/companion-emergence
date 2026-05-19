"""test_grief_touch_lookup.py — MemoryStore.exists_recent_grief_touch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.memory.store import Memory, MemoryStore


def _make_grief_event(*, referent_id: str, created_at: datetime) -> Memory:
    m = Memory.create_new(
        content=f"reached for {referent_id} — gone",
        memory_type="grief_event",
        domain="grief",
        emotions={"memory_grief": 3.5},
        metadata={"grief_referent_id": referent_id, "grief_subtype": "recall_touch"},
    )
    object.__setattr__(m, "created_at", created_at)
    return m


def test_exists_recent_grief_touch_returns_true_within_window(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    recent = _make_grief_event(referent_id="mem-x", created_at=datetime.now(UTC))
    store.create(recent)
    assert store.exists_recent_grief_touch("mem-x", hours=2.0) is True


def test_exists_recent_grief_touch_returns_false_outside_window(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    old = _make_grief_event(
        referent_id="mem-x",
        created_at=datetime.now(UTC) - timedelta(hours=5),
    )
    store.create(old)
    assert store.exists_recent_grief_touch("mem-x", hours=2.0) is False


def test_exists_recent_grief_touch_distinguishes_referents(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    other = _make_grief_event(referent_id="mem-y", created_at=datetime.now(UTC))
    store.create(other)
    assert store.exists_recent_grief_touch("mem-x", hours=2.0) is False
    assert store.exists_recent_grief_touch("mem-y", hours=2.0) is True
