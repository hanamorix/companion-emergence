from datetime import UTC, datetime, timedelta

from brain.maker.sources import (
    current_emotional_intensity,
    dreams_since,
    soul_pending_count,
)
from brain.memory.store import Memory, MemoryStore


def test_soul_pending_count_zero_on_empty(tmp_path):
    assert soul_pending_count(tmp_path) == 0


def test_current_emotional_intensity_is_peak_of_recent(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    store.create(Memory.create_new("a", "conversation", "us", emotions={"joy": 6.0}))
    store.create(Memory.create_new("b", "conversation", "us", emotions={"grief": 3.0}))
    assert current_emotional_intensity(store) == 6.0  # peak across recent
    store.close()


def test_dreams_since_counts_only_after_cutoff(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    old = Memory.create_new("old dream", "dream", "us")
    object.__setattr__(old, "created_at", datetime.now(UTC) - timedelta(hours=10))
    new = Memory.create_new("new dream", "dream", "us")
    store.create(old)
    store.create(new)
    cutoff = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    assert dreams_since(store, cutoff) == 1
    store.close()
