from datetime import UTC, datetime, timedelta

import pytest

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


def test_live_current_emotional_intensity_is_the_peak_short_average(tmp_path, emotion_engine_stub):
    """B2 phase B: with the engine live the maker's input is the highest short average, not the store's peak."""
    from brain.emotion.engine import Reading

    store = MemoryStore(tmp_path / "memories.db")
    try:
        store.create(Memory.create_new("a", "conversation", "us", emotions={"joy": 9.0}))
        emotion_engine_stub.live = True
        emotion_engine_stub.readings = {
            "grief": Reading(short=3.5, long=0.0, difference=0.0, shift=0),
            "awe": Reading(short=0.0, long=0.0, difference=0.0, shift=0),
        }
        assert current_emotional_intensity(store) == 3.5
        emotion_engine_stub.readings = {}
        assert current_emotional_intensity(store) == 0.0
    finally:
        store.close()


@pytest.mark.no_base_failure(reason='pins that an in-memory store reads as not live, so the maker takes the old aggregate path, as the old code did')
def test_in_memory_store_takes_the_old_path_even_when_the_engine_is_live(emotion_engine_stub):
    from brain.emotion.engine import Reading

    emotion_engine_stub.live = True
    emotion_engine_stub.readings = {"awe": Reading(short=9.9, long=0.0, difference=0.0, shift=0)}
    store = MemoryStore(":memory:")
    try:
        store.create(Memory.create_new("a", "conversation", "us", emotions={"joy": 6.0}))
        assert current_emotional_intensity(store) == 6.0
    finally:
        store.close()
