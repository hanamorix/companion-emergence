"""Tests for brain.emotion.aggregate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.emotion.aggregate import aggregate_state
from brain.emotion.state import EmotionalState
from brain.memory.store import Memory


def _mem(emotions: dict[str, float], age_hours: float = 0.0) -> Memory:
    return Memory(
        id=f"m-{emotions}-{age_hours}",
        content="x",
        memory_type="conversation",
        domain="us",
        created_at=datetime.now(UTC) - timedelta(hours=age_hours),
        emotions=dict(emotions),
    )


def test_aggregate_empty_returns_empty_state():
    result = aggregate_state([])
    assert isinstance(result, EmotionalState)
    assert result.emotions == {}


def test_aggregate_max_pools_per_emotion():
    memories = [
        _mem({"love": 6.0, "curiosity": 4.0}),
        _mem({"love": 8.0, "defiance": 3.0}),
    ]
    result = aggregate_state(memories)
    assert result.emotions["love"] == 8.0
    assert result.emotions["curiosity"] == 4.0
    assert result.emotions["defiance"] == 3.0


def test_aggregate_ignores_unknown_emotions_silently():
    memories = [_mem({"not_a_real_emotion": 9.0, "love": 5.0})]
    result = aggregate_state(memories)
    assert "love" in result.emotions
    assert "not_a_real_emotion" not in result.emotions


def test_dropped_unregistered_emotion_warns_once(caplog):
    import logging

    from brain.emotion.aggregate import _warned_unregistered

    _warned_unregistered.clear()  # isolate from other test runs
    mem = Memory.create_new(
        content="x",
        memory_type="note",
        domain="us",
        emotions={"warmth": 8.0},  # warmth = unregistered persona-extension
    )
    with caplog.at_level(logging.WARNING):
        aggregate_state([mem])
        aggregate_state([mem])  # second call must NOT re-warn
    warnings = [r for r in caplog.records if "warmth" in r.getMessage()]
    assert len(warnings) == 1, (
        f"expected exactly one warning for the dropped stored emotion, got {len(warnings)}"
    )
