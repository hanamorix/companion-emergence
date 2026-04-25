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
