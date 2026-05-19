"""test_emotion.py — memory_grief vocabulary registration + decay shape."""

from __future__ import annotations

from brain.emotion.decay import apply_decay
from brain.emotion.state import EmotionalState
from brain.emotion.vocabulary import get as get_emotion
from brain.grief import policy


def test_memory_grief_registered_with_30_day_half_life() -> None:
    emotion = get_emotion("memory_grief")
    assert emotion is not None, "memory_grief must be registered in the baseline vocabulary"
    assert emotion.decay_half_life_days == policy.DECAY_HALF_LIFE_DAYS == 30.0
    assert emotion.category == "complex"


def test_memory_grief_decays_at_30_day_half_life() -> None:
    state = EmotionalState()
    state.set("memory_grief", 8.0)
    # 30 days = 30 * 86400 seconds; expect intensity to halve.
    apply_decay(state, elapsed_seconds=30 * 86400.0)
    assert state.emotions["memory_grief"] == 4.0
