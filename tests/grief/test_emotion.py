"""test_emotion.py — memory_grief vocabulary registration."""

from __future__ import annotations

from brain.emotion.vocabulary import get as get_emotion
from brain.grief import policy


def test_memory_grief_registered_with_30_day_half_life() -> None:
    emotion = get_emotion("memory_grief")
    assert emotion is not None, "memory_grief must be registered in the baseline vocabulary"
    assert emotion.decay_half_life_days == policy.DECAY_HALF_LIFE_DAYS == 30.0
    assert emotion.category == "complex"
