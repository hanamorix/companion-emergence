"""Pure seed-selection logic for the dream cycle.

No I/O and no engine dependencies — every function takes plain Memory /
EmotionalState / Crystallization objects so it is unit-testable in isolation.
See docs/superpowers/specs/2026-05-26-multi-signal-dream-seeds-design.md.
"""

from __future__ import annotations

import re

from brain.emotion.state import EmotionalState
from brain.memory.store import Memory
from brain.soul.crystallization import Crystallization

# Calibration defaults (overridable by callers / DreamEngine fields).
MOOD_FLOOR = 0.5
MIN_CONGRUENT = 3
REFRACTORY_WINDOW = 5
W_IDENTITY = 1.0
W_GRIEF = 1.0
W_REFRACTORY = 2.0


def emotional_congruence(memory: Memory, mood: EmotionalState) -> float:
    """Sum over emotions shared by mood and memory of (mood_intensity * memory_value)."""
    total = 0.0
    for name, intensity in mood.emotions.items():
        mv = memory.emotions.get(name, 0.0)
        if mv > 0.0:
            total += intensity * mv
    return total


def mood_is_active(mood: EmotionalState, *, floor: float = MOOD_FLOOR) -> bool:
    """True when a dominant emotion is active above the recoloring floor."""
    if mood.dominant is None:
        return False
    return mood.emotions.get(mood.dominant, 0.0) >= floor
