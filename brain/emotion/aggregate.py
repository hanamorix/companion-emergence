"""Aggregate a current EmotionalState from a list of memories.

Reflex uses this to evaluate arc triggers: what is the persona's
current emotional state, synthesized across recent memories.

Strategy: max-pool per emotion. The strongest signal across the
input memories wins — matches how OG reflex_engine read peaks,
not averages, for threshold evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable

from brain.emotion.state import EmotionalState
from brain.emotion.vocabulary import get as _get_emotion
from brain.memory.store import Memory


def aggregate_state(memories: Iterable[Memory]) -> EmotionalState:
    """Return an EmotionalState that is the per-emotion max across inputs.

    Unknown emotions (not in the registered vocabulary) are silently
    skipped — a persona's old memories may contain retired emotion
    names that no longer validate via EmotionalState.set.
    """
    pooled: dict[str, float] = {}
    for mem in memories:
        for name, intensity in mem.emotions.items():
            try:
                value = float(intensity)
            except (TypeError, ValueError):
                continue
            if value <= 0.0:
                continue
            if _get_emotion(name) is None:
                continue
            if value > pooled.get(name, 0.0):
                pooled[name] = value

    state = EmotionalState()
    for name, value in pooled.items():
        try:
            state.set(name, value)
        except (KeyError, ValueError):
            # clamp violation or validation failure — skip
            continue
    return state
