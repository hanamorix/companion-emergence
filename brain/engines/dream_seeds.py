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


_STOPWORDS = frozenset({
    "the", "and", "but", "for", "with", "that", "this", "was", "were",
    "you", "your", "she", "her", "him", "his", "they", "them", "are",
    "not", "had", "has", "have", "from", "out", "about", "into", "what",
})


def _tokens(text: str) -> frozenset[str]:
    """Normalised content words: lowercased, >=3 chars, minus stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(w for w in words if len(w) >= 3 and w not in _STOPWORDS)


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of the two texts' content-word sets (0.0..1.0)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def identity_congruence(
    memory: Memory,
    crystallizations: list[Crystallization],
) -> float:
    """Resonance-weighted max lexical overlap between the memory's content and
    any active crystallization's moment. Lexical (Jaccard token overlap), NOT
    embedding-based — the project's only EmbeddingProvider is a non-semantic
    fake, so a cosine would be inert. No time decay — identity is permanent;
    influence scales with resonance, not age. 0.0 when there are no
    crystallizations.
    """
    if not crystallizations:
        return 0.0
    best = 0.0
    for c in crystallizations:
        weighted = _token_overlap(memory.content, c.moment) * (c.resonance / 10.0)
        if weighted > best:
            best = weighted
    return best
