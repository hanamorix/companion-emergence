"""Reach-out emotion map (W8).

When a reach-out is DELIVERED, it moves her felt state a little: a base
tenderness (reaching toward the user is warm) plus one source-specific accent.
Pure + tunable — the constants and table are the single tuning surface.
"""
from __future__ import annotations

_BASE_TENDERNESS = 0.15
_ACCENT_MAGNITUDE = 0.10

# source -> accent channel (all registered baseline emotions). "" = tenderness only.
_SOURCE_ACCENT: dict[str, str] = {
    "dream": "vulnerability",
    "emotion_spike": "vulnerability",
    "crystallization": "pride",
    "voice_reflection": "curiosity",
    "research_completion": "curiosity",
    "recall_resonance": "nostalgia",
    "reflex_firing": "",
}


def reach_emotions_for(source: str) -> dict[str, float]:
    """Emotion vector for a delivered reach-out of the given source.

    Unknown/future source -> base tenderness only (fail-safe).
    """
    out = {"tenderness": _BASE_TENDERNESS}
    accent = _SOURCE_ACCENT.get(source, "")
    if accent:
        out[accent] = _ACCENT_MAGNITUDE
    return out
