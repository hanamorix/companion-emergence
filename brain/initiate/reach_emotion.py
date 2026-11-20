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


def reach_emotions_for(source: str, *, engine_scale: bool = False) -> dict[str, float]:
    """Emotion vector for a delivered reach-out of the given source.

    Unknown/future source -> base tenderness only (fail-safe). ``engine_scale``
    gives the same nudge on the engine's 0-10 scale, ten times the old small
    values (B1 PR 3). The caller chooses it with
    `brain.emotion.write_scale.writes_engine_scale` (B1 PR 7).
    """
    scale = 10.0 if engine_scale else 1.0
    out = {"tenderness": _BASE_TENDERNESS * scale}
    accent = _SOURCE_ACCENT.get(source, "")
    if accent:
        out[accent] = _ACCENT_MAGNITUDE * scale
    return out
