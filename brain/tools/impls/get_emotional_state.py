"""get_emotional_state tool implementation."""

from __future__ import annotations

from pathlib import Path

from brain.emotion import engine
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.utils.emotion import format_emotion_summary


def get_emotional_state(
    *,
    store: MemoryStore,  # noqa: ARG001 — kept for dispatcher signature symmetry
    hebbian: HebbianMatrix,  # noqa: ARG001 — kept for dispatcher signature symmetry
    persona_dir: Path,
) -> dict:
    """Return Nell's current emotional state.

    The numbers are the emotion engine's short averages. Returns dominant emotion, top-5
    ranked list, full score map, and a human-readable summary string.

    Returns
    -------
    dict with keys:
        dominant  — top emotion name or None
        top_5     — [{"emotion": str, "score": float}, ...]
        all       — {emotion: score, ...}
        summary   — human-readable multi-line string
    """
    # Readers show absent while a recovery rebuild is pending. Accepted edge: while a recovery rebuild is pending
    # or retrying, read() returns nothing and this tool says no emotions are recorded.
    scores: dict[str, float] = engine.short_values(engine.engine_for(persona_dir).read())

    if not scores:
        return {
            "dominant": None,
            "top_5": [],
            "all": {},
            "summary": "No active emotions recorded.",
        }

    sorted_emotions = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    dominant = sorted_emotions[0][0] if sorted_emotions else None
    top_5 = [{"emotion": name, "score": score} for name, score in sorted_emotions[:5]]

    summary = format_emotion_summary(scores)

    return {
        "dominant": dominant,
        "top_5": top_5,
        "all": scores,
        "summary": summary,
    }
