"""get_emotional_state tool implementation."""

from __future__ import annotations

from pathlib import Path

from brain.emotion import engine
from brain.emotion.aggregate import aggregate_state
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.utils.emotion import format_emotion_summary


def get_emotional_state(
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Return Nell's current emotional state.

    The numbers are the emotion engine's short averages when the engine is live;
    otherwise the max-pool per emotion across all active memories (same strategy
    as reflex engine). Returns dominant emotion, top-5
    ranked list, full score map, and a human-readable summary string.

    Returns
    -------
    dict with keys:
        dominant  — top emotion name or None
        top_5     — [{"emotion": str, "score": float}, ...]
        all       — {emotion: score, ...}
        summary   — human-readable multi-line string
    """
    live_engine = engine.engine_for(persona_dir)
    if live_engine.is_live():
        # Readers show absent while a recovery rebuild is pending. Accepted edge: while a recovery rebuild is pending
        # or retrying, read() returns nothing and this tool says no emotions are recorded.
        scores: dict[str, float] = {name: r.short for name, r in live_engine.read().items()}
    else:
        active_memories = store.list_active()

        emotional_state = aggregate_state(active_memories)

        scores = dict(emotional_state.emotions)

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
