"""get_emotional_state tool implementation."""

from __future__ import annotations

from pathlib import Path

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

    Aggregates across all active memories using max-pool per emotion
    (same strategy as reflex engine). Returns dominant emotion, top-5
    ranked list, full score map, and a human-readable summary string.

    Returns
    -------
    dict with keys:
        dominant  — top emotion name or None
        top_5     — [{"emotion": str, "score": float}, ...]
        all       — {emotion: score, ...}
        summary   — human-readable multi-line string
    """
    active_memories = store.search_text("", active_only=True)
    # search_text("") matches all rows; use list_all fallback if store has
    # a list_all method, otherwise rely on the empty-query LIKE match.
    # The LIKE '%''%' with empty escaped query returns all rows.
    if not active_memories:
        # Also try listing by count — if store is truly empty that's fine.
        pass

    emotional_state = aggregate_state(active_memories)

    scores: dict[str, float] = dict(emotional_state.emotions)

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
