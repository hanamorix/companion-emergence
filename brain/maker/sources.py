"""brain.maker.sources — live charge signals, read from existing stores.

Emotional intensity (peak of the current aggregate), soul crystallisation
activity (current pending count — caller takes the delta), and dream activity
(dream-typed memories since a cutoff). No new producer hooks.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from brain.emotion.aggregate import aggregate_state
from brain.memory.store import MemoryStore, _row_to_memory

logger = logging.getLogger(__name__)

_RECENT_EMOTION_WINDOW = 200


def current_emotional_intensity(store: MemoryStore) -> float:
    """Peak intensity across the current emotional aggregate (0.0 if none).

    ``MemoryStore.list_by_type`` requires a concrete type string (it filters
    ``memory_type = ?``), so there is no ``None``-means-all path. Mirror
    ``brain/bridge/persona_state.py::_build_emotions``: pull the recent active
    emotion-carrying rows directly, aggregate, and take the peak.
    """
    try:
        rows = store._conn.execute(  # noqa: SLF001
            "SELECT * FROM memories "
            "WHERE active = 1 "
            "AND emotions_json IS NOT NULL "
            "AND emotions_json != '{}' "
            "ORDER BY created_at DESC LIMIT ?",
            (_RECENT_EMOTION_WINDOW,),
        ).fetchall()
        mems = [_row_to_memory(row) for row in rows]
    except Exception:
        logger.exception("maker.sources: listing recent memories failed")
        return 0.0
    state = aggregate_state(mems)
    if not state.emotions:
        return 0.0
    return max(state.emotions.values())


def dreams_since(store: MemoryStore, cutoff_iso: str) -> int:
    """Count dream-typed memories created at or after the cutoff."""
    try:
        cutoff = datetime.fromisoformat(cutoff_iso)
    except (ValueError, TypeError):
        return 0
    try:
        dreams = store.list_by_type("dream", active_only=True, limit=500)
    except Exception:
        logger.exception("maker.sources: listing dreams failed")
        return 0
    return sum(1 for m in dreams if m.created_at >= cutoff)


def soul_pending_count(persona_dir: Path) -> int:
    """Current auto_pending soul-candidate count (caller takes the delta)."""
    try:
        from brain.soul.review import count_eligible_pending
        return count_eligible_pending(persona_dir)
    except Exception:
        logger.exception("maker.sources: soul pending count failed")
        return 0
