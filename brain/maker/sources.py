"""brain.maker.sources — live charge signals, read from existing stores.

Emotional intensity (peak of the engine's short averages when the engine is live,
else of the current aggregate), soul crystallisation
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
    """Peak intensity across the current emotional state (0.0 if none).

    When the emotion engine is live this is the highest short average it reads;
    otherwise the peak of the aggregate below.

    ``MemoryStore.list_by_type`` requires a concrete type string (it filters
    ``memory_type = ?``), so there is no ``None``-means-all path. Mirror
    ``brain/bridge/persona_state.py::_build_emotions``: pull the recent active
    emotion-carrying rows directly, aggregate, and take the peak.
    """
    return max(_current_feelings(store).values(), default=0.0)


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


def _current_feelings(store: MemoryStore) -> dict[str, float]:
    """Her current feelings by name: the engine's short averages when it is live, else the recent aggregate."""
    live_engine = store.emotion_engine()
    if live_engine is not None and live_engine.is_live():
        # Readers show absent while a recovery rebuild is pending. Accepted edge: while a recovery rebuild is pending
        # or retrying, read() returns nothing, so there are no feelings and the intensity is 0.0.
        return {name: r.short for name, r in live_engine.read().items()}
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
        return {}
    return dict(aggregate_state(mems).emotions)


def making_source(charge, store: MemoryStore) -> str | None:
    """The source accent key for a making: whichever of soul, dream and emotion built most of the charge (#721).

    Ties go soul, then dream, then emotion. A charge with no parts (all zero, as one an older brain left) has no
    source, so the making carries the base nudge only.
    """
    parts = (("soul", charge.soul_part), ("dream", charge.dream_part), ("emotion", charge.emotion_part))
    name, size = max(parts, key=lambda part: part[1])  # max keeps the first of equal parts: the tie order above
    if size <= 0.0:
        return None
    if name != "emotion":
        return name
    # Emotion built it: her strongest feeling now says which. Only grief and joy have an accent.
    feelings = _current_feelings(store)
    strongest = max(feelings, key=feelings.__getitem__) if feelings else None
    return strongest if strongest in ("grief", "joy") else None
