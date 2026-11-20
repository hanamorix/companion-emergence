"""brain.maker.sources — live charge signals, read from existing stores.

Emotional intensity (peak of the emotion engine's short averages), soul crystallisation
activity (current pending count — caller takes the delta), and dream activity
(dream-typed memories since a cutoff). No new producer hooks.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def current_emotional_intensity(store: MemoryStore) -> float:
    """Peak intensity across the current emotional state (0.0 if none): the highest short average the
    emotion engine reads. An in-memory store has no persona directory, so it has no engine to ask and shows 0.0.
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
    """Her current feelings by name: the emotion engine's short averages. An in-memory store has no persona
    directory, so it has no engine to ask and shows none."""
    live_engine = store.emotion_engine()
    if live_engine is None:
        return {}
    # Readers show absent while a recovery rebuild is pending. Accepted edge: while a recovery rebuild is pending
    # or retrying, read() returns nothing, so there are no feelings and the intensity is 0.0.
    return {name: r.short for name, r in live_engine.read().items()}


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
