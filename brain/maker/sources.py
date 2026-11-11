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
    live_engine = store.emotion_engine()
    if live_engine is None:
        return 0.0
    # Readers show absent while a recovery rebuild is pending. Accepted edge: while a recovery rebuild is pending
    # or retrying, read() returns nothing and the intensity is 0.0.
    return max((r.short for r in live_engine.read().values()), default=0.0)


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
