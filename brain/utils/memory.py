"""Shared memory helpers used by multiple engines."""

from __future__ import annotations

from datetime import UTC, datetime

from brain.memory.store import MemoryStore


def days_since_human(store: MemoryStore, now: datetime) -> float:
    """Days since the most recent memory_type='conversation'. 999.0 if none.

    Used by reflex + research engines to gate on persona-silence duration.
    """
    convos = store.list_by_type("conversation", active_only=True, limit=1)
    if not convos:
        return 999.0
    latest = convos[0].created_at
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=UTC)
    return (now - latest).total_seconds() / 86400.0
