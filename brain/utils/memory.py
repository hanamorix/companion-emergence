"""Shared memory helpers used by multiple engines."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from brain.memory.store import Memory, MemoryStore, _row_to_memory

logger = logging.getLogger(__name__)


def days_since_human(
    now: datetime,
    *,
    persona_dir: Path,
) -> float:
    """Days since the user last spoke. 999.0 if no conversation file holds a user turn.

    The time of the user's last turn is ``brain.ingest.buffer.last_user_turn_ts`` (the active conversation files, then the
    archive when no active file holds a user turn). The function reads no memory and no presence file: a recalled
    memory's ``created_at`` can move, a conversation file's user row cannot.

    ``last_user_turn_ts`` is imported here, not at the top: ``brain.ingest`` pulls the ingest pipeline and the provider
    stack, and this module is imported by the emotion engine (through ``engine_utils``) in every process.

    Used by reflex + research engines to gate on persona-silence duration, by get_body_state/boot brain-tools, and by the
    /persona/state aggregator.
    """
    from brain.ingest.buffer import last_user_turn_ts

    last_turn = last_user_turn_ts(persona_dir)
    if last_turn is None:
        return 999.0
    return (now - last_turn).total_seconds() / 86400.0


def is_conversation_derived(memory: Memory) -> bool:
    """True when a memory came from a conversation buffer.

    Production ingest stores extracted chat memories under the extracted label
    (``fact``, ``observation``, ``feeling``, etc.) rather than the legacy
    ``memory_type="conversation"``. The stable contract is therefore the
    conversation tag and/or ``metadata.source_summary == "conversation:<sid>"``.
    Keep the legacy type check so old migrated/test data still works.
    """
    if memory.memory_type == "conversation":
        return True
    if any(str(tag) == "conversation" for tag in memory.tags):
        return True
    source_summary = memory.metadata.get("source_summary")
    return isinstance(source_summary, str) and source_summary.startswith("conversation:")


def list_conversation_memories(
    store: MemoryStore,
    *,
    active_only: bool = True,
    limit: int | None = None,
) -> list[Memory]:
    """List memories derived from chat sessions, newest first.

    This centralizes the conversation-derived contract for engines that need
    recent chat residue. It deliberately does not query only
    ``memory_type='conversation'`` because production ingest never writes that
    type for extracted memories.
    """
    sql = "SELECT * FROM memories"
    if active_only:
        sql += " WHERE active = 1"
    sql += " ORDER BY created_at DESC"
    try:
        rows = store._conn.execute(sql).fetchall()  # noqa: SLF001 - same-tier utility
    except Exception:  # noqa: BLE001
        logger.warning("conversation memory scan failed", exc_info=True)
        return []

    out: list[Memory] = []
    for row in rows:
        try:
            memory = _row_to_memory(row)
        except Exception:  # noqa: BLE001
            continue
        if not is_conversation_derived(memory):
            continue
        out.append(memory)
        if limit is not None and len(out) >= limit:
            break
    return out
