"""add_journal tool implementation.

Per spec §3.3: writes a private journal_entry memory (memory_type="journal_entry").
Always sets metadata.private=True and source="brain_authored". Emits a
journal_entry_added behavioral_log entry on success.

The journal is the brain's safe space — see feedback_journal_is_brain_safe_space.md.
The chat system message reinforces the privacy contract every turn (per
feedback_contracts_adjacent_to_data.md). This tool's role is only to write;
the contract enforcement happens at chat-composition time.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.behavioral.log import append_behavioral_event
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore

logger = logging.getLogger(__name__)


def add_journal(
    content: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Write a private journal_entry memory and log the event.

    Returns a dict with keys:
        created_id   — the new memory's UUID string
        memory_type  — always "journal_entry"
    """
    memory = Memory.create_new(
        content=content,
        memory_type="journal_entry",
        domain="self",
        emotions={},
        metadata={
            "private": True,
            "source": "brain_authored",
            "reflex_arc_name": None,
            "auto_generated": False,
        },
    )
    store.create(memory)

    # Emit behavioral_log entry. Best-effort: if logging fails, the memory is
    # still written — log failure is recoverable, memory loss is not.
    try:
        append_behavioral_event(
            persona_dir / "behavioral_log.jsonl",
            kind="journal_entry_added",
            name=memory.id,
            timestamp=datetime.now(UTC),
            source="brain_authored",
            reflex_arc_name=None,
            emotional_state=dict(memory.emotions),
        )
    except (OSError, ValueError) as exc:
        logger.warning("add_journal: behavioral_log append failed: %s", exc)

    return {
        "created_id": memory.id,
        "memory_type": "journal_entry",
    }
