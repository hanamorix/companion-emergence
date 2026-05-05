"""Body events — auto-journal when the body crosses a threshold.

Currently one event kind: `record_climax_event`, called from the
`add_memory` tool when an LLM-tagged memory commits with `climax >= 7`.
Writes a private journal_entry memory referencing the originating memory,
and emits a `climax_event` behavioral_log entry.

Hook point is `add_memory` (the LLM tool surface), NOT `MemoryStore.create`.
This avoids:
- Firing on migrator-imported OG memories (one-shot data import, not lived events)
- Firing on engine-generated memories (dream/research/reflex narratives that
  may reference climax in third person)
- Recursing on the journal_entry memory we ourselves create here (we use
  `store.create` directly, bypassing the add_memory tool path)

Privacy contract honored: journal_entry has `private=True, source='climax_event',
auto_generated=True`. Chat composition surfaces metadata only, never content.

Per spec §2 (climax as full vocabulary citizen) + journal-as-safe-space memo
(`feedback_journal_is_brain_safe_space.md`) + no-silent-failures discipline
(`feedback_implementation_plan_discipline.md`).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from brain.behavioral.log import append_behavioral_event
from brain.memory.store import Memory, MemoryStore

logger = logging.getLogger(__name__)

CLIMAX_THRESHOLD = 7.0


def record_climax_event(
    *,
    originating_memory: Memory,
    store: MemoryStore,
    persona_dir: Path,
) -> str | None:
    """Write a private climax journal_entry + behavioral_log entry.

    Returns the new memory id on success, None on any failure (logged at
    WARN — never raises into the caller's path).

    The journal_entry carries the originating memory's emotion snapshot
    so it shows up in aggregate exactly like the originating memory does.
    Max-pooling means the duplicate climax intensity doesn't compound the
    reset — same input, same output.
    """
    try:
        ts = originating_memory.created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        snippet = (originating_memory.content or "")[:100].replace("\n", " ").strip()
        content = f"the body crested. {ts.isoformat()}. context: {snippet}"

        journal = Memory.create_new(
            content=content,
            memory_type="journal_entry",
            domain="self",
            emotions=dict(originating_memory.emotions),
            metadata={
                "private": True,
                "source": "climax_event",
                "reflex_arc_name": None,
                "auto_generated": True,
                "originating_memory_id": originating_memory.id,
            },
        )
        store.create(journal)
    except (OSError, sqlite3.Error, ValueError) as exc:
        # Narrow: I/O + parse + storage errors absorb. Programming bugs
        # (TypeError, AttributeError, KeyError) propagate so we see them.
        logger.warning(
            "record_climax_event: journal_entry write failed (originating=%s): %s",
            originating_memory.id, exc,
        )
        return None

    # Behavioral log emit — best-effort. Memory is the load-bearing record;
    # log failure is a degraded-narrative outcome, not data loss.
    try:
        append_behavioral_event(
            persona_dir / "behavioral_log.jsonl",
            kind="climax_event",
            name=journal.id,
            timestamp=datetime.now(UTC),
            source="climax_event",
            reflex_arc_name=None,
            emotional_state=dict(originating_memory.emotions),
        )
    except (OSError, ValueError) as exc:
        logger.warning(
            "record_climax_event: behavioral_log append failed (memory=%s): %s",
            journal.id, exc,
        )

    return journal.id
