"""Pending-candidate queue — a separate store OUTSIDE memories.db.

TEMP (Root 2 stopgap — remove when the Phase 5 dream cycle lands to replace it).

Automatically-generated ("firehose") memories are ENQUEUED here as *candidates*
rather than written straight into ``memories.db``. A candidate is **not a stored
memory**: it never appears in recall, forgetting, hebbian, or the graveyard. On
the idle heartbeat tick the consolidation gate (``brain.engines.consolidation``)
drains this queue and, per candidate, discards it (reject), ``store.create``s it
(promote), or folds it into an existing memory (merge). Because a rejected
candidate was never a ``memories.db`` row, none of the main-DB machinery
(grief/forgetting/hebbian) ever applies to it.

Backing store: ``<persona_dir>/pending_candidates.jsonl``, guarded by the same
``file_lock`` pattern used for ``soul_candidates.jsonl`` (see brain/soul/review.py)
so a concurrent enqueue is never lost across a drain's read-modify-truncate.

Write routing (``route_write``) keys on ``memory_type``, NOT the call site: a
type is gated (enqueued) unless it is in ``GATE_BYPASS_TYPES``. This is robust to
the two variable-typed writers (a reflex arc's ``arc.output_memory_type`` — one
value is the *deliberate* ``journal_entry`` — and conversation-ingest's
``item.label``): each row self-classifies by its own type. ``journal_entry``
(deliberate; feeds the weekly self-narrative) and ``initiate_outbound`` (its own
dedup needs immediate DB visibility) bypass the gate and commit directly.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.memory.store import Memory, MemoryStore
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

_QUEUE_FILENAME = "pending_candidates.jsonl"

# Types that BYPASS the gate and are written straight to memories.db. Everything
# else produced at an automatic write site is enqueued. Denylist (not allowlist)
# because reflex arc output types are open-ended; fail-safe direction is to gate
# an unknown automatic type (delayed/vetted, recoverable) rather than leak it into
# recall. Owner-confirmed BROAD scope (Roy, 2026-08-11): only journal_entry and
# initiate_outbound bypass.
GATE_BYPASS_TYPES: frozenset[str] = frozenset({"journal_entry", "initiate_outbound"})

# Only these monologue-EPISODE types are eligible for the Pass-1 salience drop —
# they carry a real 0..10 importance signal (extractor sets importance=salience*10).
# Dreams (importance auto-derives to ~0 when emotion-flat), research/heartbeat/
# reflex/initiate (same emotion-derived default), and monologue_trace (pinned 0.3)
# are EXEMPT: they go through dedup only, never salience-drop. Owner directive
# (Roy, 2026-08-11) — a single flat floor would nuke legitimate flat content.
SALIENCE_ELIGIBLE_TYPES: frozenset[str] = frozenset(
    {"monologue", "monologue_emotion", "monologue_soul_candidate"}
)


class PendingQueue:
    """Append-only JSONL queue of pending candidates for one persona."""

    def __init__(self, persona_dir: str | Path) -> None:
        self.path = Path(persona_dir) / _QUEUE_FILENAME

    def enqueue(self, mem: Memory, *, source: str) -> None:
        """Append a candidate (the Memory serialised + provenance) under lock."""
        entry = {
            **mem.to_dict(),
            "_source": source,
            "_enqueued_at": datetime.now(UTC).isoformat(),
        }
        line = json.dumps(entry, ensure_ascii=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path):
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read_recent(self, memory_type: str, *, limit: int) -> list[Memory]:
        """Return up to `limit` most-recent candidates of `memory_type`.

        Lock-free read (tolerates a partial last line from a concurrent
        append via ``read_jsonl_skipping_corrupt``). Newest first — entries
        are appended oldest→newest, so reversed order is most-recent-first.
        Used by interior-continuity (a DIFFERENT path from recall); returns
        ``Memory`` objects for interface parity with ``store.list_by_type``.
        """
        if limit <= 0 or not self.path.exists():
            return []
        rows = read_jsonl_skipping_corrupt(self.path)
        out: list[Memory] = []
        for entry in reversed(rows):
            if entry.get("memory_type") != memory_type:
                continue
            try:
                out.append(Memory.from_dict(entry))
            except (KeyError, ValueError, TypeError):
                continue
            if len(out) >= limit:
                break
        return out

    def drain(self) -> list[dict]:
        """Atomically take the whole queue: read all entries, then truncate.

        Held under ``file_lock`` over the read+truncate window (the proven
        soul-candidate pattern) so a candidate enqueued concurrently lands
        either in this batch or in the emptied file (next tick) — never lost.
        The lock is released before the (slow) gate processing runs on the
        returned batch. Returns raw dicts (carrying ``_source``/``_enqueued_at``);
        the gate reconstructs ``Memory.from_dict`` for promotion.
        """
        if not self.path.exists():
            return []
        with file_lock(self.path):
            rows = read_jsonl_skipping_corrupt(self.path)
            # Truncate the data file in place (the lock sidecar is separate).
            open(self.path, "w", encoding="utf-8").close()
        return rows


def route_write(store: MemoryStore, mem: Memory, *, source: str) -> str:
    """Route an automatic write by memory_type: enqueue if gated, else create.

    A gated type (anything not in ``GATE_BYPASS_TYPES``) is enqueued as a
    candidate; a bypass type is written straight to ``memories.db``. Locates the
    persona's queue via ``store.persona_dir`` so callers need only the store +
    the Memory they already hold. Returns ``mem.id`` (matching ``store.create``).
    """
    if mem.memory_type in GATE_BYPASS_TYPES:
        return store.create(mem)
    PendingQueue(store.persona_dir).enqueue(mem, source=source)
    return mem.id
