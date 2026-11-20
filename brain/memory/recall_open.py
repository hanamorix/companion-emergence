"""One door for "a memory was opened in full": the #231 recall-bump consolidation.

Every path that opens a memory in full routes through ``open_memory``: the
``read_full_memory`` tool AND every full-render site in the recall block. One
open means one +1.0 recall bump with ``last_accessed_at`` set to now and, when
a persona_dir is present, one reappraisal enqueue and one report of the showing
to the emotion engine, no matter which path did the opening. That
removes the per-tier re-derivation of "rendered full => full bump + enqueue"
that let ``SNIPPET_MODE_ENABLED`` gate a full render's bump off at some sites
(fading, semantic-snippet) but not others.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.memory.store import Memory, MemoryStore

log = logging.getLogger(__name__)


def open_memory(
    mem: Memory,
    *,
    store: MemoryStore,
    persona_dir: Path | None,
    seen: set[str] | None = None,
    pending_ids: list[str] | None = None,
) -> None:
    """Register that ``mem`` was opened in full. One open event does:

    - ``recall_count`` += 1.0 AND ``last_accessed_at`` set to now, ALWAYS: the
      one ``UPDATE`` of ``store.get(bump=True)``. A full open is an access
      whichever path opened it (the recall block or ``read_full_memory``).
    - a reappraisal enqueued for ``mem.id`` when ``persona_dir`` is not None.
      The legacy/test-only no-persona path cannot build a ``PendingQueue``, so
      it bumps recall_count and simply skips the enqueue.
    - one report of the showing to the emotion engine, ``on_memory_shown(mem,
      now)``, when ``persona_dir`` is not None. The repeat rule (the same
      memory shown again within 24 hours counts for less) is the engine's, not
      the door's. The legacy/test-only no-persona path can build neither a
      ``PendingQueue`` nor an engine, so it bumps and skips both.
    - dedup via ``seen``: if provided, an id already in the set is a no-op;
      otherwise the id is added, so one open per memory per pass. Pass
      ``seen=None`` for a standalone open (each tool call is its own
      event, with per-call behaviour unchanged).

    ``pending_ids`` (restored batching, #231 follow-up): when given (and
    ``persona_dir`` is not None), the enqueue is NOT issued immediately —
    ``mem.id`` is appended to this collector instead, and the caller is
    responsible for flushing it with ONE ``PendingQueue.enqueue_reappraisals``
    call after the whole passive render pass finishes (matching the
    pre-consolidation batched write at 4c916b24 — one file-lock/open/append
    for every full-open id, instead of one per id). Leave ``pending_ids=None``
    for a standalone open (e.g. ``read_full_memory``): that path
    opens exactly one memory per call, so there is nothing to batch, and it
    keeps enqueuing immediately as before. The engine report is not batched:
    each open reports at once.
    """
    if seen is not None:
        if mem.id in seen:
            return
        seen.add(mem.id)

    store.get(mem.id, bump=True)

    if persona_dir is not None:
        if pending_ids is not None:
            pending_ids.append(mem.id)
        else:
            from brain.memory.pending import PendingQueue

            PendingQueue(persona_dir).enqueue_reappraisal(mem.id, source="recall")

        # Accepted edge: the bump above commits before this report is queued. A crash between the two loses this
        # showing from the engine's queue. last_accessed_at stays set, so a later rebuild still sees the row as
        # accessed. At most one showing is lost, so it is not guarded further.
        try:
            from brain.emotion import engine

            engine.engine_for(persona_dir).on_memory_shown(mem, datetime.now(UTC))
        except Exception:  # noqa: BLE001
            log.exception("open_memory: engine report failed")
