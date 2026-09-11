"""One door for "a memory was opened in full" — the #231 recall-bump consolidation.

Every path that opens a memory in full routes through ``open_memory``: the
deliberate ``read_full_memory`` tool AND every passive full-render site in the
recall block. One open means one +1.0 recall bump and, when a persona_dir is
present, one reappraisal enqueue, no matter which path did the opening. That
removes the per-tier re-derivation of "rendered full => full bump + enqueue"
that let ``SNIPPET_MODE_ENABLED`` gate a full render's bump off at some sites
(fading, semantic-snippet) but not others.
"""

from __future__ import annotations

from pathlib import Path

from brain.memory.store import Memory, MemoryStore


def open_memory(
    mem: Memory,
    *,
    store: MemoryStore,
    persona_dir: Path | None,
    deliberate: bool,
    seen: set[str] | None = None,
) -> None:
    """Register that ``mem`` was opened in full. One open event does:

    - ``recall_count`` += 1.0, ALWAYS (the full-open bump, on every path).
    - ``last_accessed_at`` touched ONLY when ``deliberate`` is True. A
      deliberate read is a genuine access event; a passive auto-full-render is
      not, so it must not reset the freshness anchor (which would feed a
      surface -> recency ranking loop).
    - a reappraisal enqueued for ``mem.id`` when ``persona_dir`` is not None.
      The legacy/test-only no-persona path cannot build a ``PendingQueue``, so
      it bumps recall_count and simply skips the enqueue.
    - dedup via ``seen``: if provided, an id already in the set is a no-op;
      otherwise the id is added, so one open per memory per pass. Pass
      ``seen=None`` for a standalone open (each deliberate tool call is its own
      event, with per-call behaviour unchanged).
    """
    if seen is not None:
        if mem.id in seen:
            return
        seen.add(mem.id)

    if deliberate:
        # +1.0 recall AND last_accessed_at — the exact single UPDATE that
        # ``store.get(bump=True)`` already performed for the deliberate read.
        store.get(mem.id, bump=True)
    else:
        # +1.0 recall ONLY — ``bump_recall`` deliberately leaves
        # last_accessed_at untouched (a passive surface is not an access event).
        store.bump_recall(mem.id, 1.0)

    if persona_dir is not None:
        from brain.memory.pending import PendingQueue

        PendingQueue(persona_dir).enqueue_reappraisal(mem.id, source="recall")
