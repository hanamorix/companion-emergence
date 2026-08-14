"""Tier 2 read path #2 — deliberate recall.

She reaches back for a past thought; matching traces have their recall_count
bumped (store.get), which raises forgetting salience — reconstructing a thought
is what keeps it reconstructable.
"""
from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

_DEFAULT_LIMIT = 5


def recall_monologue(
    query: str,
    limit: int = _DEFAULT_LIMIT,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Return monologue_trace memories whose content matches `query` (case-
    insensitive token substring). Bumps recall_count on each returned trace."""
    tokens = [t for t in query.lower().split() if t]
    # monologue_trace is now GATED — recent traces live in the pending-candidate
    # queue (not memories.db). Read a generous recent window from the queue and
    # token-match. TEMP (Root 2 stopgap). Promoted traces (rare) are not searched
    # here — accepted; the common case is recent un-promoted traces.
    from brain.memory.pending import PendingQueue

    candidates = PendingQueue(persona_dir).read_recent(MONOLOGUE_TRACE_TYPE, limit=200)
    matched = []
    seen: set[str] = set()
    for mem in candidates:
        haystack = mem.content.lower()
        if any(tok in haystack for tok in tokens) and mem.id not in seen:
            seen.add(mem.id)
            matched.append(mem)
        if len(matched) >= limit:
            break

    results = []
    for mem in matched:
        store.get(mem.id)  # keep-sharp bump — now a NO-OP for a queue trace (id
        #                    not a memories.db row); harmless. Accepted (round-4).
        results.append(
            {
                "content": mem.content,
                "state": mem.state,  # always 'active' (active_only=True; fading traces not fetched)
                "ts": mem.created_at.isoformat(),
            }
        )
    return {"query": query, "count": len(results), "monologues": results}
