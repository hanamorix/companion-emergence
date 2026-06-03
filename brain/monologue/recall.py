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
    candidates = store.list_by_type(MONOLOGUE_TRACE_TYPE, active_only=True, limit=None)
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
        store.get(mem.id)  # bump recall_count + last_accessed (keep-sharp)
        results.append(
            {
                "content": mem.content,
                "state": mem.state,  # always 'active' (active_only=True; fading traces not fetched)
                "ts": mem.created_at.isoformat(),
            }
        )
    return {"query": query, "count": len(results), "monologues": results}
