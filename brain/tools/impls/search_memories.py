"""search_memories tool implementation."""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.impls._common import _mem_to_result


def search_memories(
    query: str,
    emotion: str | None = None,
    limit: int = 5,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Search memories by content keyword + optional emotion filter.

    Uses store.search_text for the initial candidate pool (active only).
    If emotion is provided, memories whose emotions dict contains that emotion
    key are boosted to the front of the result list. Cap at limit.

    Returns
    -------
    dict with keys:
        query          — the original query string
        emotion_filter — the emotion filter (or None)
        count          — number of results returned
        memories       — list of _mem_to_result dicts
    """
    # Pull a larger candidate pool so emotion boosting has material to work with.
    candidates = store.search_text(query, active_only=True, limit=limit * 2)

    if emotion is not None:
        emotion_lower = emotion.lower().strip()
        # Partition: emotion-matching memories first, then the rest.
        boosted = [m for m in candidates if emotion_lower in {k.lower() for k in m.emotions}]
        rest = [m for m in candidates if m not in boosted]
        ordered = boosted + rest
    else:
        ordered = candidates

    results = [_mem_to_result(m) for m in ordered[:limit]]

    return {
        "query": query,
        "emotion_filter": emotion,
        "count": len(results),
        "memories": results,
    }
