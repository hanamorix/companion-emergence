"""search_memories tool implementation."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.impls._common import _mem_to_result

logger = logging.getLogger(__name__)

_TOKEN_MIN_LEN = 4  # matches _RECALL_TOKEN_MIN_LEN in brain/chat/prompt.py
_TOKEN_LIMIT = 6

_CORECALL_DELTA = 0.1       # gentle nudge; cf. add_memory/ingest at 0.5
_CORECALL_FANOUT = 4        # anchor links to at most this many other results
_CORECALL_MIN_RESULTS = 2   # below this there is nothing to associate


def _reinforce_corecall(hebbian, memories: list) -> None:
    """Strengthen star edges anchor(results[0]) -> each of the next _CORECALL_FANOUT.

    Fail-soft: a reinforcement error must never break recall. Decay-subordinate:
    only existing hebbian state is touched (no new salience term); abandoned
    edges are GC'd by the heartbeat hebbian decay pass.
    """
    try:
        if len(memories) < _CORECALL_MIN_RESULTS:
            return
        anchor = memories[0]
        for m in memories[1 : 1 + _CORECALL_FANOUT]:
            if m.id != anchor.id:
                hebbian.strengthen(anchor.id, m.id, delta=_CORECALL_DELTA)
    except Exception:  # noqa: BLE001 — reinforcement is best-effort
        logger.debug("co-recall reinforcement failed", exc_info=True)


def _query_tokens(query: str) -> list[str]:
    """Split a multi-word query into searchable tokens.

    Strips short stopword-shaped fragments (len < _TOKEN_MIN_LEN) so
    "Henryk preferences personality identity" becomes ["Henryk",
    "preferences", "personality", "identity"] rather than being searched
    as a single LIKE phrase that matches nothing.
    """
    pieces = re.split(r"[^A-Za-z0-9]+", query)
    seen: set[str] = set()
    out: list[str] = []
    for p in pieces:
        if len(p) < _TOKEN_MIN_LEN or p.lower() in seen:
            continue
        seen.add(p.lower())
        out.append(p)
        if len(out) >= _TOKEN_LIMIT:
            break
    return out or [query]


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

    Splits multi-word queries into individual tokens and searches each
    separately, then deduplicates. This prevents the previous behaviour
    where 'Henryk preferences personality identity' was passed as a single
    LIKE phrase and returned zero results even though hundreds of memories
    mentioned 'Henryk'.

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
    tokens = _query_tokens(query)
    seen_ids: set[str] = set()
    candidates = []
    for token in tokens:
        for mem in store.search_text(token, active_only=True, limit=limit * 2):
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                candidates.append(mem)

    if emotion is not None:
        emotion_lower = emotion.lower().strip()
        # Partition: emotion-matching memories first, then the rest.
        # Use id-set membership (O(n)) rather than object identity (O(n²)).
        boosted = [m for m in candidates if emotion_lower in {k.lower() for k in m.emotions}]
        boosted_ids = {m.id for m in boosted}
        rest = [m for m in candidates if m.id not in boosted_ids]
        ordered = boosted + rest
    else:
        ordered = candidates

    _reinforce_corecall(hebbian, ordered[: _CORECALL_FANOUT + 1])
    results = [_mem_to_result(m) for m in ordered[:limit]]

    return {
        "query": query,
        "emotion_filter": emotion,
        "count": len(results),
        "memories": results,
    }
