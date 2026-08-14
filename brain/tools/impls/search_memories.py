"""search_memories tool implementation."""

from __future__ import annotations

import logging
from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.relevance import SNIPPET_MAX_CHARS, rank_memories
from brain.memory.store import MemoryStore
from brain.tools.impls._common import _mem_to_result

logger = logging.getLogger(__name__)

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


def _snippet_result(memory) -> dict:
    """Slim a Memory to a SNIPPET result: truncated body + id + snippet marker.

    Full bodies are surfaced only via ``read_full_memory(id)`` — the model pulls
    the few candidates it actually wants, so mere surfacing stops inflating
    salience. Keeps ``id`` prominent so the follow-up read is a copy-paste.
    """
    result = _mem_to_result(memory)
    body = result.get("content") or ""
    if len(body) > SNIPPET_MAX_CHARS:
        body = body[: SNIPPET_MAX_CHARS - 1].rstrip() + "…"
    result["content"] = body
    result["snippet"] = True
    return result


def search_memories(
    query: str,
    emotion: str | None = None,
    limit: int = 5,
    exclude_ids: list[str] | None = None,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Search memories by content relevance + optional emotion filter.

    Ranks the committed pool by blended relevance (BM25 text-match + importance
    + hebbian spreading-activation + recency) via ``rank_memories``. The raw
    multi-word query is passed straight through — its tokenize+OR split now
    lives in ``store._to_fts_match`` (so 'Henryk preferences personality' finds
    memories mentioning ANY token, as a union, not the empty AND-intersection).

    ``exclude_ids`` (already-surfaced + explicitly-rejected ids) are dropped
    from the ranked set before top-k, so the model can fetch the next tranche.

    If emotion is provided, memories whose emotions dict contains that emotion
    key are boosted to the front of the result list. Cap at limit.

    Retrieval is **bump-free** — surfacing does not touch ``recall_count`` (only
    a deliberate ``read_full_memory`` → ``store.get()`` bumps).
    # NEEDS-DECISION (owner, morning) ND-1: whether to also pass bump=False to
    # update()/deactivate()'s internal get() to fully decouple recall_count from
    # the write path — provisional: leave it (out of P2's retrieval scope). See
    # changes/p2-relevance/decisions.md.

    Returns
    -------
    dict with keys:
        query          — the original query string
        emotion_filter — the emotion filter (or None)
        count          — number of results returned
        memories       — list of snippet-result dicts (``snippet: true`` + id)
    """
    exclude = frozenset(exclude_ids or ())
    ranked = rank_memories(store, hebbian, query, limit=limit, exclude_ids=exclude)
    candidates = [m for m, _ in ranked]

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
    results = [_snippet_result(m) for m in ordered[:limit]]

    return {
        "query": query,
        "emotion_filter": emotion,
        "count": len(results),
        "memories": results,
    }
