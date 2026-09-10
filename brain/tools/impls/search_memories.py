"""search_memories tool implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from brain.memory.embeddings import build_embedding_cache, cosine_similarity
from brain.memory.hebbian import HebbianMatrix
from brain.memory.relevance import rank_memories, snippet_length
from brain.memory.semantic_recall import build_semantic_candidate_pool
from brain.memory.store import Memory, MemoryStore
from brain.tools.impls._common import _mem_to_result

logger = logging.getLogger(__name__)

_CORECALL_DELTA = 0.1       # gentle nudge; cf. add_memory/ingest at 0.5
_CORECALL_FANOUT = 4        # anchor links to at most this many other results
_CORECALL_MIN_RESULTS = 2   # below this there is nothing to associate

SearchMode = Literal["semantic", "lexical"]


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
    max_chars = snippet_length(len(body))
    if len(body) > max_chars:
        body = body[: max_chars - 1].rstrip() + "…"
    result["content"] = body
    result["snippet"] = True
    return result


def _lexical_candidates(
    store: MemoryStore,
    hebbian: HebbianMatrix,
    query: str,
    *,
    limit: int,
    exclude: frozenset[str],
) -> list[Memory]:
    """Today's keyword ranker: BM25 text-match + importance + hebbian
    spreading-activation + recency, via ``rank_memories``. Unchanged
    behavior — this is exactly what ``search_memories`` did before the
    ``mode`` toggle existed, extracted so both modes share the same
    emotion-boost/formatting tail below."""
    ranked = rank_memories(store, hebbian, query, limit=limit, exclude_ids=exclude)
    return [m for m, _ in ranked]


def _semantic_top_k(
    store: MemoryStore,
    persona_dir: Path,
    query: str,
    *,
    limit: int,
    exclude: frozenset[str],
) -> list[Memory] | None:
    """Top-K cosine-ranked memories for an ACTIVE search call.

    Embeds ``query`` once via the shared process-cached embedding provider
    (``build_embedding_cache`` → ``build_embedding_provider``, cached by
    model_id — no per-call model reload), cosines it against every
    actively-cached memory vector (Stage 3's ``build_semantic_candidate_pool``:
    active memories that already have a cached vector under the current
    model_id — never triggers a new embed for an uncached memory, the same
    warm-up contract passive recall uses), and returns the top ``limit``
    memories in cosine-descending order.

    Deliberately does NOT reuse ``semantic_recall``'s option-4 shape/tier
    classifier — that machinery decides whether to surface an unsolicited
    passive-recall block at all. Here the model explicitly asked for a
    search, so a plain top-k cosine ranking is the natural "semantic search"
    behavior, mirroring how the lexical path is a plain top-k relevance
    ranking too.

    Returns ``None`` (never raises) when semantic search cannot run right
    now — no cached vectors yet, an embedding-model failure, or any other
    error anywhere in this path — so the caller falls back to the lexical
    path. Mirrors ``run_semantic_recall``'s fail-soft posture: the whole body
    is wrapped so a broken/missing local model or a transient store error
    only demotes this call to lexical, never breaks the tool.
    """
    try:
        embeddings_cache = build_embedding_cache(persona_dir)
    except Exception:  # noqa: BLE001 — fail-soft: never break the tool call
        logger.exception(
            "search_memories(semantic): failed to open embedding cache — falling back to lexical"
        )
        return None
    try:
        try:
            pool = build_semantic_candidate_pool(store, embeddings_cache)
            if not pool:
                return None
            try:
                query_vec = embeddings_cache.embed_query(query)
            except Exception:  # noqa: BLE001 — fail-soft
                logger.exception(
                    "search_memories(semantic): query embed failed — falling back to lexical"
                )
                return None

            scored = [
                (mid, cosine_similarity(query_vec, vec))
                for mid, (_, vec) in pool.items()
                if mid not in exclude
            ]
            if not scored:
                return None
            scored.sort(key=lambda pair: -pair[1])
            return [pool[mid][0] for mid, _ in scored[:limit]]
        except Exception:  # noqa: BLE001 — fail-soft: ANY failure demotes to lexical, never raises
            logger.warning(
                "search_memories(semantic): semantic path failed after opening the embedding "
                "cache — falling back to lexical",
                exc_info=True,
            )
            return None
    finally:
        embeddings_cache.close()


def search_memories(
    query: str,
    emotion: str | None = None,
    limit: int = 5,
    exclude_ids: list[str] | None = None,
    mode: SearchMode = "semantic",
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Search memories by content relevance + optional emotion filter.

    ``mode`` picks the retrieval path (default ``"semantic"``):
      - ``"semantic"``: embeds ``query`` once and ranks the persona's cached
        memory vectors by cosine similarity, top-k (see ``_semantic_top_k``).
        Meaning-based — catches a paraphrase with no shared keyword. Fails
        soft to ``"lexical"`` the instant semantic retrieval can't run right
        now (no cached vectors yet / embedding model unavailable / any embed
        error) — the returned ``mode`` reflects the path actually used.
      - ``"lexical"``: today's blended keyword ranker — BM25 text-match +
        importance + hebbian spreading-activation + recency, via
        ``rank_memories`` (see ``_lexical_candidates``). Unchanged from
        before this mode toggle existed. The raw multi-word query is passed
        straight through — its tokenize+OR split lives in
        ``store._to_fts_match`` (so 'Henryk preferences personality' finds
        memories mentioning ANY token, as a union, not the empty
        AND-intersection).

    Both modes return at most ``limit`` candidates, already ranked;
    ``exclude_ids`` (already-surfaced + explicitly-rejected ids) are dropped
    before that top-k, so the model can fetch the next tranche.

    If emotion is provided, memories whose emotions dict contains that emotion
    key are boosted to the front of the result list. Cap at limit.

    Retrieval is **bump-free** in both modes — surfacing does not touch
    ``recall_count`` (only a deliberate ``read_full_memory`` → ``store.get()``
    bumps). ND-1 resolved (owner, 2026-08-14): ``store.update()``/
    ``store.deactivate()``'s internal existence-check now also passes
    ``bump=False``, so only a genuine full-read bumps ``recall_count``. See
    ``changes/p2-relevance/decisions.md``. The semantic path never calls
    ``store.get()`` at all — its candidate pool comes from
    ``store.list_active()`` (a plain SELECT with no bump parameter), matching
    that contract structurally rather than by convention.

    Returns
    -------
    dict with keys:
        query          — the original query string
        mode           — the retrieval path actually used ("semantic" or
                          "lexical" — differs from the requested mode only
                          on a semantic fail-soft fallback)
        emotion_filter — the emotion filter (or None)
        count          — number of results returned
        memories       — list of snippet-result dicts (``snippet: true`` + id)
    """
    exclude = frozenset(exclude_ids or ())

    candidates: list[Memory] | None = None
    resolved_mode: SearchMode = mode
    if mode == "semantic":
        candidates = _semantic_top_k(store, persona_dir, query, limit=limit, exclude=exclude)
        if candidates is None:
            resolved_mode = "lexical"
    if candidates is None:
        candidates = _lexical_candidates(store, hebbian, query, limit=limit, exclude=exclude)

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
        "mode": resolved_mode,
        "emotion_filter": emotion,
        "count": len(results),
        "memories": results,
    }
