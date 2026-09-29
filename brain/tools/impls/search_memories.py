"""search_memories tool implementation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from brain.memory import embeddings as embeddings_mod
from brain.memory.embedding_matrix import build_embedding_matrix
from brain.memory.embeddings import cosine_similarity
from brain.memory.hebbian import HebbianMatrix
from brain.memory.relevance import CANDIDATE_POOL, rank_memories, snippet_length
from brain.memory.semantic_recall import (
    build_semantic_candidate_pool,
    genuine_first_coarse_cut,
    rank_and_gate,
)
from brain.memory.store import Memory, MemoryStore
from brain.tools.impls._common import _mem_to_result

logger = logging.getLogger(__name__)

_CORECALL_DELTA = 0.1       # gentle nudge; cf. add_memory/ingest at 0.5
_CORECALL_FANOUT = 4        # anchor links to at most this many other results
_CORECALL_MIN_RESULTS = 2   # below this there is nothing to associate

SearchMode = Literal["semantic", "lexical"]
SearchOrder = Literal["relevance", "age"]


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


def _query_words(query: str) -> list[str]:
    """Every word of the kindled's own query, lower-cased and de-duplicated, in
    order. No stopword drop, no length floor, no cap (spec §5/§7, S81, which
    supersedes S57 for the tool): the query is deliberate and short, so a
    lowercase name not yet on the known-names list, or any other word she
    chose, is still searched."""
    seen: set[str] = set()
    words: list[str] = []
    for m in re.finditer(r"[A-Za-z0-9]+", query):
        low = m.group().lower()
        if low not in seen:
            seen.add(low)
            words.append(low)
    return words


def _keyword_candidates(
    store: MemoryStore,
    hebbian: HebbianMatrix,
    query: str,
    *,
    exclude: frozenset[str],
) -> list[Memory]:
    """The tool's keyword search, every candidate best-first (name-recall fix
    R4, spec §5, S81): BM25 text-match + importance + hebbian
    spreading-activation + recency via ``rank_memories`` over EVERY word of
    ``query`` (`_query_words`), one OR query, the store admitting each word.
    Monologue-family hits follow genuine ones (spec §4, S16): ``genuine_first``
    ranks them after every genuine match in the ranker's candidate pool and its
    final order, so a family hit never takes a slot from a genuine one.
    ``exclude`` ids are removed before ranking.

    R5 seam: the known-names query (its hits ahead of everything, S79) belongs
    in front of this list in BOTH modes; it is not built yet.
    """
    words = _query_words(query)
    if not words:
        return []
    ranked = rank_memories(
        store, hebbian, words, limit=CANDIDATE_POOL, exclude_ids=exclude, genuine_first=True
    )
    return [m for m, _ in ranked]


def _lexical_candidates(
    store: MemoryStore,
    hebbian: HebbianMatrix,
    query: str,
    *,
    limit: int,
    exclude: frozenset[str],
) -> list[Memory]:
    """``mode="lexical"``: the keyword search (``_keyword_candidates``), first
    ``limit`` results. Shares the emotion-boost/formatting tail below with the
    semantic mode."""
    return _keyword_candidates(store, hebbian, query, exclude=exclude)[:limit]


def _merge_keyword_below_semantic(
    semantic: list[Memory], keyword: list[Memory], *, cap: int
) -> list[Memory]:
    """The semantic results, then the keyword hits that are not already
    semantic hits, in the slots the semantic results leave under ``cap``
    (spec §5: keyword hits fill the leftover slots and take the next
    positions; a memory found by both keeps its semantic position, once; no
    slot is reserved for a keyword hit)."""
    taken = {m.id for m in semantic}
    room = max(0, cap - len(semantic))
    fill = [m for m in keyword if m.id not in taken][:room]
    return [*semantic, *fill]


def _semantic_top_k(
    store: MemoryStore,
    persona_dir: Path,
    query: str,
    *,
    limit: int,
    exclude: frozenset[str],
) -> list[Memory] | None:
    """Top-K semantically matched memories for an ACTIVE search call.

    #231 RERANKER RE-ARCHITECTURE (Q1, resolved 2026-09-10): embeds ``query``
    once via the shared process-cached embedding provider
    (``build_embedding_provider()``, cached by model_id — no per-call model
    reload; F1 #259 increment 8: the query embed is transient/never
    persisted, so it goes straight through the provider with no cache row
    to write), cosines it against every actively-cached memory vector
    (Stage 3's ``build_semantic_candidate_pool``: active memories that
    already have a cached vector under the current model_id — never
    triggers a new embed for an uncached memory, the same warm-up contract
    passive recall uses) as a CHEAP COARSE CUT to ``relevance.CANDIDATE_
    POOL`` (filled genuine-first, spec §4 S77), then scores the coarse cut with
    ``semantic_recall.rank_and_gate``
    — the SAME path choice, floor and scale passive recall uses (name-recall
    fix R2, spec §2): a per-message-width cross-encoder rerank gated by the
    calibrated, anchor-normalized rerank floor when >= 5 real candidates fit
    (F2b #276 §2/§4), otherwise (fewer than 5 fit or exist, or the reranker
    failed to load/score, or its normalization fell back) the cosine ranking
    gated by the cosine floor. A reranker failure no longer demotes this call
    to lexical. This site writes NO calibration row on either path (S56:
    calibration rows stay passive-recall only, as today).

    The CALIBRATED floor of the path taken decides semantic-vs-lexical: if
    NOTHING clears it — or no cosine gate can be had (the cosine bootstrap
    failed) — this returns ``None`` (the tool's EXISTING empty-semantic→
    lexical fallback — never returns nothing, never hands back semantic junk
    that never cleared a floor). Otherwise returns the top ``limit``
    floor-clearing memories in the path's own order (name-recall fix R3, spec
    §4): every genuine memory by descending score, then every monologue-family
    memory by descending score, so a monologue-family memory never takes a
    result slot from a floor-clearing genuine one. The rerank prefix already
    took genuine candidates first (``rank_and_gate``). That order is what the
    default ``order="relevance"`` returns; ``order="age"`` still re-sorts the
    matched set by date (and ``emotion`` still boosts) in ``search_memories``'
    unchanged tail (plan P-21), so a newer monologue-family memory can precede
    an older genuine one there by the caller's own request.

    Deliberately does NOT reuse ``semantic_recall``'s option-4 surfacing
    tiers (≤5 full / 6-9 / cap-at-9) — that machinery decides whether to
    surface an unsolicited passive-recall block at all, and how much of it
    to show in full vs snippet. Here the model explicitly asked for a
    search, so a plain top-k ranking is the natural "semantic search"
    behavior, mirroring how the lexical path is a plain top-k relevance
    ranking too.

    Returns ``None`` (never raises) when semantic search cannot run right
    now — no cached vectors yet, an embedding-model failure, or any other
    error anywhere in this path — so the caller falls back to the lexical
    path. Mirrors ``run_semantic_recall``'s fail-soft posture: the whole
    body is wrapped so a broken/missing local model or a transient store
    error only demotes this call to lexical, never breaks the tool.
    """
    try:
        matrix = build_embedding_matrix(store.db_path)
        pool = build_semantic_candidate_pool(store, matrix)
        if not pool:
            return None
        try:
            # Looked up via the MODULE (not a bare imported name) so a
            # test's monkeypatch on `embeddings.build_embedding_provider` is
            # honored — mirrors `run_semantic_recall`'s/`is_duplicate`'s
            # identical dynamic lookup.
            embedder = embeddings_mod.build_embedding_provider()
            query_vec = embedder.embed(query).astype("float32")
        except Exception:  # noqa: BLE001 — fail-soft
            logger.exception(
                "search_memories(semantic): query embed failed — falling back to lexical"
            )
            return None

        cosine_scored = [
            (mid, cosine_similarity(query_vec, vec))
            for mid, (_, vec) in pool.items()
            if mid not in exclude
        ]
        if not cosine_scored:
            return None
        # Spec §4, S77: the 50-candidate pool is filled genuine-first.
        coarse = genuine_first_coarse_cut(cosine_scored, pool)

        gated = rank_and_gate(
            store,
            query,
            pool,
            coarse,
            embedder_model_id=embedder.model_id(),
            log_calibration=False,
        )
        if gated is None:
            return None
        cleared = [(mid, score) for mid, score in gated.ranked if score >= gated.pass_mark]
        if not cleared:
            return None
        return [pool[mid][0] for mid, _ in cleared[:limit]]
    except Exception:  # noqa: BLE001 — fail-soft: ANY failure demotes to lexical, never raises
        logger.warning(
            "search_memories(semantic): semantic path failed — falling back to lexical",
            exc_info=True,
        )
        return None


def search_memories(
    query: str,
    emotion: str | None = None,
    limit: int = 5,
    exclude_ids: list[str] | None = None,
    mode: SearchMode = "semantic",
    order: SearchOrder = "relevance",
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Search memories by content relevance + optional emotion filter.

    ``mode`` picks the retrieval path (default ``"semantic"``):
      - ``"semantic"``: embeds ``query`` once and ranks the persona's cached
        memory vectors (see ``_semantic_top_k``). Meaning-based — catches a
        paraphrase with no shared keyword. Name-recall fix R4 (spec §5): the
        keyword search (below) is merged in under the semantic results,
        filling only the slots they leave under ``limit``; a memory found by
        both appears once, at its semantic position. Fails soft to
        ``"lexical"`` the instant semantic retrieval can't run right now (no
        cached vectors yet / embedding model unavailable / any embed error /
        nothing clearing a floor) — the returned ``mode`` reflects the path
        actually used ("semantic" iff at least one semantic result
        contributed).
      - ``"lexical"``: the blended keyword ranker — BM25 text-match +
        importance + hebbian spreading-activation + recency, via
        ``rank_memories`` (see ``_keyword_candidates``). EVERY word of the
        query is sent, with no stopword drop, no length floor and no cap (spec
        §5, S81), so 'Henryk preferences personality' finds memories
        mentioning ANY word, as a union, not the empty AND-intersection, and a
        lowercase name not yet on the known-names list is still found.

    ``order`` picks how the MATCHED set (whichever ``mode`` produced it) is
    ordered before the final ``limit`` slice (#231, Planning-signed-off
    option A):
      - ``"relevance"`` (default): today's behavior, byte-identical — the
        matched candidates are fetched at ``limit`` and used as-is, in
        whatever order ``mode`` already ranked them.
      - ``"age"``: WIDENS the internal fetch to ``CANDIDATE_POOL`` (today 50)
        for BOTH modes — lexical calls ``rank_memories(..., limit=
        CANDIDATE_POOL)``; semantic reranks up to ``CANDIDATE_POOL``
        floor-clearing candidates in ``_semantic_top_k`` (#231's calibrated
        reranker-floor gate still applies — "age" only widens the fetch,
        it never skips the floor) — THEN sorts that wider matched set by
        ``created_at`` DESC, THEN slices to
        the caller's real ``limit``. A naive re-sort of an already-``limit``-
        capped set would be subtly broken (it could only ever re-order the
        few candidates ``mode`` happened to already rank highest) — widening
        the fetch first is what makes an "age" ordering actually surface the
        newest matches, not just the newest of the top few relevance hits.
        Matching still happens FIRST by ``mode``; ``order`` only re-sorts the
        matched set, never changes which memories matched.

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
                          "lexical" — never echoes the raw requested value;
                          differs from a requested "semantic" only on a
                          fail-soft fallback, and any non-"semantic" request
                          — including an invalid/garbage value — is reported
                          as "lexical", the path that actually ran)
        resolved_order — the ordering actually applied ("relevance" or
                          "age" — any non-"age" requested value, including
                          an invalid/garbage one, is reported as "relevance")
        emotion_filter — the emotion filter (or None)
        count          — number of results returned
        memories       — list of snippet-result dicts (``snippet: true`` + id)
    """
    exclude = frozenset(exclude_ids or ())

    # #231 Fix 3 (owner ruling — resolved_mode must never lie): normalize
    # the requested mode up front rather than echoing it verbatim. Any value
    # other than "semantic" (including a garbage/invalid one — the enum is
    # advisory at the schema level, not enforced at the call boundary) is
    # treated as "lexical" from the start, so `resolved_mode` always names
    # the retrieval path that is actually about to run, before it runs.
    resolved_mode: SearchMode = "semantic" if mode == "semantic" else "lexical"

    # #231 Change 2: same advisory-enum posture as `mode` — any value other
    # than "age" (including garbage/invalid) normalizes to "relevance".
    resolved_order: SearchOrder = "age" if order == "age" else "relevance"

    # "age" widens the internal fetch to CANDIDATE_POOL so there is an
    # actually-wide matched set to age-sort before the real `limit` slice;
    # "relevance" fetches exactly `limit`, unchanged from before this
    # toggle existed — byte-identical default behavior.
    fetch_limit = CANDIDATE_POOL if resolved_order == "age" else limit

    candidates: list[Memory] | None = None
    if resolved_mode == "semantic":
        semantic = _semantic_top_k(store, persona_dir, query, limit=fetch_limit, exclude=exclude)
        if semantic is None:
            resolved_mode = "lexical"
        else:
            # Name-recall fix R4 (spec §5, S8/S35, P-21): the keyword search
            # merges in below the semantic results, filling the slots they leave
            # under the fetch limit. The reported mode stays "semantic": at
            # least one semantic result contributed.
            candidates = _merge_keyword_below_semantic(
                semantic,
                _keyword_candidates(store, hebbian, query, exclude=exclude),
                cap=fetch_limit,
            )
    if candidates is None:
        candidates = _lexical_candidates(store, hebbian, query, limit=fetch_limit, exclude=exclude)

    if resolved_order == "age":
        candidates = sorted(candidates, key=lambda m: m.created_at, reverse=True)

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
        "resolved_order": resolved_order,
        "emotion_filter": emotion,
        "count": len(results),
        "memories": results,
    }
