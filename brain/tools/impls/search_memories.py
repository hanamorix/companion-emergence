"""search_memories tool implementation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from brain.memory import keyword_budget
from brain.memory.hebbian import HebbianMatrix
from brain.memory.relevance import (
    CANDIDATE_POOL,
    lead_with_names,
    names_in,
    rank_memories,
    rank_name_hits,
    snippet_length,
)
from brain.memory.semantic_recall import (
    assemble_with_guarantees,
    genuine_first_memories,
    merge_with_keyword,
    search_paragraphs,
)
from brain.memory.store import Memory, MemoryStore, split_by_raw_query_floor
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


def _keyword_words(query: str, persona_dir: Path | None) -> frozenset[str]:
    """The tool's keyword token set for the per-paragraph emptiness test (spec
    §3): the words the store's raw-query builder admits from the kindled's own
    query (`split_by_raw_query_floor`, today's 3-character floor, no stopword
    drop, S99), plus the words of the known names in it (tested after name
    protection, so a paragraph holding only a listed name is kept). A paragraph
    with none of these words is dropped, exactly the words the keyword side of
    this call would search."""
    words = [m.group().lower() for m in re.finditer(r"[A-Za-z0-9]+", query)]
    kept, _short = split_by_raw_query_floor(words)
    names = names_in(persona_dir, query)
    return frozenset(kept) | {word for name in names for word in name.split()}


def _keyword_candidates(
    store: MemoryStore,
    hebbian: HebbianMatrix,
    query: str,
    *,
    exclude: frozenset[str],
    persona_dir: Path | None = None,
    started: float | None = None,
) -> list[Memory]:
    """The tool's keyword search, every candidate best-first (both modes): today's
    ranker exactly, plus the name query (name-recall fix R4/R5, spec §5/§7, S99).

    The general search is what this tool has always done: the raw ``query``
    string through ``rank_memories`` (BM25 text-match + importance + hebbian
    spreading-activation + recency), which the store tokenizes itself (3+
    character words, OR-joined). No every-word rule, no tiers, no cap change and
    no short-token admission for the tool (S99: finer kindled control over the
    tool is its own issue, #319); ``exclude`` ids are removed before ranking.
    The monologue family ranks after every genuine hit, in the candidate pool
    and in the result (``genuine_first``, spec §4 / criterion 8, which names the
    tool; not one of the S81 rules S99 removes).

    Name protection (R5, spec §5, S27, S35), BOTH modes: the known names in her
    query (matched on the raw words, before any stopword or length rule;
    ``persona_dir`` locates the list) run ONE extra keyword query, sent as FTS
    phrases; its hits lead this list (S89: those the general search also found
    first, then name-only hits, then the general hits). No graveyard is involved
    here. Without a listed name in her query the result is exactly today's.

    ``started`` (name-recall fix S145, S155, S157): the budget clock reading at
    the top of the tool call; the name query's continuation walk runs under the
    per-message recall time budget from it (`rank_name_hits`).
    """
    ranked = rank_memories(
        store, hebbian, query, limit=CANDIDATE_POOL, exclude_ids=exclude, genuine_first=True
    )
    general = [m for m, _ in ranked]
    names = names_in(persona_dir, query)
    name_ranked = rank_name_hits(
        store, hebbian, names, limit=CANDIDATE_POOL, exclude_ids=exclude, started=started
    )
    # The monologue family follows every genuine hit (spec §4, S16, criterion 8:
    # "in passive recall and the tool"); a separate rule from the words searched.
    return genuine_first_memories(lead_with_names([m for m, _ in name_ranked], general, names))


def _lexical_candidates(
    store: MemoryStore,
    hebbian: HebbianMatrix,
    query: str,
    *,
    limit: int,
    exclude: frozenset[str],
    persona_dir: Path | None = None,
    started: float | None = None,
) -> list[Memory]:
    """``mode="lexical"``: today's keyword ranker plus the name query (R5, S99),
    the first ``limit`` results. Shares the emotion-boost/formatting tail below
    with the semantic mode."""
    hits = _keyword_candidates(
        store, hebbian, query, exclude=exclude, persona_dir=persona_dir, started=started
    )
    return hits[:limit]


@dataclass(frozen=True)
class _SemanticTopK:
    """The semantic side of a call (name-recall fix R6, S97/S98): each
    result with its result group, in the semantic order, and the guaranteed
    ids in their S64 order (``semantic_recall.merge_with_keyword`` input)."""

    grouped: list[tuple[Memory, str]]
    guaranteed: list[str]


def _semantic_top_k(
    store: MemoryStore,
    persona_dir: Path,
    query: str,
    *,
    limit: int,
    exclude: frozenset[str],
) -> _SemanticTopK | None:
    """Top-K semantically matched memories for an ACTIVE search call.

    #231 RERANKER RE-ARCHITECTURE (Q1, resolved 2026-09-10), per paragraph
    since name-recall fix R6 (spec §3, S29: "the tool's explicit query is also
    split by paragraph"): ``semantic_recall.search_paragraphs`` splits
    ``query`` into paragraphs, embeds them in one batch through the shared
    process-cached embedding provider, cosines each against every active
    memory that already has a cached vector under the current model_id (never
    triggers a new embed for an uncached memory, the warm-up contract passive
    recall uses; ``exclude`` ids are removed first), cuts each paragraph's
    pool to the top 50 genuine plus the family in its plain top 50 (spec §4
    S77), merges the cuts (a memory goes to the paragraph with its best
    cosine) and scores each paragraph with ``semantic_recall.rank_and_gate``
    within the one per-message time budget: the SAME path choice, floor and
    scale passive recall uses (name-recall fix R2, spec §2), a
    per-paragraph-width cross-encoder rerank gated by the calibrated,
    anchor-normalized rerank floor when >= 5 real candidates fit its share
    (F2b #276 §2/§4), otherwise (fewer than 5 fit or exist, or the reranker
    failed to load/score, or its normalization fell back) the cosine ranking
    gated by the cosine floor. A reranker failure no longer demotes this call
    to lexical. This site writes NO calibration row on either path (S56:
    calibration rows stay passive-recall only, as today).

    The CALIBRATED floor of the path taken decides semantic-vs-lexical: if
    NOTHING clears it — or no cosine gate exists yet (the cosine bootstrap
    was not computed yet, or failed) — this returns ``None`` (the tool's EXISTING empty-semantic→
    lexical fallback — never returns nothing, never hands back semantic junk
    that never cleared a floor). Otherwise returns up to ``limit``
    floor-clearing memories, each with its result group, and the guaranteed
    ids, in the spec §4 order across paragraphs
    (``semantic_recall.assemble_with_guarantees``; the caller merges the
    keyword hits in with ``merge_with_keyword``, S97/S98): reranked genuine,
    reranked monologue-family, cosine-path genuine (each cosine-path
    paragraph's best genuine memory at its head), cosine-scale
    monologue-family (including every reranked paragraph's cosine tail, S82,
    S86), each result gated by its own scale's floor only, each paragraph's
    best genuine memory guaranteed a slot within ``limit`` (S33, S55, S64).
    That order is what the default ``order="relevance"`` returns;
    ``order="age"`` still re-sorts the matched set by date (and ``emotion``
    still boosts) in ``search_memories``' unchanged tail (plan P-21), so a
    newer monologue-family memory can precede an older genuine one there by
    the caller's own request.

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
        # Name-recall fix R6 (spec §3, S29): her query is searched per
        # paragraph like passive recall (same split, pool, budget, allocation
        # and per-paragraph rerank or cosine path; no calibration row, S56).
        # A paragraph with none of the words this call's keyword side
        # searches (`_keyword_words`, S99) is dropped.
        search = search_paragraphs(
            store,
            query,
            keyword_words=_keyword_words(query, persona_dir),
            exclude=exclude,
            log_calibration=False,
        )
        if search is None:
            return None
        # The per-paragraph assembly (spec §3/§4, S64): each result faces only
        # its own scale's floor; guaranteed slots per paragraph within `limit`.
        assembled, guaranteed = assemble_with_guarantees(search, limit)
        if not assembled:
            return None
        return _SemanticTopK(
            grouped=[(search.pool[hit.memory_id][0], hit.group) for hit in assembled],
            guaranteed=guaranteed,
        )
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
        keyword search (below) is merged with the semantic results in the
        six-group order of ``dev_constants.RECALL_RESULT_GROUP_ORDER`` (spec
        §4, S97/S98: keyword genuine hits above the non-reranked
        monologue-family results), under ``limit``; a memory found by both
        appears once, as a semantic result. Fails soft to
        ``"lexical"`` the instant semantic retrieval can't run right now (no
        cached vectors yet / embedding model unavailable / any embed error /
        nothing clearing a floor) — the returned ``mode`` reflects the path
        actually used ("semantic" iff at least one semantic result
        contributed).
      - ``"lexical"``: today's blended keyword ranker — BM25 text-match +
        importance + hebbian spreading-activation + recency, via
        ``rank_memories`` (see ``_keyword_candidates``). The raw multi-word
        query is passed straight through, exactly as before this build: its
        tokenize+OR split lives in ``store._to_fts_match`` (so 'Henryk
        preferences personality' finds memories mentioning ANY token, as a
        union, not the empty AND-intersection), including the store's
        3-character floor (spec §7, S99: finer kindled control over which
        words this tool searches is its own issue, #319). The one addition is
        name protection (R5, spec §5): a name on the known-names list (matched
        on her raw words, before any stopword or length rule) adds the one
        name query, whose hits lead the keyword results in both modes.

    ``order`` picks how the MATCHED set (whichever ``mode`` produced it) is
    ordered before the final ``limit`` slice (#231, Planning-signed-off
    option A):
      - ``"relevance"`` (default): the matched candidates are used as-is
        (sliced to ``limit``), in whatever order ``mode`` already ranked them.
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
    # The per-call budget clock (S132, S157): what is left of the recall budget
    # for the name query's continuation is measured from here.
    started = keyword_budget.now()
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
    # toggle existed.
    fetch_limit = CANDIDATE_POOL if resolved_order == "age" else limit

    candidates: list[Memory] | None = None
    if resolved_mode == "semantic":
        semantic = _semantic_top_k(store, persona_dir, query, limit=fetch_limit, exclude=exclude)
        if semantic is None:
            resolved_mode = "lexical"
        else:
            # Name-recall fix R4 (spec §5, S8/S35, P-21): the keyword search
            # (today's, plus the name query, S99) merges in below the semantic
            # results, filling the slots they leave under the fetch limit. The
            # reported mode stays "semantic": at least one semantic result
            # contributed.
            keyword = _keyword_candidates(
                store, hebbian, query, exclude=exclude, persona_dir=persona_dir, started=started
            )
            # Spec §4, S97/S98: the six result groups in the dev-constants
            # order, each paragraph's guaranteed slot kept first.
            candidates = merge_with_keyword(
                semantic.grouped, keyword, cap=fetch_limit, guaranteed=semantic.guaranteed
            )
    if candidates is None:
        candidates = _lexical_candidates(
            store,
            hebbian,
            query,
            limit=fetch_limit,
            exclude=exclude,
            persona_dir=persona_dir,
            started=started,
        )

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
