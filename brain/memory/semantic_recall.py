"""semantic_recall.py — semantic-PRIMARY retrieval + option-4 surfacing.

Stage 3 of the local-semantic-retrieval build
(``~/.claude/plans/memory-dream-rework-semantic-retrieval-brief.md``,
decisions 3-5, DIRECTION CORRECTED 2026-09-09: semantic is PRIMARY, the
existing lexical/importance/hebbian/recency blend is the FALLBACK/backstop),
RE-ARCHITECTED 2026-09-10 (#231, "RERANKER RE-ARCHITECTURE" section): the
cosine-era per-persona auto-calibration (`SemanticCalibration`,
`classify_semantic_shape`, `brain/memory/semantic_calibration.py`) is
REMOVED — the cold red-team proved deriving a floor/gap from the corpus's
own inter-memory cosine spread doesn't generalize (breaks silently on
tight/diffuse/bimodal corpora, because the query-match cosine scale is
MODEL-FIXED, not corpus-shaped). Replaced by a cross-encoder RERANKER
(`brain/memory/reranker.py`) + a FIXED, empirically-set floor
(`RERANK_FLOOR`, this module) on the reranker's score — query-conditioned,
so a fixed cutoff is trustworthy in a way a cosine floor never was.

Recall runs semantic cosine as a CHEAP COARSE CUT (narrow the pool before
the comparatively expensive reranker), then reranks the (auto-scaled-width)
survivors, then floor-gates the RERANKER score to decide relevance. When
that produces a CONCLUSIVE result (at least one candidate clears
`RERANK_FLOOR`) that result is surfaced and the existing lexical path never
runs for that turn. When NOTHING clears the floor — or the candidate pool is
empty/sparse (cold-start / idle backfill hasn't caught up yet — the
"graceful warm-up" contract), or any embedding/reranker-infra failure — this
module returns ``None`` and the caller (``brain.chat.prompt.
_build_recall_block``) falls through UNCHANGED to the existing lexical/blend
retrieval, exactly as it behaved before this stage. This module never
touches that fallback path.

This module owns:
  - the semantic candidate-pool builder (active-STATE memories that already
    have a cached vector under the CURRENT model_id — never triggers a new
    embed for an uncached memory; that bulk-embed job is Stage 2's, off this
    hot path)
  - the query embed (the ONE allowed synchronous in-turn embed, decision 4)
  - the cosine coarse-cut (cheap pre-filter to `relevance.CANDIDATE_POOL`)
  - the rerank call (`reranker.build_reranker_provider` +
    `reranker.get_rerank_width`, auto-scaled to a measured per-host latency)
  - the floor-gated standout selection (`select_standouts`) — replaces
    `classify_semantic_shape`'s cosine standout/clump judgment
  - the surfacing-tier decision (which candidate ids are "full" vs
    "snippet"). Rendering (actual body/snippet text, recall-counter ticks)
    stays owned by ``brain.chat.prompt``, mirroring how it already
    renders/bumps the lexical path — this module only decides WHICH ids go
    in which bucket, in reranker-SELECTION order; presentation order is the
    caller's call (spec: "the reranker selects, the normal sort orders the
    presentation").

Does NOT touch: the lexical/blend fallback itself (untouched, reused
as-is), the embed-on-write / idle-backfill machinery (Stage 2, unaffected),
or clustering (Stage 5, unaffected).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain.memory import reranker as reranker_mod
from brain.memory.embeddings import build_embedding_cache, cosine_similarity, hash_content
from brain.memory.relevance import CANDIDATE_POOL
from brain.memory.store import Memory, MemoryStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed reranker floor (#231 RERANKER RE-ARCHITECTURE) — the abstention
# decision. A plain module constant, NOT a tunables.py entry: this is
# PHYSIOLOGY (decides what the persona notices as relevant, same class as
# the salience/forgetting cutoffs tunables.py explicitly fences OUT) and it
# mirrors the SEMANTIC_FLOOR_BOOTSTRAP module constant it replaces. NEVER
# per-corpus-derived — that was the trap this re-architecture exists to
# kill. Only the auto-scale latency budget (`reranker.LATENCY_BUDGET_
# SECONDS`) is an ops tunable.
#
# SET EMPIRICALLY (2026-09-10) against the REAL `Xenova/ms-marco-MiniLM-L-
#6-v2` cross-encoder (fastembed 0.8.0), scoring the committed #88 pair
# (`tests/unit/brain/chat/test_semantic_primary_recall.py`) plus tight /
# diffuse / bimodal corpus-shape probes (throwaway script, not committed —
# see the #231 build report for the full score table). Cross-encoder scores
# are RAW, UNCALIBRATED logits (this model's range across the sample was
# roughly -11.5 .. +3.7) — NOT a [0, 1] probability; do not compare this
# value against a cosine score.
#
#   decoy/unrelated max (hard negatives, excluding one intentionally
#     ambiguous near-duplicate-topic probe) = -9.7442
#   genuine-match min (weakest real paraphrase across every shape)  = -7.8570
#
# Floor picked ~25% of the way from the decoy max toward the genuine min
# (biased toward the DECOY side per the build brief: "a false negative on a
# real match is worse; lexical fallback catches exact-name misses") —
# giving every genuine match in the sample a comfortable margin above the
# floor while sitting clearly above the worst decoy/unrelated score.
RERANK_FLOOR = -9.25

# The largest standout cluster the surfacing rule ever recognises. NOT a
# tunable, part of the fixed shape of the three-tier surfacing rule.
# Corrected 2026-09-10 (Fixing finding (i)): the old cosine-era "10+ = clump
# -> lexical" bucket is DROPPED — a trustworthy per-candidate reranker floor
# means 10+ above-floor candidates are 10+ genuinely relevant results, not
# an undifferentiated clump, so they're simply capped at this count instead
# of demoted to the lexical fallback.
MAX_STANDOUT_COUNT = 9

# Surfacing-tier boundary (decision 5, option 4) — the fixed SHAPE of the
# three-tier rule.
FULL_INJECT_STANDOUT_MAX = 5  # <=5 clear standouts: all rendered in full


@dataclass(frozen=True)
class SemanticSurfacing:
    """The surfacing-tier id split for a CONCLUSIVE (at least one
    above-floor candidate) reranked result.

    Both lists are in reranker-SELECTION order (highest reranker score
    first) — the reranker decides membership and ranking of the standout
    set; the caller decides PRESENTATION order for the snippet tier (spec:
    "the reranker selects, the normal sort orders the presentation").
    """

    full_ids: list[str]
    snippet_ids: list[str]


def select_standouts(reranked_desc: list[tuple[str, float]]) -> SemanticSurfacing | None:
    """Floor-gate a sorted-descending (memory_id, reranker_score) list into
    surfacing tiers.

    Replaces `classify_semantic_shape`'s cosine-era standout/clump judgment:
    with a query-conditioned, empirically-fixed floor on the RERANKER score,
    every candidate is judged on its OWN merit — there is no more "bunched
    clump" to detect via a relative-gap scan. Every candidate whose score
    clears `RERANK_FLOOR` is a standout.

    Returns ``None`` when NOTHING clears the floor — INCONCLUSIVE, the
    caller falls back to lexical (matching `run_semantic_recall`'s existing
    contract).

    Tiers (decision 5, option 4 — UNCHANGED by this re-architecture, only
    the score source and floor mechanism changed):
      - <=5 standouts: all rendered in full.
      - 6-9 standouts: top 5 full, the rest snippet.
      - >=10 standouts: capped at `MAX_STANDOUT_COUNT` (top 5 full + 4
        snippet) — NOT demoted to lexical (see that constant's docstring).

    `reranked_desc` must already be sorted descending by score (the
    caller's job — this function trusts the ordering, mirroring the old
    `classify_semantic_shape`/`surfacing_tiers` contract).
    """
    standouts = [(mid, score) for mid, score in reranked_desc if score >= RERANK_FLOOR]
    if not standouts:
        return None
    capped_ids = [mid for mid, _ in standouts[:MAX_STANDOUT_COUNT]]
    if len(capped_ids) <= FULL_INJECT_STANDOUT_MAX:
        return SemanticSurfacing(full_ids=capped_ids, snippet_ids=[])
    return SemanticSurfacing(
        full_ids=capped_ids[:FULL_INJECT_STANDOUT_MAX],
        snippet_ids=capped_ids[FULL_INJECT_STANDOUT_MAX:],
    )


def build_semantic_candidate_pool(
    store: MemoryStore, embeddings_cache
) -> dict[str, tuple[Memory, np.ndarray]]:
    """Active-STATE memories that already have a cached vector under THIS
    cache's model_id, paired with that vector.

    Deliberately NEVER computes a new embedding for an uncached memory —
    that would be exactly the forbidden hot-path bulk embed. An active
    memory with no cached vector yet (the idle backfill hasn't reached it)
    simply isn't a semantic candidate this turn. This IS the "graceful
    warm-up" contract from the spec: semantic coverage grows as the corpus
    embeds; a cold/sparse persona degrades to the lexical fallback (empty
    pool here -> `run_semantic_recall` returns None) until backfill catches
    up.

    Fold-in fix (b), #231 (2026-09-10): filters to ``mem.state == "active"``.
    ``store.list_active()`` filters only the ``active`` deactivation flag,
    NOT ``state`` — a memory in ``state="fading"`` is still ``active=1`` and
    would otherwise enter this pool, rendering under BOTH the "active:"
    section (if its summary happened to be embedded and score well) AND the
    "softened (fading)" section, double-bumping it. Filtering here makes the
    semantic candidate pool structurally disjoint from the fading partition
    `_build_recall_block` computes separately — no downstream dedup needed
    (see the corrected comment at that call site, fold-in fix (b)).

    Uses `store.list_active()` rather than a per-id `store.get()` loop:
    `list_active()` is a plain SELECT with no bump parameter at all, so this
    scan structurally cannot inflate `recall_count`/`last_accessed_at` —
    scoring must never tick the counter (only actual surfacing does, via the
    caller's `bump_recall` calls on the final selected set).
    """
    hash_to_vector = dict(embeddings_cache.all_hashes_and_vectors())
    if not hash_to_vector:
        return {}
    pool: dict[str, tuple[Memory, np.ndarray]] = {}
    for mem in store.list_active():
        if mem.state != "active":
            continue
        vec = hash_to_vector.get(hash_content(mem.content))
        if vec is not None:
            pool[mem.id] = (mem, vec)
    return pool


@dataclass(frozen=True)
class SemanticRecallResult:
    """A CONCLUSIVE semantic-primary recall — the caller renders this and
    skips the lexical fallback entirely for this turn.

    `full` / `snippet` are Memory lists in reranker-SELECTION order;
    `scores` maps memory_id -> reranker score for callers that want the raw
    number (tests, logging). NOTE: unlike the pre-#231 cosine-era version,
    `scores` only covers candidates that were actually reranked (the
    auto-scaled-width slice of the cosine coarse-cut), not the whole pool.
    """

    full: list[Memory]
    snippet: list[Memory]
    scores: dict[str, float]


def run_semantic_recall(
    store: MemoryStore,
    persona_dir: Path,
    user_input: str,
) -> SemanticRecallResult | None:
    """Attempt semantic-PRIMARY recall for one turn.

    Embeds `user_input` (~34ms, synchronous — the ONE allowed in-turn embed,
    spec decision 4) via this persona's embedding cache/provider, cosines it
    against the model_id-scoped candidate pool as a CHEAP COARSE CUT (top-
    `relevance.CANDIDATE_POOL`), reranks an auto-scaled-width slice of that
    coarse cut with a cross-encoder (`reranker.build_reranker_provider` +
    `reranker.get_rerank_width`), and floor-gates the reranker score
    (`select_standouts`, `RERANK_FLOOR`) to decide relevance (#231 RERANKER
    RE-ARCHITECTURE — replaces the pre-#231 cosine standout/clump
    classifier).

    Returns a populated `SemanticRecallResult` ONLY when at least one
    candidate clears `RERANK_FLOOR`. Returns `None` for every INCONCLUSIVE
    case:
      - nothing clears the floor,
      - an empty or sparse candidate pool (cold-start / idle backfill not
        caught up — "graceful warm-up"),
      - ANY failure ANYWHERE in this function — constructing the local
        embedding/reranker provider/cache, embedding the query, building
        the candidate pool, cosine scoring, reranking, or floor-gating
        (fail-soft: a broken/missing local model, or a transient store
        error such as a locked sqlite db during the background backfill,
        must never break recall — it only demotes this turn to
        lexical-primary, matching the spec's warm-up contract, and — per
        the #231 build brief — a reranker failure demotes to the LEXICAL
        backstop, never to raw cosine ranking, the unreliable signal the
        reranker replaces). The whole body is wrapped in a broad `except
        Exception` for exactly this reason.

    Never renders anything and never bumps `recall_count` itself — the
    caller (`brain.chat.prompt._build_recall_block`) owns rendering and the
    recall-counter ticks, exactly as it already does for the lexical path.
    On `None`, the caller falls through to that EXISTING lexical/blend path,
    unchanged.
    """
    try:
        embeddings_cache = build_embedding_cache(persona_dir)
    except Exception:  # noqa: BLE001 — fail-soft: never break recall
        log.exception("run_semantic_recall: failed to open embedding cache — falling back to lexical")
        return None
    try:
        try:
            pool = build_semantic_candidate_pool(store, embeddings_cache)
            if not pool:
                return None
            try:
                query_vec = embeddings_cache.embed_query(user_input)
            except Exception:  # noqa: BLE001 — fail-soft
                log.exception("run_semantic_recall: query embed failed — falling back to lexical")
                return None

            cosine_scored = [
                (mid, cosine_similarity(query_vec, vec)) for mid, (_, vec) in pool.items()
            ]
            cosine_scored.sort(key=lambda pair: -pair[1])
            coarse = cosine_scored[:CANDIDATE_POOL]

            reranker_provider = reranker_mod.build_reranker_provider()
            width = reranker_mod.get_rerank_width(len(coarse), reranker_provider)
            to_rerank = coarse[:width]
            rerank_ids = [mid for mid, _ in to_rerank]
            documents = [pool[mid][0].content for mid in rerank_ids]
            rerank_scores = list(reranker_provider.rerank(user_input, documents))
            reranked = list(zip(rerank_ids, rerank_scores, strict=True))
            reranked.sort(key=lambda pair: -pair[1])

            tiers = select_standouts(reranked)
            if tiers is None:
                return None

            full = [pool[mid][0] for mid in tiers.full_ids]
            snippet = [pool[mid][0] for mid in tiers.snippet_ids]
            return SemanticRecallResult(full=full, snippet=snippet, scores=dict(reranked))
        except Exception:  # noqa: BLE001 — fail-soft: ANY failure demotes to lexical, never raises
            log.warning(
                "run_semantic_recall: semantic path failed after opening the embedding cache "
                "— falling back to lexical",
                exc_info=True,
            )
            return None
    finally:
        # #231 Fix 4: this `finally` sat outside the inner `except Exception`
        # above, so a pathological `close()` error could escape this
        # function's documented "never raises" contract (previously caught
        # only by the caller-side wrap in `_build_recall_block`). Make
        # cleanup self-contained: a close error is caught/logged here and
        # never propagates, matching the "ANY failure ... never raises"
        # contract this function's own docstring states.
        try:
            embeddings_cache.close()
        except Exception:  # noqa: BLE001 — fail-soft: close() must never break the contract
            log.warning("run_semantic_recall: embeddings_cache.close() failed", exc_info=True)
