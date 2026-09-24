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
(`brain/memory/reranker.py`) + a floor on the reranker's score —
query-conditioned, so a cutoff on it is trustworthy in a way a cosine floor
never was. Originally a FIXED, empirically-set module constant
(`RERANK_FLOOR`); cut over by F2a inc8 (#250 §7/§8) to a per-persona,
per-runtime-model floor read live from `MemoryStore.get_reranker_floor`,
derived+persisted daily against the actual corpus (`floor_calibration.py`).

Recall runs semantic cosine as a CHEAP COARSE CUT (narrow the pool before
the comparatively expensive reranker), then reranks the (auto-scaled-width)
survivors, then floor-gates the RERANKER score to decide relevance. When
that produces a CONCLUSIVE result (at least one candidate clears the
calibrated floor) that result is surfaced and the existing lexical path
never runs for that turn. When NOTHING clears the floor — or the candidate pool is
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

from brain.memory import embeddings as embeddings_mod
from brain.memory import reranker as reranker_mod
from brain.memory.embedding_matrix import EmbeddingMatrix, build_embedding_matrix
from brain.memory.embeddings import cosine_similarity
from brain.memory.relevance import CANDIDATE_POOL
from brain.memory.store import Memory, MemoryStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reranker abstention floor (#231 RERANKER RE-ARCHITECTURE, cut over to a
# DB-adaptive value by F2a inc8, #250 §7/§8): the bare `RERANK_FLOOR = -9.25`
# module constant this section used to hold is GONE. The floor is now read
# LIVE, per call, from `MemoryStore.get_reranker_floor(reranker_model_id)`
# (`brain/memory/store.py`'s `reranker_floor_calibration` table, written by
# the daily calibration tick — `brain.memory.floor_calibration.
# derive_and_persist_floor`, F2a inc7). Keyed by the RUNTIME reranker
# model_id (`reranker_provider.model_id()` — whichever of the fp32/fp16
# candidates the precision self-check actually shipped), so a precision
# flip or any future reranker swap never applies a floor fit against one
# score scale to scores from another.
#
# No hardcoded fallback value lives here — but as of F2a inc8's bootstrap
# ruling (#250 §7 UPDATED, Roy 2026-09-18) this is no longer a placeholder
# comment: `MemoryStore.get_reranker_floor` itself ALWAYS returns a servable
# floor when no persisted row exists yet, serving a derived, process-wide
# cached BOOTSTRAP floor instead of `None` (see
# `floor_calibration.get_bootstrap_floor`). This decouples semantic recall's
# EXISTENCE from the daily calibration tick ever having fired for the
# runtime model_id — the earlier design ("no row -> None -> fall back to
# lexical, exactly like an empty/sparse candidate pool") permanently
# coupled recall to the tick (disabled calibration, or a recall running
# before the tick's first idle moment, silently and PERMANENTLY demoted to
# lexical-only even with embeddings present) — see the spec's §7 UPDATED
# note for the full rationale. `run_semantic_recall` still returns `None`
# on a floor-read failure (the bootstrap computation's OWN fail-soft path —
# a reranker load/fit error), treated exactly like any other "semantic path
# not ready yet" precondition (empty/sparse candidate pool, embed failure,
# ...); it is just no longer the ROUTINE fresh-install/no-tick-yet case.
# Once the daily tick DOES persist a real corpus-derived floor for this
# model_id, that persisted row supersedes the bootstrap on every subsequent
# call — see `floor_calibration.derive_and_persist_floor`'s cold-start
# branch (a separate, PERSISTED cold-start fit the tick itself computes and
# writes, distinct from this module's transient, never-persisted bootstrap).
#
# `select_standouts` below takes the floor as an explicit parameter rather
# than reading a module global — the call site (this module's
# `run_semantic_recall`) is the one place with a `MemoryStore` and a
# resolved runtime model_id in scope to look it up.

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


def select_standouts(reranked_desc: list[tuple[str, float]], floor: float) -> SemanticSurfacing | None:
    """Floor-gate a sorted-descending (memory_id, reranker_score) list into
    surfacing tiers.

    Replaces `classify_semantic_shape`'s cosine-era standout/clump judgment:
    with a query-conditioned floor on the RERANKER score, every candidate is
    judged on its OWN merit — there is no more "bunched clump" to detect via
    a relative-gap scan. Every candidate whose score clears `floor` is a
    standout.

    `floor` (F2a inc8, #250 §7/§8 cutover) is the CALLER's resolved,
    per-persona, per-runtime-model calibrated floor
    (`MemoryStore.get_reranker_floor`) — this function stays a pure,
    directly-testable comparison, same shape as before the cutover, only the
    floor's SOURCE changed (out of scope per the spec: "F2a changes what the
    floor IS ... not where/how it's consulted").

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
    standouts = [(mid, score) for mid, score in reranked_desc if score >= floor]
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
    store: MemoryStore, matrix: EmbeddingMatrix
) -> dict[str, tuple[Memory, np.ndarray]]:
    """Active-STATE memories that already have a vector on their row under
    the warm matrix's model_id, paired with that vector.

    Deliberately NEVER computes a new embedding for an unembedded memory —
    that would be exactly the forbidden hot-path bulk embed. An active
    memory with no row vector yet (embed-on-write hasn't reached it, or the
    later idle backfill hasn't caught it up) simply isn't a semantic
    candidate this turn. This IS the "graceful warm-up" contract from the
    spec: semantic coverage grows as the corpus embeds; a cold/sparse
    persona degrades to the lexical fallback (empty pool here ->
    `run_semantic_recall` returns None) until backfill catches up.

    Sources vectors from `matrix.snapshot()` (F1 increment 2) — a
    `{memory_id: vector}` map keyed identically to the row's own id, so
    joining against `store.list_active()` is a direct id lookup, no
    content-hash join needed (the old `embeddings_cache.all_hashes_and_
    vectors()` + `hash_content(mem.content)` join this replaces).

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
    vectors = matrix.snapshot()
    if not vectors:
        return {}
    pool: dict[str, tuple[Memory, np.ndarray]] = {}
    for mem in store.list_active():
        if mem.state != "active":
            continue
        vec = vectors.get(mem.id)
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

    F2b (#276 §2): as of the per-query anchor-median normalization, this is
    the NORMALIZED score (`raw - median(anchor_scores)`, or raw unmodified
    on a near-degenerate-width no-op — see `reranker.normalize_against_
    anchors`) — the SAME value the floor gate actually compared against,
    never the pre-normalization raw reranker score. Anchor documents are
    never candidates, so they never appear here.
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
    spec decision 4) via the shared process-cached embedding provider
    (`build_embedding_provider()` — F1 #259 increment 8: the per-recall
    query embed is transient and is never cached/persisted, so it goes
    straight through the provider with no cache row to write), cosines it
    against the model_id-scoped candidate pool as a CHEAP COARSE CUT (top-
    `relevance.CANDIDATE_POOL`), reranks an auto-scaled-width slice of that
    coarse cut with a cross-encoder (`reranker.build_reranker_provider` +
    `reranker.get_rerank_width`), and floor-gates the reranker score
    (`select_standouts`, against the CALIBRATED floor read live via
    `store.get_reranker_floor(reranker_provider.model_id())` — F2a inc8,
    #250 §7/§8 cutover) to decide relevance (#231 RERANKER RE-ARCHITECTURE
    — replaces the pre-#231 cosine standout/clump classifier).

    Returns a populated `SemanticRecallResult` ONLY when at least one
    candidate clears the operative floor (persisted, or — F2a inc8, #250 §7
    UPDATED — the derived bootstrap when no persisted row exists yet; see
    the module-docstring note above `select_standouts`). Returns `None` for
    every INCONCLUSIVE case:
      - nothing clears the floor,
      - an empty or sparse candidate pool (cold-start / idle backfill not
        caught up — "graceful warm-up"),
      - the bootstrap computation itself failed (no persisted row AND the
        bootstrap fit raised — a reranker load/fit error) — this is now the
        ONLY way "no floor" demotes to lexical; a merely-absent persisted
        row no longer does, on its own, since the bootstrap always fills it,
      - ANY failure ANYWHERE in this function — constructing the local
        embedding/reranker provider, embedding the query, building the
        candidate pool, cosine scoring, reranking, reading the calibrated
        floor, or floor-gating (fail-soft: a broken/missing local model, a
        transient store error such as a locked sqlite db during the
        background backfill, or a floor-read error, must never break
        recall — it only demotes this turn to lexical-primary, matching the
        spec's warm-up contract, and — per the #231 build brief — a
        reranker failure demotes to the LEXICAL backstop, never to raw
        cosine ranking, the unreliable signal the reranker replaces). The
        whole body is wrapped in a broad `except Exception` for exactly
        this reason.

    Never renders anything and never bumps `recall_count` itself — the
    caller (`brain.chat.prompt._build_recall_block`) owns rendering and the
    recall-counter ticks, exactly as it already does for the lexical path.
    On `None`, the caller falls through to that EXISTING lexical/blend path,
    unchanged.
    """
    try:
        matrix = build_embedding_matrix(store.db_path)
        pool = build_semantic_candidate_pool(store, matrix)
        if not pool:
            return None
        try:
            # Looked up via the MODULE (not a bare imported name) so a
            # test's monkeypatch on `embeddings.build_embedding_provider` is
            # honored — mirrors `is_duplicate`'s/`MemoryStore.embed_row`'s
            # identical dynamic lookup. F1 #259 increment 8: the per-recall
            # query embed is transient (never persisted), so it goes
            # straight through the process-cached provider — no cache row
            # to write or evict.
            query_vec = embeddings_mod.build_embedding_provider().embed(user_input).astype("float32")
        except Exception:  # noqa: BLE001 — fail-soft
            log.exception("run_semantic_recall: query embed failed — falling back to lexical")
            return None

        cosine_scored = [
            (mid, cosine_similarity(query_vec, vec)) for mid, (_, vec) in pool.items()
        ]
        cosine_scored.sort(key=lambda pair: -pair[1])
        coarse = cosine_scored[:CANDIDATE_POOL]

        reranker_provider = reranker_mod.build_reranker_provider(store=store)
        # #231-fix: calibrate on REAL candidate-pool documents (a small
        # sample off the front of the already cosine-sorted `coarse`
        # list) rather than a synthetic placeholder — see
        # reranker.get_rerank_width's docstring.
        calibration_sample = [
            pool[mid][0].content
            for mid, _ in coarse[: reranker_mod.CALIBRATION_SAMPLE_SIZE]
        ]
        width = reranker_mod.get_rerank_width(len(coarse), reranker_provider, calibration_sample)
        to_rerank = coarse[:width]
        rerank_ids = [mid for mid, _ in to_rerank]
        real_documents = [pool[mid][0].content for mid in rerank_ids]
        # F2b (#276 §2/§4): one combined rerank() call (real candidates +
        # hardware-derived anchor count), normalized against the anchor
        # median BEFORE anything downstream (the log write, the floor gate)
        # ever sees a score — `normalize_against_anchors` is inc1's already-
        # built, already-tested helper; this call site only wires it in, it
        # does not reimplement any of its arithmetic. `scored_ids` is the
        # PREFIX of `rerank_ids` that was actually scored this call
        # (`result.real_width` <= `width` — fewer than `width` only when
        # anchors were reserved out of it, spec §3); `result.scores` is
        # positionally aligned with `scored_ids` 1:1. Anchors themselves
        # never appear in `result.scores`/`scored_ids` — this is computed
        # entirely inside the helper and never leaves it, so there is
        # nothing here that could leak an anchor id/content into the
        # log write, the gate, or the surfaced result below.
        normalization = reranker_mod.normalize_against_anchors(
            reranker_provider, user_input, real_documents, width
        )
        scored_ids = rerank_ids[: normalization.real_width]
        rerank_scores = normalization.scores
        try:
            # F2a (#250 inc4), re-pointed by F2b (#276 §5): real-query
            # calibration logging (spec Section 4/5). `user_input` is logged
            # byte-identical to what was just embedded/reranked above — no
            # synthetic/reconstructed query. `scored_ids`/`rerank_scores`
            # are the SAME already-computed, already-NORMALIZED per-turn
            # values that feed the floor gate just below (computed once,
            # above, reused as-is here) — never the raw pre-normalization
            # score, and never more ids than were actually scored this call
            # (`scored_ids`, not the full `rerank_ids`, when anchors
            # narrowed `real_width` below `width`). `log_calibration_sample`
            # stamps the current score_scale on this row itself. One
            # bounded INSERT, off the hot path in every sense but this
            # single cheap write (I6). Wrapped separately from the outer
            # fail-soft `except` below so a logging failure can NEVER demote
            # a good semantic result to the lexical fallback — it only
            # loses that one turn's calibration row.
            store.log_calibration_sample(
                query=user_input,
                candidate_ids=scored_ids,
                reranker_scores=rerank_scores,
                reranker_model_id=reranker_provider.model_id(),
            )
        except Exception:  # noqa: BLE001 — fail-soft: logging must never break recall
            log.warning(
                "run_semantic_recall: calibration log write failed — continuing",
                exc_info=True,
            )
        reranked = list(zip(scored_ids, rerank_scores, strict=True))
        reranked.sort(key=lambda pair: -pair[1])

        # F2a inc8 (#250 §7 UPDATED): the floor is read LIVE per call, keyed
        # by the RUNTIME reranker model_id (whichever of fp32/fp16 the
        # precision self-check actually shipped — matches how inc4's
        # calibration-log write and inc7's tick both key by
        # `reranker_provider.model_id()`). No persisted row yet (daily tick
        # has never fired for this model_id) no longer means INCONCLUSIVE —
        # `get_reranker_floor` serves a derived bootstrap instead (see this
        # module's own top-of-file comment). `floor_row is None` now fires
        # ONLY on the bootstrap's own fail-soft path (a reranker load/fit
        # failure); this check stays as the fail-soft demotion, never
        # invents a placeholder numeric floor itself.
        floor_row = store.get_reranker_floor(reranker_provider.model_id())
        if floor_row is None:
            log.info(
                "run_semantic_recall: no floor available (bootstrap computation failed) for %s — "
                "falling back to lexical",
                reranker_provider.model_id(),
            )
            return None

        # Observability (F2a inc8 scope item 5 — carries the red-team's MED
        # note): cheap, off the critical timing (one debug log line) —
        # lets live testing see which floor value actually gated this turn
        # and whether it's still the cold-start bootstrap or a real
        # corpus-derived fit, without adding per-turn work beyond the log
        # call itself.
        log.debug(
            "run_semantic_recall: floor=%.4f model=%s cold_start=%s "
            "sample_pairs=%d updated_at=%s",
            floor_row["floor"],
            reranker_provider.model_id(),
            floor_row["is_cold_start"],
            floor_row["sample_pairs"],
            floor_row["updated_at"],
        )

        tiers = select_standouts(reranked, floor_row["floor"])
        if tiers is None:
            return None

        full = [pool[mid][0] for mid in tiers.full_ids]
        snippet = [pool[mid][0] for mid in tiers.snippet_ids]
        return SemanticRecallResult(full=full, snippet=snippet, scores=dict(reranked))
    except Exception:  # noqa: BLE001 — fail-soft: ANY failure demotes to lexical, never raises
        log.warning(
            "run_semantic_recall: semantic path failed — falling back to lexical",
            exc_info=True,
        )
        return None
