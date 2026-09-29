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
the comparatively expensive reranker), then reranks the (per-message-width)
survivors, then floor-gates the RERANKER score to decide relevance. When
that produces a CONCLUSIVE result (at least one candidate clears the
calibrated floor) that result leads the "active:" section; since name-recall
fix R4 (spec §5) the keyword search still runs every turn and fills the
slots the semantic results leave, instead of being suppressed.

Name-recall fix R2 (spec §2): the reranker is no longer the only way to a
semantic result. When fewer than 5 real candidates fit the rerank budget or
exist, the reranker fails to load or score, or its anchor normalization has
no median, the turn takes the NO-RERANK path: the coarse cut's candidates are
ranked by cosine and gated by a COSINE floor (`MemoryStore.get_cosine_floor`,
its own table, bootstrapped from the same bundled pairs until the daily tick
calibrates it). A reranker failure therefore no longer demotes the turn to
keyword-only. The two scales never mix: a reranked candidate is gated by the
normalized rerank floor, a cosine-path candidate by the cosine floor, and each
path's calibration row carries its own true scale. The module returns `None`
(the caller falls through UNCHANGED to the existing lexical/blend retrieval)
only for an empty/sparse pool (cold-start "graceful warm-up"), an embed
failure, a cosine bootstrap failure (no gate is possible), or when nothing
clears the floor of the path taken. This module never touches that fallback
path.

This module owns:
  - the semantic candidate-pool builder (active-STATE memories that already
    have a cached vector under the CURRENT model_id — never triggers a new
    embed for an uncached memory; that bulk-embed job is Stage 2's, off this
    hot path)
  - the query embed (the ONE allowed synchronous in-turn embed, decision 4)
  - the cosine coarse-cut (cheap pre-filter to `relevance.CANDIDATE_POOL`,
    genuine-first: the top 50 genuine plus the family in the plain top 50,
    `genuine_first_coarse_cut`, spec §4, S77)
  - the rerank call (`reranker.build_reranker_provider` +
    `reranker.rerank_for_recall`, its width fitted per message to the
    measured rerank cost on this host) and its no-rerank alternative
    (`rank_and_gate`: reranked path or cosine path, one gate per scale)
  - the floor-gated standout selection (`select_standouts`) — replaces
    `classify_semantic_shape`'s cosine standout/clump judgment
  - the surfacing-tier decision (which candidate ids are "full" vs
    "snippet"). Rendering (actual body/snippet text, recall-counter ticks)
    stays owned by ``brain.chat.prompt``, mirroring how it already
    renders/bumps the lexical path — this module only decides WHICH ids go
    in which bucket, in the path's own order (name-recall fix R3, spec §4:
    genuine memories first, then the monologue family, each by the path's
    score); the caller renders that order as it is (R4, plan P-11: no
    presentation re-sort) with the keyword hits merged in below it.

Does NOT touch: the lexical/blend fallback itself (untouched, reused
as-is), the embed-on-write / idle-backfill machinery (Stage 2, unaffected),
or clustering (Stage 5, unaffected).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from brain.dev_constants import MONOLOGUE_FAMILY_TYPES
from brain.memory import embeddings as embeddings_mod
from brain.memory import reranker as reranker_mod
from brain.memory.embedding_matrix import EmbeddingMatrix, build_embedding_matrix
from brain.memory.embeddings import cosine_similarity
from brain.memory.relevance import CANDIDATE_POOL
from brain.memory.store import (
    CALIBRATION_SCORE_SCALE,
    COSINE_SCORE_SCALE,
    Memory,
    MemoryStore,
)

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
# note for the full rationale. A floor-read failure (the bootstrap
# computation's OWN fail-soft path — a reranker load/fit error) means the
# reranker cannot gate this turn, so (name-recall fix R2, spec §2) the turn
# takes the cosine path instead (`rank_and_gate`); it is no longer the
# ROUTINE fresh-install/no-tick-yet case either.
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

    Both lists are in the path's own order, as handed to `select_standouts`
    (name-recall fix R3: genuine memories by score, then monologue-family
    memories by score) — the path decides membership and ranking of the
    standout set, and the caller renders that order unchanged (name-recall
    fix R4, plan P-11: the snippet tier is no longer re-sorted).
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

    `reranked_desc` must already be in the path's final order (the caller's
    job — this function trusts the ordering, mirroring the old
    `classify_semantic_shape`/`surfacing_tiers` contract): sorted descending by
    score, except that (name-recall fix R3, spec §4) `rank_and_gate` places
    every genuine memory ahead of every monologue-family one, so the
    `MAX_STANDOUT_COUNT` cut below never drops a floor-clearing genuine
    memory while keeping a monologue-family one.
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

    `full` / `snippet` are Memory lists in the path's own order (name-recall
    fix R3: genuine memories by score, then monologue-family memories by
    score; `full` is the first five). `hits` is the SAME list as one ordered,
    path- and paragraph-tagged sequence, DERIVED from `full` + `snippet` by
    `run_semantic_recall` (a result built by hand leaves it empty). The
    R4 prompt assembly reads `full` + `snippet` only, never `hits`, so there is
    one source of truth.
    `scores` maps memory_id -> reranker score for callers that want the raw
    number (tests, logging). NOTE: unlike the pre-#231 cosine-era version,
    `scores` only covers candidates that were actually scored (the
    per-message-width prefix of the coarse cut on the reranked path, the
    whole coarse cut on the cosine path), not the whole pool.

    F2b (#276 §2): as of the per-query anchor-median normalization, this is
    the NORMALIZED score (`raw - median(anchor_scores)`, or raw unmodified
    on a near-degenerate-width no-op — see `reranker.normalize_against_
    anchors`) — the SAME value the floor gate actually compared against,
    never the pre-normalization raw reranker score. Anchor documents are
    never candidates, so they never appear here.

    Name-recall fix R2: on the cosine path (`path == "cosine"`) `scores` are
    the raw cosine similarities of the coarse cut and `pass_mark` is the
    cosine floor; the two scales are never mixed within one result.
    """

    full: list[Memory]
    snippet: list[Memory]
    scores: dict[str, float]
    # Name-recall fix R3 (plan R3 row): the same standouts as ONE ordered list
    # (`full` then `snippet`, the path's own order), each tagged with the path
    # that scored it and the paragraph that produced it (diagnostics and R6's
    # per-paragraph assembly; the R4 prompt assembly reads `full` + `snippet`).
    hits: list[SemanticHit] = field(default_factory=list)
    # Name-recall fix R2 (spec §2, C2c): which path produced this result, the
    # scale `scores` are on ('normalized' reranker scores or raw 'cosine') and
    # the pass mark that was actually applied to them. Never mixed: a result
    # is wholly one path.
    path: str = "reranked"
    scale: str = CALIBRATION_SCORE_SCALE
    pass_mark: float | None = None


RERANKED_PATH = "reranked"
COSINE_PATH = "cosine"

# Every semantic result is one query today, so its paragraph tag is 0; R6's
# per-paragraph search assigns the real index.
SINGLE_QUERY_PARAGRAPH = 0


@dataclass(frozen=True)
class SemanticHit:
    """One surfaced semantic result, tagged for the final assembly (plan R3
    row, P-10): its `path` ("reranked" / "cosine"), the `paragraph` that
    produced it, the `score` on that path's scale, and whether it belongs to
    the monologue family (spec §4)."""

    memory: Memory
    score: float
    path: str
    paragraph: int
    monologue_family: bool


def is_monologue_family(memory: Memory) -> bool:
    """True for the kindled's own generated monologue memories (spec §4,
    S16): the types in `MONOLOGUE_FAMILY_TYPES`. Genuine memories are
    everything else."""
    return memory.memory_type in MONOLOGUE_FAMILY_TYPES


def genuine_first_memories(memories: list[Memory]) -> list[Memory]:
    """`memories` with every genuine memory ahead of every monologue-family
    one, each group keeping its incoming order (a stable partition). Used for
    the KEYWORD path (name-recall fix R4, spec §4 final order): keyword
    monologue-family hits follow keyword genuine hits, no score multiplier."""
    genuine = [m for m in memories if not is_monologue_family(m)]
    family = [m for m in memories if is_monologue_family(m)]
    return genuine + family


def genuine_first(ids: list[str], pool: dict[str, tuple[Memory, np.ndarray]]) -> list[str]:
    """`ids` with every genuine memory ahead of every monologue-family one,
    each group keeping its incoming (cosine) order. Used for the rerank
    prefix: the width fit and the reranker take genuine candidates first
    (spec §4, S13/S16; no score multiplier)."""
    genuine = [mid for mid in ids if not is_monologue_family(pool[mid][0])]
    family = [mid for mid in ids if is_monologue_family(pool[mid][0])]
    return genuine + family


def genuine_first_coarse_cut(
    cosine_scored: list[tuple[str, float]],
    pool: dict[str, tuple[Memory, np.ndarray]],
    size: int = CANDIDATE_POOL,
) -> list[tuple[str, float]]:
    """One query's coarse cut (name-recall fix R3 follow-up, spec §4, S77
    revised): the `size` best GENUINE memories by descending cosine, then the
    monologue-family memories that sit in today's plain cosine top-`size`, by
    descending cosine. No new constant, no score multiplier, scores unchanged.

    The cut can therefore hold more than `size` entries (up to 2 x `size`),
    and it guarantees both halves of spec §4/S16: a family memory can never
    keep a genuine one out of it (genuine memories fill their own `size`
    places), and a family memory that plain cosine would have admitted is
    still in it (so it can surface, ranked after the genuine ones). Rerank
    slots go to this order's prefix, i.e. genuine first: family memories are
    reranked only if width remains, and `rerank_for_recall` only ever sees the
    first `CANDIDATE_POOL` documents.

    One pass over the candidates: a single sort by cosine, then a walk that
    stops as soon as the plain top-`size` is behind it and `size` genuine
    memories are held (the family flag is computed only for entries walked,
    once each). Ties keep input order, as the plain cosine sort did.
    `cosine_scored` may arrive unsorted.
    """
    ordered = sorted(cosine_scored, key=lambda pair: -pair[1])
    genuine: list[tuple[str, float]] = []
    family: list[tuple[str, float]] = []
    for rank, pair in enumerate(ordered):
        if rank >= size and len(genuine) >= size:
            break
        if is_monologue_family(pool[pair[0]][0]):
            if rank < size:
                family.append(pair)
        elif len(genuine) < size:
            genuine.append(pair)
    return genuine + family


def genuine_first_ranking(
    scored: list[tuple[str, float]], pool: dict[str, tuple[Memory, np.ndarray]]
) -> list[tuple[str, float]]:
    """`(id, score)` pairs in a path's final order: genuine memories by
    descending score, then monologue-family memories by descending score.
    The sort is stable, so equal scores keep their incoming order."""
    return sorted(scored, key=lambda pair: (is_monologue_family(pool[pair[0]][0]), -pair[1]))


@dataclass(frozen=True)
class GatedRanking:
    """Every candidate one turn's gate examined, best first, on ONE scale,
    plus the pass mark that scale is gated by (name-recall fix R2, spec §2).

    `ranked` is `(memory_id, score)` in the path's final order (name-recall
    fix R3, spec §4): every genuine memory by descending score, then every
    monologue-family memory by descending score. Scores are normalized
    reranker scores for the reranked path (the fitted prefix only), raw
    cosine similarities for the cosine path (the whole coarse cut). The
    caller applies `pass_mark` (`select_standouts` for passive recall, a
    plain filter in the tool), both of which keep this order; this object
    never mixes the two scales.
    """

    path: str
    scale: str
    ranked: list[tuple[str, float]]
    pass_mark: float


def _log_calibration_row(
    store: MemoryStore,
    query: str,
    ids: list[str],
    scores: list[float],
    model_id: str,
    pool: dict[str, tuple[Memory, np.ndarray]],
    scale: str,
) -> None:
    """One `calibration_log` row for `query`: `ids` and `scores` aligned 1:1
    (with the recall-time document snapshot), stamped with the scale the
    scores are ACTUALLY on. Fail-soft: a logging failure only loses this
    turn's row, it never demotes a good semantic result to lexical."""
    try:
        store.log_calibration_sample(
            query=query,
            candidate_ids=ids,
            reranker_scores=scores,
            reranker_model_id=model_id,
            candidate_docs=[pool[mid][0].content for mid in ids],
            score_scale=scale,
        )
    except Exception:  # noqa: BLE001 — fail-soft: logging must never break recall
        log.warning("semantic recall: calibration log write failed — continuing", exc_info=True)


def _reranked_ranking(
    store: MemoryStore,
    query: str,
    pool: dict[str, tuple[Memory, np.ndarray]],
    coarse_ids: list[str],
    *,
    log_calibration: bool,
) -> GatedRanking | None:
    """The reranked path, or `None` when the turn must take the cosine path
    instead: the reranker failed to construct or score, fewer than the S5
    minimum of real candidates fit the budget or exist, the anchor
    normalization fell back (`did_normalize=False`: raw scores are NEVER
    gated, P-6), or the rerank floor could not be read (the bootstrap fit
    itself failed: the reranker cannot gate this turn)."""
    try:
        reranker_provider = reranker_mod.build_reranker_provider(store=store)
        # Name-recall fix R1 (spec §1): the width is fitted for THIS message
        # from its own candidates' pair token counts and the process's
        # measured cost model (no hourly sample, diagnosis H8); anchors come
        # on top of the fitted real candidates; the call is measured and
        # feeds the cost model. `coarse_ids` arrives genuine-first (R3, spec
        # §4: genuine candidates are taken first for rerank slots), each
        # group in cosine order, so the fitted prefix takes every genuine
        # candidate before any monologue-family one.
        outcome = reranker_mod.rerank_for_recall(
            reranker_provider, query, [pool[mid][0].content for mid in coarse_ids]
        )
    except Exception:  # noqa: BLE001 — a reranker failure is a cosine-path turn, not keyword-only
        log.warning("semantic recall: reranker failed — taking the cosine path", exc_info=True)
        return None
    if not outcome.reranked or outcome.normalization is None:
        log.info(
            "semantic recall: no rerank (%s, width %d) — taking the cosine path",
            outcome.hand_off,
            outcome.width,
        )
        return None
    reranker_model_id = reranker_provider.model_id()
    normalization = outcome.normalization
    scored_ids = coarse_ids[: normalization.real_width]
    rerank_scores = normalization.scores
    if log_calibration:
        # F2a (#250 inc4) / F2b (#276 §5): the real-query calibration row.
        # `query` is byte-identical to what was just embedded/reranked;
        # `scored_ids`/`rerank_scores` are the SAME already-normalized values
        # that feed the floor gate (the fitted prefix; anchors are never
        # logged). F2c inc1: `candidate_docs` is the recall-time text
        # snapshot, 1:1 with `scored_ids`. Logged BEFORE the floor is read,
        # as before R2: the rerank succeeded and these normalized scores are
        # the training data the rerank floor's own fit needs, even on a turn
        # whose rerank floor turns out to be unavailable (then the cosine
        # path also logs its own, cosine-scale, row for the same query).
        _log_calibration_row(
            store,
            query,
            scored_ids,
            list(rerank_scores),
            reranker_model_id,
            pool,
            CALIBRATION_SCORE_SCALE,
        )
    try:
        # F2a inc8 (#250 §7 UPDATED): the floor is read LIVE per call, keyed
        # by the RUNTIME reranker model_id; no persisted row serves a derived
        # bootstrap. `None` fires only on the bootstrap's own fail-soft path.
        floor_row = store.get_reranker_floor(reranker_model_id)
    except Exception:  # noqa: BLE001
        log.warning(
            "semantic recall: rerank floor read failed — taking the cosine path", exc_info=True
        )
        return None
    if floor_row is None:
        log.info(
            "semantic recall: no rerank floor available (bootstrap failed) for %s — "
            "taking the cosine path",
            reranker_model_id,
        )
        return None
    log.debug(
        "semantic recall: floor=%.4f model=%s cold_start=%s sample_pairs=%d updated_at=%s",
        floor_row["floor"],
        reranker_model_id,
        floor_row["is_cold_start"],
        floor_row["sample_pairs"],
        floor_row["updated_at"],
    )
    ranked = genuine_first_ranking(list(zip(scored_ids, rerank_scores, strict=True)), pool)
    return GatedRanking(
        path=RERANKED_PATH,
        scale=CALIBRATION_SCORE_SCALE,
        ranked=ranked,
        pass_mark=floor_row["floor"],
    )


def _cosine_ranking(
    store: MemoryStore,
    query: str,
    pool: dict[str, tuple[Memory, np.ndarray]],
    coarse: list[tuple[str, float]],
    *,
    embedder_model_id: str,
    log_calibration: bool,
) -> GatedRanking | None:
    """The no-rerank (cosine) path (spec §2, S5/S6/S22/S25/S60): the coarse
    cut's candidates ranked genuine-first then monologue-family, each by
    cosine (R3, spec §4), gated by the cosine floor
    (`store.get_cosine_floor`: persisted, else the bootstrap). `None` when no
    cosine gate can be had (the bootstrap failed): the turn then contributes
    no semantic results, never an ungated ranking.

    Passive recall (`log_calibration`) writes ONE calibration row: the first
    `MAX_STANDOUT_COUNT` (9, the semantic cap) candidates in the path's own
    order, EXAMINED by the gate, pass or fail (S60: the fit needs the
    negatives too), scale 'cosine', the embedder model id as the row's model
    id. The tool never logs (S56)."""
    ranked = genuine_first_ranking([(mid, float(c)) for mid, c in coarse], pool)
    if log_calibration and ranked:
        examined = ranked[:MAX_STANDOUT_COUNT]
        _log_calibration_row(
            store,
            query,
            [mid for mid, _ in examined],
            [score for _, score in examined],
            embedder_model_id,
            pool,
            COSINE_SCORE_SCALE,
        )
    try:
        floor_row = store.get_cosine_floor(embedder_model_id)
    except Exception:  # noqa: BLE001
        log.warning(
            "semantic recall: cosine floor read failed — no semantic results", exc_info=True
        )
        return None
    if floor_row is None:
        log.info(
            "semantic recall: no cosine floor available (bootstrap failed) for %s — "
            "no semantic results",
            embedder_model_id,
        )
        return None
    log.debug(
        "semantic recall: cosine floor=%.4f model=%s cold_start=%s sample_pairs=%d",
        floor_row["floor"],
        embedder_model_id,
        floor_row["is_cold_start"],
        floor_row["sample_pairs"],
    )
    return GatedRanking(
        path=COSINE_PATH,
        scale=COSINE_SCORE_SCALE,
        ranked=ranked,
        pass_mark=floor_row["floor"],
    )


def rank_and_gate(
    store: MemoryStore,
    query: str,
    pool: dict[str, tuple[Memory, np.ndarray]],
    coarse: list[tuple[str, float]],
    *,
    embedder_model_id: str,
    log_calibration: bool,
) -> GatedRanking | None:
    """Score one query's coarse cut and return what to gate (name-recall fix
    R2, spec §2): the reranked path when a rerank of >= 5 real candidates
    ran and normalized, otherwise the cosine path. `None` = no semantic
    result is possible this turn (no cosine gate).

    Shared by passive recall and `search_memories` so the two never diverge
    on path choice, floor or scale. `log_calibration=True` (passive recall
    only, S56) writes the turn's calibration row on whichever path ran."""
    # R3 (spec §4): genuine candidates are taken first for the rerank prefix.
    coarse_ids = genuine_first([mid for mid, _ in coarse], pool)
    gated = _reranked_ranking(store, query, pool, coarse_ids, log_calibration=log_calibration)
    if gated is not None:
        return gated
    return _cosine_ranking(
        store,
        query,
        pool,
        coarse,
        embedder_model_id=embedder_model_id,
        log_calibration=log_calibration,
    )


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
    against the model_id-scoped candidate pool as a CHEAP COARSE CUT (the top
    `relevance.CANDIDATE_POOL` genuine memories by cosine plus the
    monologue-family memories in the plain top-`CANDIDATE_POOL`, so it can
    exceed 50: `genuine_first_coarse_cut`),
    then `rank_and_gate`s the coarse cut:

      - reranked path: a cross-encoder rerank of a per-message-width prefix
        (`reranker.rerank_for_recall`, anchors on top, >= 5 real
        candidates), gated by the CALIBRATED rerank floor
        (`store.get_reranker_floor`) on the anchor-normalized score;
      - cosine path (name-recall fix R2, spec §2): when fewer than 5 real
        candidates fit or exist, the reranker fails to load/score, or its
        normalization falls back, the coarse cut ranked by cosine, gated by
        the cosine floor (`store.get_cosine_floor`). Not keyword-only.

    Passive recall logs the turn's calibration row(s), each stamped with its
    own true scale (normally one; a turn whose rerank scored but whose
    rerank floor was unavailable logs the `normalized` row, then the cosine
    path's `cosine` row), and floor-gates through `select_standouts`.

    Returns a populated `SemanticRecallResult` ONLY when at least one
    candidate clears the operative floor of the path taken. Returns `None`
    for every INCONCLUSIVE case:
      - nothing clears the floor,
      - an empty or sparse candidate pool (cold-start / idle backfill not
        caught up — "graceful warm-up"),
      - an embed failure, or a failed cosine bootstrap on the cosine path
        (no gate is possible; never an ungated ranking),
      - ANY failure ANYWHERE in this function (fail-soft: a broken/missing
        local model, a transient store error such as a locked sqlite db
        during the background backfill, or a floor-read error must never
        break recall — it only demotes this turn to lexical-primary,
        matching the spec's warm-up contract). The whole body is wrapped in
        a broad `except Exception` for exactly this reason.

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
            embedder = embeddings_mod.build_embedding_provider()
            query_vec = embedder.embed(user_input).astype("float32")
        except Exception:  # noqa: BLE001 — fail-soft
            log.exception("run_semantic_recall: query embed failed — falling back to lexical")
            return None

        cosine_scored = [
            (mid, cosine_similarity(query_vec, vec)) for mid, (_, vec) in pool.items()
        ]
        # Spec §4, S77: top 50 genuine + the family memories in the plain top 50.
        coarse = genuine_first_coarse_cut(cosine_scored, pool)

        gated = rank_and_gate(
            store,
            user_input,
            pool,
            coarse,
            embedder_model_id=embedder.model_id(),
            log_calibration=True,
        )
        if gated is None:
            return None

        tiers = select_standouts(gated.ranked, gated.pass_mark)
        if tiers is None:
            return None

        full = [pool[mid][0] for mid in tiers.full_ids]
        snippet = [pool[mid][0] for mid in tiers.snippet_ids]
        scores = dict(gated.ranked)
        hits = [
            SemanticHit(
                memory=mem,
                score=scores[mem.id],
                path=gated.path,
                paragraph=SINGLE_QUERY_PARAGRAPH,
                monologue_family=is_monologue_family(mem),
            )
            for mem in (*full, *snippet)
        ]
        return SemanticRecallResult(
            full=full,
            snippet=snippet,
            scores=scores,
            hits=hits,
            path=gated.path,
            scale=gated.scale,
            pass_mark=gated.pass_mark,
        )
    except Exception:  # noqa: BLE001 — fail-soft: ANY failure demotes to lexical, never raises
        log.warning(
            "run_semantic_recall: semantic path failed — falling back to lexical",
            exc_info=True,
        )
        return None
