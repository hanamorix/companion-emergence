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
calibrates it; the bootstrap is computed once per process at process start,
never on the reply path, and a failed one is retried at the next lull by the
central cadence job: until a cosine floor exists this path renders keyword
results only, S85). A reranker failure therefore no longer demotes the turn to
keyword-only. The two scales never mix: a reranked candidate is gated by the
normalized rerank floor, a cosine-path candidate by the cosine floor, and each
path's calibration row carries its own true scale. (Spec §4, S82: a reranked
paragraph's monologue-family candidates that got no rerank slot form a cosine
TAIL, gated by the cosine floor and ranked after its reranked results; a
paragraph can then yield results on both scales, each gated only by its own
floor.) The module returns `None`
(the caller falls through UNCHANGED to the existing lexical/blend retrieval)
only for an empty/sparse pool (cold-start "graceful warm-up"), an embed
failure, no cosine floor yet (S85: no calibrated row, and the process-start
computation has not finished or failed and awaits the next-lull retry), or when
nothing clears the floor of the path taken. This module never touches that
fallback path.

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
  - the floor-gated standout selection (`select_standouts`,
    `select_gated_standouts` for a reranked paragraph's cosine tail) — replaces
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

import ast
import logging
import re
import threading
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from brain import dev_constants, tunables
from brain.dev_constants import MONOLOGUE_FAMILY_TYPES, RERANK_MIN_REAL_CANDIDATES
from brain.memory import cost_series, floor_startup, recall_diagnostics
from brain.memory import embeddings as embeddings_mod
from brain.memory import reranker as reranker_mod
from brain.memory.embedding_matrix import EmbeddingMatrix, build_embedding_matrix
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
# comment: `MemoryStore.get_reranker_floor` returns a servable floor when no
# persisted row exists yet, serving a derived, process-wide cached BOOTSTRAP
# floor instead of `None` (see `floor_calibration.get_bootstrap_floor`; name-
# recall fix S85 revised, S91/S92: that bootstrap is computed in the background
# on first need and retried on each message / at the next lull, never here).
# This decouples semantic recall's
# EXISTENCE from the daily calibration tick ever having fired for the
# runtime model_id — the earlier design ("no row -> None -> fall back to
# lexical, exactly like an empty/sparse candidate pool") permanently
# coupled recall to the tick (disabled calibration, or a recall running
# before the tick's first idle moment, silently and PERMANENTLY demoted to
# lexical-only even with embeddings present) — see the spec's §7 UPDATED
# note for the full rationale. No floor yet (the first-need bootstrap has not
# finished, or it failed and awaits the next message / lull retry) means the
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
    return _surfacing([(mid, score) for mid, score in reranked_desc if score >= floor])


def _surfacing(standouts: list[tuple[str, float]]) -> SemanticSurfacing | None:
    """The surfacing tiers of an already floor-cleared, already ordered list
    (`select_standouts`, `select_gated_standouts`): `None` when empty,
    otherwise the first `MAX_STANDOUT_COUNT` split 5 full / rest snippet."""
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
    cosine floor; the two scales are never mixed within `scores`.

    Name-recall fix S82 (spec §4): a RERANKED result can also carry a cosine
    tail, the monologue-family candidates that got no rerank slot, gated by
    the cosine floor and placed after every reranked result in `full` /
    `snippet`. `path`, `scale`, `pass_mark` and `scores` describe the
    result's own (primary) ranking only; the tail's raw cosine scores are in
    `tail_scores` (every examined tail candidate), its scale and cosine
    floor in `tail_scale` / `tail_pass_mark` (all `None` / empty without a
    tail). Each hit's own `path` and `score` in `hits` say which scale
    surfaced it; the two scales' scores are never compared.
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
    # the pass mark that was actually applied to them: the result's own
    # (primary) ranking; a reranked result's cosine tail (S82) is described
    # by the `tail_*` fields and by each hit's own `path`.
    path: str = "reranked"
    scale: str = CALIBRATION_SCORE_SCALE
    pass_mark: float | None = None
    # Spec §4, S82: the cosine tail of a reranked result (see the class doc).
    tail_scores: dict[str, float] = field(default_factory=dict)
    tail_scale: str | None = None
    tail_pass_mark: float | None = None
    # Name-recall fix R6 (spec §3, §6; S30, S53): every query of the
    # per-paragraph search (`ParagraphOutcome`, in query order) and the
    # message-level figures (see `ParagraphSearch`). `path`, `scale`,
    # `pass_mark`, `scores` and the `tail_*` fields above describe the first
    # query with a gated ranking (the only one on a one-paragraph message).
    paragraphs: tuple[ParagraphOutcome, ...] = ()
    whole_message_fallback: bool = False
    total_width: int = 0
    budget: float | None = None
    rerank_budget: float | None = None
    # Spec §3/§7 (S33, S55, S64): the guaranteed results' ids (each
    # paragraph's best floor-clearing genuine memory), reranked-path ones by
    # normalized score, then cosine-path ones by cosine. The final merge with
    # the keyword hits (`merge_with_keyword`, S97/S98) keeps them first.
    guaranteed_ids: tuple[str, ...] = ()

    def grouped(self) -> list[tuple[Memory, str]]:
        """`full` + `snippet` in order, each with its result group (spec §4,
        S97): from its hit's path when `hits` has it, else from this result's
        own `path` (a result built by hand)."""
        path_of = {hit.memory.id: hit.path for hit in self.hits}
        return [
            (memory, result_group(path_of.get(memory.id, self.path), is_monologue_family(memory)))
            for memory in [*self.full, *self.snippet]
        ]


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
    still in it (so it can surface on the cosine path, ranked after the
    genuine ones). Rerank slots go to this order's prefix, i.e. genuine first:
    family memories are reranked only if width remains, and
    `rerank_for_recall` only ever sees the first `CANDIDATE_POOL` documents,
    so with `size` or more genuine memories a family memory is never in the
    rerank prefix. On a reranked paragraph such a family memory is not
    reranked but is still a candidate: it forms the paragraph's cosine tail,
    gated by the cosine floor and ranked after the reranked results
    (`_cosine_tail`, spec §4, S82).

    One pass over the candidates: a single sort by cosine, then a walk that
    stops as soon as the plain top-`size` is behind it and `size` genuine
    memories are held (the family flag is computed only for entries walked,
    once each). Ties keep input order, as the plain cosine sort did.
    `cosine_scored` may arrive unsorted.
    """
    ordered = sorted(cosine_scored, key=lambda pair: -pair[1])
    return _walk_coarse_cut(ordered, lambda mid: is_monologue_family(pool[mid][0]), size)


def _walk_coarse_cut(
    ordered: Iterable[tuple[str, float]],
    is_family: Callable[[str], bool],
    size: int,
) -> list[tuple[str, float]]:
    """The walk of `genuine_first_coarse_cut` over `(id, cosine)` pairs
    already in descending cosine order (ties in input order): keep genuine
    memories up to `size`, family memories only while the rank is inside the
    plain top-`size`, stop once both are settled. `is_family` is called at
    most once per entry walked. `ordered` may be a lazy iterator (R6 feeds
    it from one `argsort` per query), so nothing past the stop is produced."""
    genuine: list[tuple[str, float]] = []
    family: list[tuple[str, float]] = []
    for rank, pair in enumerate(ordered):
        if rank >= size and len(genuine) >= size:
            break
        if is_family(pair[0]):
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
    caller applies `pass_mark` (`select_gated_standouts` for passive recall,
    `gated_cleared` in the tool), both of which keep this order. Every
    ranking is on ONE scale and gated only by that scale's `pass_mark`.

    `tail` (name-recall fix, spec §4, S82): on a reranked ranking, the
    monologue-family candidates that got no rerank slot, ranked by cosine and
    gated by the COSINE floor, as their own cosine-scale `GatedRanking`;
    their results follow every result of this ranking (the S44
    "cosine-path monologue-family" group). One paragraph can therefore
    produce results on both scales; the two rankings' scores are never
    compared or merged, each result faces only its own scale's floor. `None`
    on a cosine ranking, when every family candidate was reranked, or when
    the cosine floor could not be read.
    """

    path: str
    scale: str
    ranked: list[tuple[str, float]]
    pass_mark: float
    tail: GatedRanking | None = None
    # Name-recall fix R6 (S58): the real candidates the reranker actually
    # scored for this query (0 when no rerank ran). Set by `rank_and_gate`;
    # also non-zero on a cosine ranking whose query was reranked but could not
    # be gated (no rerank floor yet), since that rerank still used up part of
    # the message's 50 real candidates.
    real_width: int = 0


def gated_cleared(gated: GatedRanking) -> list[tuple[str, float, str]]:
    """`(memory_id, score, path)` for every result that clears its OWN scale's
    floor, in the spec §4 final order for one paragraph: the ranking's own
    results (genuine, then family), then its cosine tail's results. Scores of
    the two segments are never compared."""
    cleared = [(mid, score, gated.path) for mid, score in gated.ranked if score >= gated.pass_mark]
    if gated.tail is not None:
        cleared += [
            (mid, score, gated.tail.path)
            for mid, score in gated.tail.ranked
            if score >= gated.tail.pass_mark
        ]
    return cleared


def select_gated_standouts(gated: GatedRanking) -> SemanticSurfacing | None:
    """`select_standouts` for a `GatedRanking` that may carry a cosine tail:
    the cleared results of `gated_cleared` (each gated only by its own
    scale's floor), tiered together, capped at `MAX_STANDOUT_COUNT` with the
    tail last, so the cap drops tail results before any result of the
    ranking's own scale. `None` when nothing clears."""
    return _surfacing([(mid, score) for mid, score, _ in gated_cleared(gated)])


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
    coarse: list[tuple[str, float]] | None = None,
    embedder_model_id: str | None = None,
    budget_seconds: float | None = None,
    max_real: int = CANDIDATE_POOL,
    sizes: reranker_mod.PairSizes | None = None,
) -> tuple[GatedRanking | None, int]:
    """`(ranking, reranked real candidates)`. The ranking is the reranked
    path's, or `None` when the turn must take the cosine path
    instead: the reranker failed to construct or score, fewer than the S5
    minimum of real candidates fit the budget or exist, the anchor
    normalization fell back (`did_normalize=False`: raw scores are NEVER
    gated, P-6), or there is no rerank floor yet (S91: the need is flagged and
    the bootstrap starts in the background; the reranker cannot gate this
    turn).

    When `coarse` and `embedder_model_id` are given (`rank_and_gate` always
    does), the returned ranking also carries the cosine tail (spec §4, S82):
    the monologue-family candidates that got no rerank slot, gated by the
    cosine floor (`_cosine_tail`).

    `budget_seconds`, `max_real` and `sizes` go to `rerank_for_recall`
    (name-recall fix R6: a paragraph's fair share of the rerank budget, what
    is left of the message's 50 real candidates, and its pair token counts
    computed ahead). The second element is how many real candidates were
    actually reranked (0 when no rerank ran), whether or not the ranking
    could then be gated.

    When fewer than the S5 minimum of real candidates exist or are allowed
    (`max_real`, what is left of the message's 50), no rerank can run and the
    reranker is not even built (hand-off "pool" / "allocation")."""
    if min(len(coarse_ids), max_real) < RERANK_MIN_REAL_CANDIDATES:
        log.info(
            "semantic recall: no rerank (%s, %d candidates, %d allowed) — taking the cosine path",
            "pool" if len(coarse_ids) < RERANK_MIN_REAL_CANDIDATES else "allocation",
            len(coarse_ids),
            max_real,
        )
        return None, 0
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
            reranker_provider,
            query,
            [pool[mid][0].content for mid in coarse_ids],
            store=store,
            budget_seconds=budget_seconds,
            max_real=max_real,
            sizes=sizes,
        )
    except Exception:  # noqa: BLE001 — a reranker failure is a cosine-path turn, not keyword-only
        log.warning("semantic recall: reranker failed — taking the cosine path", exc_info=True)
        return None, 0
    if not outcome.reranked or outcome.normalization is None:
        log.info(
            "semantic recall: no rerank (%s, width %d) — taking the cosine path",
            outcome.hand_off,
            outcome.width,
        )
        # A rerank that ran but could not be normalized still used its width.
        return None, outcome.width if outcome.hand_off == "normalization" else 0
    reranker_model_id = reranker_provider.model_id()
    normalization = outcome.normalization
    reranked_width = normalization.real_width
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
        # by the RUNTIME reranker model_id; no persisted row serves the cached
        # bootstrap (S85 revised: never computed here). `None` = no floor yet.
        floor_row = store.get_reranker_floor(reranker_model_id)
    except Exception:  # noqa: BLE001
        log.warning(
            "semantic recall: rerank floor read failed — taking the cosine path", exc_info=True
        )
        return None, reranked_width
    if floor_row is None:
        # S91: this turn reranked but has no floor to gate the scores with. It
        # takes the cosine path, and the need is flagged: the rerank bootstrap
        # starts in the background (one at a time), never on this reply path.
        floor_startup.request_rerank_bootstrap()
        log.debug(
            "semantic recall: no rerank floor yet for %s (no calibrated row, and the "
            "first-need background bootstrap has not produced one) — taking the cosine path",
            reranker_model_id,
        )
        return None, reranked_width
    log.debug(
        "semantic recall: floor=%.4f model=%s cold_start=%s sample_pairs=%d updated_at=%s",
        floor_row["floor"],
        reranker_model_id,
        floor_row["is_cold_start"],
        floor_row["sample_pairs"],
        floor_row["updated_at"],
    )
    ranked = genuine_first_ranking(list(zip(scored_ids, rerank_scores, strict=True)), pool)
    tail = None
    if coarse is not None and embedder_model_id is not None:
        tail = _cosine_tail(
            store,
            query,
            pool,
            coarse,
            set(scored_ids),
            embedder_model_id=embedder_model_id,
        )
    return (
        GatedRanking(
            path=RERANKED_PATH,
            scale=CALIBRATION_SCORE_SCALE,
            ranked=ranked,
            pass_mark=floor_row["floor"],
            tail=tail,
            real_width=reranked_width,
        ),
        reranked_width,
    )


def _cosine_tail(
    store: MemoryStore,
    query: str,
    pool: dict[str, tuple[Memory, np.ndarray]],
    coarse: list[tuple[str, float]],
    reranked_ids: set[str],
    *,
    embedder_model_id: str,
) -> GatedRanking | None:
    """The cosine tail of a reranked paragraph (spec §4, S82): the
    monologue-family candidates of `coarse` that got no rerank slot, ranked by
    cosine and gated by the COSINE floor, on the cosine scale, ranked through
    `_cosine_ranking` but WRITING NO calibration row (S84: the daily cosine fit
    trains only on no-rerank-path rows, never on a family-only sample;
    the examined tail stays on `GatedRanking.tail` / `SemanticRecallResult.
    tail_scores` for the diagnostics record, R7). Unreranked
    GENUINE candidates are not part of the tail (S53: candidates beyond the
    width are dropped). `None` when there is no such candidate (the cosine
    floor is then not even read) or the cosine floor cannot be had; fail-soft:
    a failure here never demotes the reranked results."""
    try:
        unreranked_family = [
            (mid, cosine)
            for mid, cosine in coarse
            if mid not in reranked_ids and is_monologue_family(pool[mid][0])
        ]
        if not unreranked_family:
            return None
        return _cosine_ranking(
            store,
            query,
            pool,
            unreranked_family,
            embedder_model_id=embedder_model_id,
            log_calibration=False,
        )
    except Exception:  # noqa: BLE001 — fail-soft: the reranked results stand without the tail
        log.warning("semantic recall: cosine tail failed — reranked results only", exc_info=True)
        return None


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
    (`store.get_cosine_floor`: persisted, else the bootstrap computed at
    process start off the reply path, S85). `None` when no cosine gate exists
    yet (no calibrated row, and the startup computation has not finished or
    failed and awaits the next-lull retry): the turn then contributes no semantic
    results (keyword only), never an ungated ranking.

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
        log.debug(
            "semantic recall: no cosine floor yet for %s (no calibrated row, and the cadence "
            "job has not produced the bootstrap) — keyword only",
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
    budget_seconds: float | None = None,
    max_real: int = CANDIDATE_POOL,
    sizes: reranker_mod.PairSizes | None = None,
) -> tuple[GatedRanking | None, int]:
    """Score one query's coarse cut and return `(what to gate, real
    candidates reranked)` (name-recall fix R2, spec §2): the reranked path
    when a rerank of >= 5 real candidates ran and normalized, otherwise the
    cosine path. A `None` ranking = no semantic result is possible for this
    query (no cosine gate). The second element counts the real candidates a
    rerank actually scored (0 when none ran) whether or not a ranking could
    then be gated, so the message's 50-candidate maximum (S58) counts a
    rerank that ran on a query with no gate too.

    Shared by passive recall and `search_memories` so the two never diverge
    on path choice, floor or scale. `log_calibration=True` (passive recall
    only, S56) writes the turn's calibration row on whichever path ran.

    A reranked ranking also carries the cosine tail (spec §4, S82) of its
    unreranked monologue-family candidates (`GatedRanking.tail`); the tail
    writes NO calibration row (S84), so a reranked turn logs only its
    'normalized' row.

    Name-recall fix R6: `budget_seconds` (default: the whole tunable budget),
    `max_real` (default: the design maximum of 50) and `sizes` (default:
    computed here) are one paragraph's fair share, its part of the message's
    50 real candidates and its pair token counts; the returned ranking's
    `real_width` also says how many real candidates were reranked."""
    # R3 (spec §4): genuine candidates are taken first for the rerank prefix.
    coarse_ids = genuine_first([mid for mid, _ in coarse], pool)
    gated, reranked_width = _reranked_ranking(
        store,
        query,
        pool,
        coarse_ids,
        log_calibration=log_calibration,
        coarse=coarse,
        embedder_model_id=embedder_model_id,
        budget_seconds=budget_seconds,
        max_real=max_real,
        sizes=sizes,
    )
    if gated is not None:
        return gated, reranked_width
    cosine = _cosine_ranking(
        store,
        query,
        pool,
        coarse,
        embedder_model_id=embedder_model_id,
        log_calibration=log_calibration,
    )
    if cosine is None or not reranked_width:
        return cosine, reranked_width
    return replace(cosine, real_width=reranked_width), reranked_width


# ---------------------------------------------------------------------------
# Per-paragraph semantic search (name-recall fix R6; spec §3, §4, §7: S10, S11,
# S17, S29, S33, S34, S44, S46, S53, S55, S58, S61, S63, S64, S86; plan P-7,
# P-10, P-23, P-26-P-29)
# ---------------------------------------------------------------------------
#
# A message is split into paragraphs (`str.splitlines()`, blank ones and ones
# with no keyword word dropped); every paragraph is embedded in one batch and
# searched on its own (its own coarse cut of 50 genuine plus the family in its
# plain top 50); the cuts are merged by memory id, a memory going to the
# paragraph with its best cosine score. One per-message time budget (the
# `reranker.latency_budget_seconds` tunable) pays for everything: the
# per-message setup `T_m` (with the token-count probe) and the per-query scan
# time `T_s` (product share + coarse cut + pair-size probe; running averages in
# this process, ratio of sums, no constant) and the paragraph embed cost (a
# length-aware least-squares fit per 24-hour compaction period, S108, S118)
# come off the top, and what is left is split equally among the paragraphs for
# their reranks. Paragraphs are then served in order of their best cosine
# score, each reranking its own candidates against its own text with its own
# anchors, within its share and within what is left of the message's 50 real
# candidates; a paragraph that cannot fit 5 takes the cosine path. The
# results are assembled across paragraphs (`assemble_paragraph_results`).

# Clock seam for the per-message budget (T_m, T_s, the embed calls).
# Production: `time.monotonic`. Tests script it.
_clock: Callable[[], float] = time.monotonic

# The keyword selector's word pattern (`brain.chat.prompt._extract_recall_tokens`,
# `brain.memory.known_names`): a paragraph's words for the emptiness test.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# The three paragraph-time series are rolling 24-hour windows saved in
# memories.db (name-recall fix S146, S155: `cost_series`), so a restart
# continues from them. Samples are only buffered on the reply path (S164).
#
# `T_m`: per-message setup before the first query is embedded (paragraph
# split, the stacked matrix, the lean pool read; plan P-26). One key, x = 1:
# its estimate is the window's mean.
_message_overhead = cost_series.CostSeries("recall setup")
# `T_s`: per query, its share of the matrix product + its coarse cut + its
# reranker pair-size probe (plan P-7, P-28; S108: the per-paragraph scan time
# is a mean over the window). Shared by passive recall and the tool. One key,
# x = 1.
_scan_time = cost_series.CostSeries("paragraph scan")
# The paragraph embed cost (S108, S118): per embedder model id, one sample per
# embed batch, x = texts x the longest text's tokens, y = its in-lock seconds;
# least squares over the window.
_embed_cost = cost_series.CostSeries("paragraph embed")
# Embedder model ids whose first embed of this process has run (S117).
_embed_warmed: set[str] = set()
_embed_warm_lock = threading.Lock()
# Embedder model ids whose load/warm-up has failed once (warn once, then debug).
_embed_warm_failed: set[str] = set()


def _warm_up_embedder_once(embedder: embeddings_mod.EmbeddingProvider, model_id: str, text: str) -> None:
    """S117: the process's first embed for `model_id`, discarded (untimed,
    in no sample), as the reranker's warm-up discards its first inferences.
    Marked done only after it succeeds; two threads racing here may both warm
    up, which costs time but never skews a sample."""
    with _embed_warm_lock:
        if model_id in _embed_warmed:
            return
    embedder.embed(text)
    with _embed_warm_lock:
        _embed_warmed.add(model_id)


def _timed_probe(
    embedder: embeddings_mod.EmbeddingProvider, texts: list[str]) -> tuple[list[int] | None, float]:
    """`embedder.token_lengths_timed(texts)` and its seconds: the in-lock
    figure when the provider reports one, else the call's own time. A probe
    that raises is `None` lengths (fail-soft)."""
    began = _clock()
    try:
        lengths, in_lock = embedder.token_lengths_timed(texts)
    except Exception:  # noqa: BLE001 — fail-soft: sizing only
        log.warning("semantic recall: token-count probe failed", exc_info=True)
        return None, _clock() - began
    return lengths, (in_lock if in_lock is not None else _clock() - began)


def _reset_paragraph_time_model() -> None:
    """Test-only: forget the embed warm-up state. The three series live in the
    database and the process buffer (`cost_series._reset_pending`)."""
    with _embed_warm_lock:
        _embed_warmed.clear()
    _embed_warm_failed.clear()


def split_paragraphs(text: str, keyword_words: Collection[str] | None) -> list[str]:
    """The message's paragraphs for the per-paragraph search (spec §3, S10,
    S29, S34, S53): `str.splitlines()` (every line ending: `\\n`, `\\r\\n`,
    `\\r`, ...), blank lines dropped, each paragraph's text as written.

    With more than one paragraph, a paragraph none of whose words (the
    keyword selector's `[A-Za-z0-9]+` words, lower-cased) is in
    `keyword_words` is dropped ("ok", "lol"). The caller builds
    `keyword_words` from its own keyword tokens plus the words of the known
    names matched in the message, so the test runs after name protection
    and a paragraph holding only a known name is kept. `None` skips the
    test. A message with one non-blank paragraph keeps it (one paragraph
    behaves as today). Returns `[]` for a blank message, and `[]` when every
    paragraph was dropped (the caller then searches the whole message)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= 1 or keyword_words is None:
        return lines
    return [
        line
        for line in lines
        if any(word.lower() in keyword_words for word in _WORD_RE.findall(line))
    ]


def _serve_order(best_cosines: list[float | None]) -> list[int]:
    """Paragraph indexes in the order they are served their rerank widths
    (S58): by best cosine score, descending, ties by paragraph index;
    paragraphs with no candidate last."""
    return sorted(
        range(len(best_cosines)),
        key=lambda i: (best_cosines[i] is None, -(best_cosines[i] or 0.0), i),
    )


def _allocate_next(fitted_width: int, used: int) -> int:
    """One served paragraph's real candidates (S58): its full fitted width,
    capped by what is left of the message's 50 real candidates, or 0 (the
    cosine path) when that is below the S5 minimum."""
    width = min(fitted_width, CANDIDATE_POOL - used)
    return width if width >= RERANK_MIN_REAL_CANDIDATES else 0


def allocate_widths(fitted: list[tuple[int, float | None]]) -> list[int]:
    """Real rerank candidates per paragraph when the shares may sum above 50
    (spec §7, S58; plan P-29, criterion C9d). `fitted` is, per paragraph, the
    width fitted on its equal share and its best cosine score. Paragraphs are
    served by best cosine score (ties by index); each gets its full fitted
    width while at least 5 of the 50 remain for it, else 0 (the cosine path).
    When the widths sum to 50 or less every one is kept (a width below 5 is 0
    either way). Pure.

    Passive recall and the tool apply the same two steps (`_serve_order`,
    `_allocate_next`) one paragraph at a time, fitting each paragraph's width
    just before its rerank so it uses the cost model as the previous
    paragraph's measured rerank left it; with an unchanged cost model the
    result is this function's."""
    widths = [0] * len(fitted)
    used = 0
    for i in _serve_order([best for _, best in fitted]):
        width = _allocate_next(fitted[i][0], used)
        widths[i] = width
        used += width
    return widths


@dataclass(frozen=True)
class ParagraphOutcome:
    """One query of a message's semantic search (R6): its index (the
    `SemanticHit.paragraph` tag), its text (the rerank and calibration
    query), its gated ranking (`None` when no gate was possible: no
    candidate, or no cosine floor yet), its candidate count after the merge,
    and its best cosine score. `reranked_width` is the real candidates the
    reranker scored for it (0 when none)."""

    index: int
    query: str
    gated: GatedRanking | None
    candidate_count: int
    best_cosine: float | None
    # Real candidates a rerank scored for this query (0 when none ran), also
    # when no ranking could be gated (S58: it still counts toward the 50).
    reranked_width: int = 0


@dataclass(frozen=True)
class ParagraphSearch:
    """Everything one message's per-paragraph search produced (R6): each
    query's outcome, the merged candidates' rows, and the message-level
    figures the diagnostics record carries (S30, S53): whether the whole
    message was searched as one query (`whole_message_fallback`), the real
    candidates reranked in total (`total_width`, never above 50, S58), the
    per-message `budget` and the part of it left for reranking
    (`rerank_budget`). `paragraph_count` is the number of queries scored
    (1 on the whole-message fallback; 2 on a first message re-scored as its
    first paragraph and the rest, S63)."""

    outcomes: tuple[ParagraphOutcome, ...]
    pool: dict[str, tuple[Memory, np.ndarray]]
    whole_message_fallback: bool
    total_width: int
    budget: float
    rerank_budget: float

    @property
    def paragraph_count(self) -> int:
        return len(self.outcomes)


def paragraph_diagnostics(
    outcomes: Sequence[ParagraphOutcome],
) -> list[recall_diagnostics.ParagraphDiagnostic]:
    """The diagnostics rows of one message's outcomes, in outcome order (spec
    section 6, S15, S48, S53, S58; plan P-22, P-34).

    A row's `width` is the outcome's `reranked_width` (the real candidates the
    reranker scored for that query, 0 when none ran), never `GatedRanking.
    real_width`: the outcome carries the width also when no ranking could be
    gated, so S58's count toward the 50 is not lost. A query with a gated
    ranking writes one row for it (its own path, floor and scale) and, when a
    reranked ranking carries a cosine tail (S82), a second row for the tail
    (cosine path, the COSINE floor and scale, width 0: no real candidate is
    counted twice, so the rows' widths sum to the message's `total_width`). A
    query with no gated ranking writes one row with no pass mark and no scale,
    on the reranked path iff a rerank scored."""
    rows: list[recall_diagnostics.ParagraphDiagnostic] = []
    for outcome in outcomes:
        gated = outcome.gated
        if gated is None:
            rows.append(
                recall_diagnostics.ParagraphDiagnostic(
                    path=RERANKED_PATH if outcome.reranked_width > 0 else COSINE_PATH,
                    width=outcome.reranked_width,
                    pass_mark=None,
                    scale=None,
                )
            )
            continue
        rows.append(
            recall_diagnostics.ParagraphDiagnostic(
                path=gated.path,
                width=outcome.reranked_width,
                pass_mark=gated.pass_mark,
                scale=gated.scale,
            )
        )
        tail = gated.tail
        if tail is not None:
            rows.append(
                recall_diagnostics.ParagraphDiagnostic(
                    path=tail.path, width=0, pass_mark=tail.pass_mark, scale=tail.scale
                )
            )
    return rows


def emit_recall_diagnostics(
    persona_dir: Path | None,
    search: ParagraphSearch | None,
    *,
    source: str,
    mode: str | None = None,
) -> None:
    """Append this call's ONE diagnostics record (spec section 6, S15, S30,
    S59; plan P-22). Called from a `finally` at each call site, so it must never
    raise: the path, the rows and the append are all inside one guard (a
    `persona_dir` that is not a path, or any other failure, is logged and the
    record dropped). `search` is `None` when no per-paragraph search ran or it
    failed (no pool, embed failure, an exception): the record then has zero
    paragraphs and no budget. Reads no store and no table; the only I/O is the
    locked append."""
    try:
        path = recall_diagnostics.diagnostics_path(persona_dir)
        if search is None:
            recall_diagnostics.log_recall(
                path,
                source=source,
                mode=mode,
                paragraph_count=0,
                whole_message_fallback=False,
                total_width=0,
                budget=None,
                paragraphs=[],
            )
            return
        recall_diagnostics.log_recall(
            path,
            source=source,
            mode=mode,
            paragraph_count=search.paragraph_count,
            whole_message_fallback=search.whole_message_fallback,
            total_width=search.total_width,
            budget=search.budget,
            paragraphs=paragraph_diagnostics(search.outcomes),
        )
    except Exception as exc:  # noqa: BLE001 — diagnostics must never reach the recall caller
        # No traceback: a persistent fault here repeats on every recall (D1's rule).
        log.warning("recall diagnostics: could not emit a record (%r); dropped", exc)


def _unit_rows(vectors: list[np.ndarray]) -> np.ndarray:
    """Query vectors stacked as float32 rows scaled to unit length (a zero
    vector stays zero, so its cosines are 0.0, as `cosine_similarity`)."""
    queries = np.stack([np.asarray(v, dtype=np.float32) for v in vectors]).astype(np.float32)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    np.divide(queries, norms, out=queries, where=norms > 0)
    return queries


@dataclass(frozen=True)
class _Cut:
    """A message cut to a prefix that fits (S195, S199): its text, the token count
    the embedder's own probe gave it, and the seconds the probes took."""

    text: str
    tokens: int
    probe_seconds: float


_TRAILING_PARTIAL_WORD = re.compile(r"\s\S*\Z")


def _word_prefix(text: str, chars: int) -> str:
    """The first `chars` characters of `text`, ending at a word boundary when the
    cut falls inside a word and the prefix holds a boundary; a plain character
    prefix otherwise (one long word, text without whitespace)."""
    head = text[:chars]
    if chars < len(text) and not text[chars].isspace():
        boundary = _TRAILING_PARTIAL_WORD.search(head)
        if boundary is not None and boundary.start() > 0:
            head = head[: boundary.start()]
    return head.rstrip()


def _cut_to_tokens(
    embedder: embeddings_mod.EmbeddingProvider, text: str, total_tokens: int, cap_tokens: int
) -> _Cut | None:
    """`text` as it is when it fits `cap_tokens`, else its LONGEST prefix that does
    (S195, S199), found by bisection on the character count with the embedder's own
    token probe (the count of a prefix never shrinks as the prefix grows, and the probe
    saturates at the model's input limit, so a proportional step would crawl: stage 8
    saw 500 probes on a long paste). The prefix ends at a word boundary where it can
    (`_word_prefix`). About log2(len(text)) probes. `None` when nothing fits or the
    probe cannot size a prefix. Every probe's seconds are summed (scan-side time)."""
    if total_tokens <= cap_tokens:
        return _Cut(text, total_tokens, 0.0)
    if cap_tokens < 1 or total_tokens < 1:
        return None
    seconds = 0.0
    best: tuple[str, int] | None = None
    fits_up_to, too_long_from = 0, len(text)  # the whole text does not fit (total > cap)
    while too_long_from - fits_up_to > 1:
        middle = (fits_up_to + too_long_from) // 2
        prefix = _word_prefix(text, middle)
        if not prefix:
            fits_up_to = middle
            continue
        lengths, took = _timed_probe(embedder, [prefix])
        seconds += took
        if lengths is None or len(lengths) != 1:
            return None
        if lengths[0] <= cap_tokens:
            fits_up_to = middle
            best = (prefix, lengths[0])
        else:
            too_long_from = middle
    return None if best is None else _Cut(best[0], best[1], seconds)


def search_paragraphs(
    store: MemoryStore,
    text: str,
    *,
    keyword_words: Collection[str] | None = None,
    exclude: Collection[str] = frozenset(),
    log_calibration: bool,
) -> ParagraphSearch | None:
    """One message's per-paragraph semantic search, up to the gated
    ranking of each query (spec §3; the assembly across paragraphs is
    `assemble_paragraph_results`). Shared by passive recall
    (`log_calibration=True`) and `search_memories` (`False`, S56).

    Steps: (1) split (`split_paragraphs`); (2) the lean pool: the stacked
    unit matrix (`EmbeddingMatrix.stacked`, P-23) restricted to the rows that
    are active and in state 'active' (`MemoryStore.active_state_memory_types`,
    P-27), minus `exclude`; its time plus the token-count probe is the
    message's `T_m` sample (P-26; the embedding provider's construction, its
    model load (S109) and its first embed (S117) are one-time and are not);
    (3) the queries: one paragraph; with token counts unknown the whole
    message as one query; with the models warm every paragraph when
    `T_m + E(P, longest) + P x T_s` fits the budget (E the length-aware
    embed fit, S108), else paragraphs 1, 3, 5, ... when those fit, else the
    whole message as one query cut to what fits the time left (S29, S34, S46,
    S195; nothing fits: the first paragraph alone); before any embed
    measurement in this process the first paragraph is embedded, searched and
    timed alone first, then in the same order (all, alternates, else the rest
    of the message as ONE query cut to what fits, sized from that paragraph's
    embed seconds per token) and the message is scored as the first paragraph
    and that rest (S63, S115, S199); (4) one `embed_batch` per group (its
    in-lock seconds one embed-series sample), one matrix product per group,
    each query's coarse cut from its own scores (S77); (5) merge: a memory
    goes to the query with its best cosine (ties: the earlier query),
    keeping that score (S11, S53); (6) the reranker's pair sizes per query,
    their in-lock time added to its scan sample `T_s` (P-28; the one-time
    warm-up is not); (7) what is left of the budget after `T_m`, the embed
    fit's prediction per embed call made, and `(queries) x T_s` (this
    message's samples included) is split equally among the queries (S33,
    S46, S61: a one-query message pays its embed + scan too); (8) the
    queries are served in order
    of best cosine, each `rank_and_gate`d on its own candidates, text and
    anchors within its share and what is left of the 50 real candidates
    (S58): reranked, else the cosine path; passive recall logs each query's
    calibration row with that query as its text (S30).

    `None` when there is no candidate pool, no query text, or the embed
    fails. Other failures raise (callers fail soft)."""
    start = _clock()
    budget = float(
        tunables.get_tunable(
            "reranker.latency_budget_seconds", reranker_mod.LATENCY_BUDGET_SECONDS
        )
    )
    paragraphs = split_paragraphs(text, keyword_words)
    whole_message = False
    if not paragraphs:
        if not text.strip():
            return None
        paragraphs = [text]
        whole_message = True

    matrix = build_embedding_matrix(store.db_path)
    # The matrix's first lazy build (one DB scan per process) is a one-time
    # cost like the embedding provider's construction, so it is kept out of
    # `T_m`; once built this is a no-op.
    before_build = _clock()
    matrix.ensure_built()
    one_time = _clock() - before_build
    ids, unit = matrix.stacked()
    if not ids:
        return None
    types = store.active_state_memory_types()
    excluded = frozenset(exclude)
    rows = [i for i, mid in enumerate(ids) if mid in types and mid not in excluded]
    if not rows:
        return None
    row_index = np.asarray(rows, dtype=np.intp)
    pool_ids = [ids[i] for i in rows]

    before_provider = _clock()
    try:
        # Looked up via the MODULE so a test's monkeypatch on
        # `embeddings.build_embedding_provider` is honored.
        embedder = embeddings_mod.build_embedding_provider()
        embedder_model_id = embedder.model_id()
    except Exception:  # noqa: BLE001 — fail-soft
        log.exception("semantic recall: embedding provider unavailable — falling back to lexical")
        return None
    # One-time costs kept out of every sample (like the provider's
    # construction and the matrix's first build): the model load (S109) and
    # the process's first embed for this model id, discarded (S117). Both are
    # fail-soft: a failure here only means the one-time cost may land in a
    # sample (the warm-up is retried on the next message), and the embed below
    # decides whether semantic recall runs.
    try:
        embedder.ensure_loaded()
        _warm_up_embedder_once(embedder, embedder_model_id, paragraphs[0])
    except Exception:  # noqa: BLE001 — fail-soft
        # Warn with the traceback once per model id; later failures (the same
        # broken internals, every message) log at debug level.
        first = embedder_model_id not in _embed_warm_failed
        _embed_warm_failed.add(embedder_model_id)
        (log.warning if first else log.debug)(
            "semantic recall: embedder load/warm-up failed", exc_info=first
        )

    # Token counts (S108) for the paragraphs and, when there are several, the
    # whole message (the fallback query); its time is part of this message's
    # setup `T_m`, measured inside the provider's lock (round-3 N3-1).
    probe_texts = list(paragraphs) if len(paragraphs) == 1 else [*paragraphs, text]
    probed, probe_seconds = _timed_probe(embedder, probe_texts)
    tokens_of: dict[str, int] | None = (
        None if probed is None or len(probed) != len(probe_texts) else dict(zip(probe_texts, probed, strict=True))
    )
    _message_overhead.record(store, "", 1, before_provider - start - one_time + probe_seconds)
    overhead = _message_overhead.mean(store, "") or 0.0

    def is_family(mid: str) -> bool:
        return types[mid] in MONOLOGUE_FAMILY_TYPES

    embed_seconds: list[float] = []
    embed_calls: list[list[str]] = []

    def embed_and_cut(texts: list[str]) -> list[tuple[list[tuple[str, float]], float]]:
        """Embed `texts` in one batch and cut each; per text its coarse cut
        and its scan part (its share of the matrix product + its cut). The
        batch's embed seconds (in-lock) go to `embed_seconds` and, with the
        texts' token counts known, to the embed cost series as one sample,
        x = texts x the longest text's tokens (S108). No probe time is in it."""
        began = _clock()
        vectors, in_lock = embedder.embed_batch_timed(texts)
        seconds = in_lock if in_lock is not None else _clock() - began
        embed_seconds.append(seconds)
        embed_calls.append(list(texts))
        sizes = None if tokens_of is None else [tokens_of.get(t) for t in texts]
        if sizes is not None and None not in sizes and max(sizes) > 0:
            _embed_cost.record(store, embedder_model_id, len(texts) * max(sizes), seconds)
        began = _clock()
        scores = (unit @ _unit_rows(vectors).T)[row_index]
        product_share = (_clock() - began) / len(texts)
        out = []
        for column in range(len(texts)):
            began = _clock()
            query_scores = scores[:, column]
            order = np.argsort(-query_scores, kind="stable")
            cut = _walk_coarse_cut(
                ((pool_ids[j], float(query_scores[j])) for j in order), is_family, CANDIDATE_POOL
            )
            out.append((cut, product_share + (_clock() - began)))
        return out

    def predicted_embed(queries: list[str]) -> float | None:
        """The embed cost series' prediction for embedding `queries` as one
        batch (S108): `o + r x (queries x longest query's tokens)`."""
        fit = _embed_cost.fit(store, embedder_model_id)
        sizes = None if tokens_of is None else [tokens_of.get(q) for q in queries]
        if fit is None or sizes is None or None in sizes:
            return None
        return fit.predict(len(queries) * max(sizes))

    def cut_query(text_to_cut: str, cap: int) -> tuple[str, float] | None:
        """S195/S199: `text_to_cut` cut to a prefix of at most `cap` tokens
        (`_cut_to_tokens`); its token count is recorded for the embed prediction.
        Returns `(query, probe seconds)`, or `None` when nothing fits or the probe
        cannot size it."""
        total = tokens_of.get(text_to_cut) if tokens_of is not None else None
        if total is None or cap < 1:
            return None
        cut = _cut_to_tokens(embedder, text_to_cut, total, cap)
        if cut is None:
            return None
        tokens_of[cut.text] = cut.tokens
        return cut.text, cut.probe_seconds

    try:
        count = len(paragraphs)
        scan_time = _scan_time.mean(store, "")
        whole = predicted_embed(paragraphs) if count > 1 else None
        # S195: the alternate paragraphs 1, 3, 5, ... (of the paragraphs that
        # survived the empty-keyword drop).
        alternates = paragraphs[::2]

        def embed_token_cap(whole_text: str) -> int:
            """S195/S202: the largest token count whose predicted single-text embed fits the
            budget after the setup and one query's scan (the S108 embed cost model); 0 when
            nothing fits or there is no embed fit."""
            fit = _embed_cost.fit(store, embedder_model_id)
            room = budget - overhead - (scan_time or 0.0)
            if fit is None or room < fit.overhead:
                return 0
            if fit.rate <= 0.0:
                return tokens_of[whole_text]
            return int((room - fit.overhead) / fit.rate)

        if count == 1:
            queries = list(paragraphs)
            # S202: a single paragraph too long to embed in the time left is cut like the
            # whole-message fallback (S195). Without an embed fit or a scan time (the first
            # message) or when nothing fits, it is embedded as it is (S61, as built).
            one_cut: tuple[str, float] | None = None
            if tokens_of is not None and scan_time is not None:
                predicted = predicted_embed(paragraphs)
                if predicted is not None and overhead + predicted + scan_time > budget:
                    one_cut = cut_query(paragraphs[0], embed_token_cap(paragraphs[0]))
            if one_cut is None:
                searched = embed_and_cut(queries)
            else:
                queries = [one_cut[0]]
                searched = [(c, part + one_cut[1]) for c, part in embed_and_cut(queries)]
        elif tokens_of is None:
            # Token counts unknown (round-3 N3-3: checked before the
            # first-message rule): no prediction can be made, one query.
            queries, whole_message = [text], True
            searched = embed_and_cut(queries)
        elif whole is not None and scan_time is not None:
            if overhead + whole + count * scan_time <= budget:
                queries = list(paragraphs)
                searched = embed_and_cut(queries)
            else:
                alternates_embed = predicted_embed(alternates)
                if (
                    alternates_embed is not None
                    and overhead + alternates_embed + len(alternates) * scan_time <= budget
                ):
                    queries = alternates
                    searched = embed_and_cut(queries)
                else:
                    # S195: the whole message as ONE query, cut to what fits the
                    # time left (the S108 embed cost model): the largest token
                    # count whose predicted single-text embed fits the budget
                    # after the setup and one query's scan. Nothing fits: the
                    # first paragraph alone (what a one-paragraph message does,
                    # S61).
                    cut = cut_query(text, embed_token_cap(text))
                    if cut is None:
                        queries = [paragraphs[0]]
                        searched = embed_and_cut(queries)
                    else:
                        queries, whole_message = [cut[0]], True
                        searched = [(c, part + cut[1]) for c, part in embed_and_cut(queries)]
        else:
            # S46/S63/S115/S199: no measurement yet in this process. Time the
            # first paragraph alone (embed + scan), then decide on count x that
            # time, in S195's order: every paragraph, else paragraphs 1, 3, 5,
            # ..., else the first paragraph plus the rest of the message as ONE
            # query cut to what fits (sized from the first paragraph's measured
            # embed seconds per token).
            first = embed_and_cut(paragraphs[:1])
            first_time = embed_seconds[0] + first[0][1]
            if overhead + count * first_time <= budget:
                queries = list(paragraphs)
                searched = first + embed_and_cut(paragraphs[1:])
            elif overhead + len(alternates) * first_time <= budget:
                queries = alternates
                searched = first + (embed_and_cut(alternates[1:]) if len(alternates) > 1 else [])
            else:
                rest = "\n".join(paragraphs[1:])
                rest_tokens, rest_probe = _timed_probe(embedder, [rest])
                if rest_tokens is not None and len(rest_tokens) == 1:
                    tokens_of[rest] = rest_tokens[0]
                first_tokens = tokens_of.get(paragraphs[0], 0)
                rest_query: tuple[str, float] | None = (rest, 0.0)
                if tokens_of.get(rest) is not None:
                    room = budget - overhead - first_time - first[0][1]
                    if room <= 0:
                        cap = 0
                    elif first_tokens > 0 and embed_seconds[0] > 0.0:
                        cap = int(room / (embed_seconds[0] / first_tokens))
                    else:
                        cap = tokens_of[rest]
                    rest_query = cut_query(rest, cap)
                if rest_query is None:
                    queries = [paragraphs[0]]
                    searched = first
                else:
                    # The rest query's token count was probed above; the probes' time
                    # is that query's scan-side work (first message only).
                    queries = [paragraphs[0], rest_query[0]]
                    searched = first + [
                        (cut, part + rest_probe + rest_query[1])
                        for cut, part in embed_and_cut(queries[1:])
                    ]
    except Exception:  # noqa: BLE001 — fail-soft
        log.exception("semantic recall: query embed failed — falling back to lexical")
        return None

    # Merge (S11, S53): each memory goes to the query with its best cosine.
    best: dict[str, tuple[float, int]] = {}
    for qi, (cut, _) in enumerate(searched):
        for mid, cosine in cut:
            if mid not in best or cosine > best[mid][0]:
                best[mid] = (cosine, qi)
    assigned = [[(mid, c) for mid, c in cut if best[mid][1] == qi] for qi, (cut, _) in enumerate(searched)]
    memories = store.get_active_by_ids(list(best))
    vector_of = dict(zip(pool_ids, row_index, strict=True))
    pool = {mid: (memories[mid], unit[vector_of[mid]]) for mid in best if mid in memories}
    assigned = [[(mid, c) for mid, c in cut if mid in pool] for cut in assigned]

    # Pair sizes per rerankable query, their in-lock time added to its scan
    # sample (P-28; round-3 N3-1: a wait behind another thread's rerank, e.g.
    # the re-measure job's, is not timed as work).
    samples = [part for _, part in searched]
    sizes: dict[int, reranker_mod.PairSizes] = {}
    rerankable = [qi for qi, cut in enumerate(assigned) if len(cut) >= RERANK_MIN_REAL_CANDIDATES]
    # A reranker that cannot be built or warmed up fails once per message: the
    # remaining queries take the cosine path without retrying the load.
    reranker_down = False
    ram_model_id = reranker_mod.resolve_reranker_model_id()
    if rerankable and not reranker_mod.ram_figure_ready(ram_model_id, store):
        # S200: no usable memory model on this machine, so no re-scoring runs, and nothing
        # of the reranker is built, loaded, warmed up or probed on this reply path. The
        # need is flagged; the next lull runs the ladder (`rerank_ram_figure`). Every
        # query takes the plain meaning-match path (`max_real=0` below).
        reranker_mod.note_ram_figure_needed(ram_model_id)
        floor_startup.request_ram_figure()  # S201/S210: the direct chat measures it after this reply
        log.info("semantic recall: no usable RAM model yet — cosine path for this message (S200)")
        reranker_down = True
    elif rerankable:
        if reranker_mod.ram_figure_refresh_wanted(ram_model_id):
            # H4b: the gate served this process's kept fit (the window no longer gives one):
            # the direct chat refreshes it in the background after this reply, like the bridge's lull job.
            floor_startup.request_ram_figure()
        try:
            reranker_provider = reranker_mod.build_reranker_provider(store=store)
            # The one-time warm-up (plan P-1) runs before the probes are
            # timed, so the ONNX session load never lands in `T_s`.
            first_docs = _prefix_documents(pool, assigned[rerankable[0]])
            reranker_mod.warm_up_for_recall(reranker_provider, queries[rerankable[0]], first_docs)
        except Exception:  # noqa: BLE001 — fail-soft: the cosine path serves every query
            log.warning(
                "semantic recall: reranker unavailable — cosine path for this message",
                exc_info=True,
            )
            reranker_down = True
        for qi in [] if reranker_down else rerankable:
            began = _clock()
            try:
                sizes[qi] = reranker_mod.recall_pair_sizes(
                    reranker_provider, queries[qi], _prefix_documents(pool, assigned[qi])
                )
            except Exception:  # noqa: BLE001 — this query's rank_and_gate retries and fails soft
                log.warning("semantic recall: pair-size probe failed", exc_info=True)
                continue
            in_lock = sizes[qi].seconds
            samples[qi] += in_lock if in_lock is not None else _clock() - began
    _scan_time.record_many(store, "", [(1, sample) for sample in samples])

    # The rerank budget (S46, S61, S108; plan P-28): the budget minus `T_m`,
    # the embed cost series' prediction for the queries embedded, and every
    # query's scan time (running average), this message's samples included,
    # never below zero; shared equally (S33). With token counts unknown the
    # prediction is replaced by this message's measured embed seconds.
    # One prediction per embed call actually made (each call pays its own
    # overhead: the S63 first message makes two).
    predictions = [predicted_embed(call) for call in embed_calls]
    embed_term = None if None in predictions else sum(predictions)
    if embed_term is None:
        embed_term = sum(embed_seconds)
    spent = overhead + embed_term + len(queries) * (_scan_time.mean(store, "") or 0.0)
    rerank_budget = max(0.0, budget - spent)
    share = rerank_budget / len(queries)

    best_cosines = [max((c for _, c in cut), default=None) for cut in assigned]
    gated_by_query: dict[int, GatedRanking | None] = {}
    width_by_query: dict[int, int] = {}
    used = 0
    for qi in _serve_order(best_cosines):
        if not assigned[qi]:
            gated_by_query[qi] = None
            continue
        gated, reranked = rank_and_gate(
            store,
            queries[qi],
            pool,
            assigned[qi],
            embedder_model_id=embedder_model_id,
            log_calibration=log_calibration,
            budget_seconds=share,
            max_real=0 if reranker_down else CANDIDATE_POOL - used,
            sizes=sizes.get(qi),
        )
        gated_by_query[qi] = gated
        width_by_query[qi] = reranked
        used += reranked
    outcomes = tuple(
        ParagraphOutcome(
            index=qi,
            query=queries[qi],
            gated=gated_by_query.get(qi),
            candidate_count=len(assigned[qi]),
            best_cosine=best_cosines[qi],
            reranked_width=width_by_query.get(qi, 0),
        )
        for qi in range(len(queries))
    )
    return ParagraphSearch(
        outcomes=outcomes,
        pool=pool,
        whole_message_fallback=whole_message,
        total_width=used,
        budget=budget,
        rerank_budget=rerank_budget,
    )


def _prefix_documents(
    pool: dict[str, tuple[Memory, np.ndarray]], coarse: list[tuple[str, float]]
) -> list[str]:
    """A query's candidate documents in the order `rank_and_gate` hands them
    to the reranker (genuine first, each group by cosine)."""
    return [pool[mid][0].content for mid in genuine_first([mid for mid, _ in coarse], pool)]


@dataclass(frozen=True)
class AssembledHit:
    """One result of `assemble_paragraph_results`: the memory id, its score
    on its own path's scale, that path, the paragraph (query index) it came
    from and whether it is a monologue-family memory."""

    memory_id: str
    score: float
    path: str
    paragraph: int
    monologue_family: bool

    @property
    def group(self) -> str:
        return result_group(self.path, self.monologue_family)


# The six result groups of a recall (spec §4, S97). Their ORDER is
# `dev_constants.RECALL_RESULT_GROUP_ORDER` (S98), read at call time; the
# names here are the identifiers that list uses.
RERANKED_GENUINE = "reranked_genuine"
RERANKED_FAMILY = "reranked_family"
COSINE_GENUINE = "cosine_genuine"
KEYWORD_GENUINE = "keyword_genuine"
COSINE_FAMILY = "cosine_family"
KEYWORD_FAMILY = "keyword_family"
RESULT_GROUPS = frozenset(
    {RERANKED_GENUINE, RERANKED_FAMILY, COSINE_GENUINE, KEYWORD_GENUINE, COSINE_FAMILY, KEYWORD_FAMILY}
)


def result_group(path: str, monologue_family: bool) -> str:
    """The group of a semantic result: by the scale that gated it and whether
    it is a monologue-family memory. A monologue-family memory on the cosine
    scale (the cosine path, or a reranked paragraph's unreranked tail) is
    `COSINE_FAMILY`, the non-reranked family group of S97."""
    if path == RERANKED_PATH:
        return RERANKED_FAMILY if monologue_family else RERANKED_GENUINE
    return COSINE_FAMILY if monologue_family else COSINE_GENUINE


def _group_order_problem(order: Sequence[object]) -> tuple[int | None, str] | None:
    """The first thing wrong with a result-group list, as ``(index, why)`` (``index``
    is the 0-based position of the bad entry, None for a group the list lacks), or
    None when the list names each of the six groups exactly once."""
    seen: set[str] = set()
    for index, entry in enumerate(order):
        if not isinstance(entry, str):
            return index, f"entry {index + 1} of {len(order)} is {entry!r}, not a group name"
        if entry not in RESULT_GROUPS:
            return index, f"entry {index + 1} of {len(order)}, {entry!r}, is not a result group"
        if entry in seen:
            return index, f"entry {index + 1} of {len(order)}, {entry!r}, is listed twice"
        seen.add(entry)
    missing = sorted(RESULT_GROUPS - seen)
    if missing:
        return None, f"the list does not name {', '.join(repr(m) for m in missing)}"
    return None


def _dev_constants_location(
    index: int | None, path: str | Path | None = None, length: int | None = None
) -> tuple[str, int | None]:
    """``(file, line)`` of the ``RECALL_RESULT_GROUP_ORDER`` entry at ``index`` in the
    source file of ``dev_constants`` (or of ``path``, a test seam): the line of that
    element when the source is a literal tuple/list with no starred element and (when
    ``length`` is given) as many elements as the runtime list; else the line of the
    assignment (also for a missing group, which has no element). The line is None when
    the file cannot be read or parsed. Run only when a list is already known to be bad."""
    where = str(path if path is not None else getattr(dev_constants, "__file__", None) or "brain/dev_constants.py")
    try:
        tree = ast.parse(Path(where).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return where, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        else:
            continue
        if not any(t.id == "RECALL_RESULT_GROUP_ORDER" for t in targets) or node.value is None:
            continue
        value = node.value
        if (
            index is not None
            and isinstance(value, (ast.Tuple, ast.List))
            and not any(isinstance(e, ast.Starred) for e in value.elts)
            and (length if length is not None else len(value.elts)) == len(value.elts)
            and index < len(value.elts)
        ):
            return where, value.elts[index].lineno
        return where, node.lineno
    return where, None


def result_group_order() -> tuple[str, ...]:
    """`dev_constants.RECALL_RESULT_GROUP_ORDER`, read now (S98), checked to
    name each of the six groups exactly once. A mis-edited list raises
    `ValueError` whose text names the bad entry, the file and the line number
    (S143)."""
    order = tuple(dev_constants.RECALL_RESULT_GROUP_ORDER)
    problem = _group_order_problem(order)
    if problem is not None:
        index, why = problem
        where, line = _dev_constants_location(index, length=len(order))
        place = f"{where}:{line}" if line is not None else f"{where} (line unknown)"
        raise ValueError(
            f"{place}: RECALL_RESULT_GROUP_ORDER is mis-edited: {why}. "
            f"It must name each of {sorted(RESULT_GROUPS)} exactly once."
        )
    return order


# A list that does not name each group once fails at import (loudly), not on
# every recall turn inside the fail-soft wrapper (S98 review F6).
result_group_order()


def order_result_groups(
    groups: Mapping[str, Sequence[str]], cap: int, guaranteed: Sequence[str] = ()
) -> list[str]:
    """The final result ids, at most `cap` (spec §3, §4, §7; S33, S55, S64,
    S97, S98; plan P-10): the `guaranteed` ids are kept first, in the order
    given (the S64 order), up to `cap` (the excess dropped); the remaining
    slots go to the other ids in group order (`result_group_order()`), each
    group in its own order. The kept ids are returned in group order; an id
    listed in more than one group counts once, at its first group in that
    order. A guaranteed id must be in one of `groups`."""
    order = result_group_order()
    cap = max(cap, 0)
    in_order = list(dict.fromkeys(mid for name in order for mid in groups.get(name, ())))
    listed = set(in_order)
    kept: set[str] = set()
    for mid in guaranteed:
        if len(kept) >= cap:
            break
        if mid in listed:
            kept.add(mid)
    for mid in in_order:
        if len(kept) >= cap:
            break
        kept.add(mid)
    return [mid for mid in in_order if mid in kept]


def paragraph_result_groups(
    search: ParagraphSearch,
) -> tuple[dict[str, list[AssembledHit]], list[AssembledHit]]:
    """Every floor-clearing semantic result of a message, by group, each
    group in its own order, and the guaranteed results in the S64 order
    (spec §3, §4, §7; plan P-10; criteria C8b, C8c):

      reranked genuine / reranked family: by anchor-normalized score across
          paragraphs;
      cosine genuine: each cosine-path paragraph's best genuine memory at the
          head (by cosine), then the rest by cosine;
      cosine family: the monologue-family results on the cosine scale by
          cosine, those of cosine-path paragraphs and every reranked
          paragraph's cosine tail (S82), regrouped across paragraphs and
          de-duplicated (S86).

    Only results that clear their own path's floor enter (scales are never
    compared: reranked groups sort by normalized score, cosine groups by
    cosine). Guaranteed (S33, S44, S55, S64): each reranked paragraph's best
    genuine result, by normalized score, then each cosine-path paragraph's,
    by cosine; a paragraph whose best genuine memory does not clear its floor
    (or that has only monologue-family results) has none."""
    pool = search.pool

    def family(mid: str) -> bool:
        return is_monologue_family(pool[mid][0])

    members: dict[str, list[AssembledHit]] = {
        RERANKED_GENUINE: [],
        RERANKED_FAMILY: [],
        COSINE_GENUINE: [],
        COSINE_FAMILY: [],
    }
    guaranteed_reranked: list[AssembledHit] = []
    guaranteed_cosine: list[AssembledHit] = []
    for outcome in search.outcomes:
        gated = outcome.gated
        if gated is None:
            continue
        rankings = [gated] if gated.tail is None else [gated, gated.tail]
        for ranking in rankings:
            for mid, score in ranking.ranked:
                if score >= ranking.pass_mark:
                    hit = AssembledHit(mid, score, ranking.path, outcome.index, family(mid))
                    members[hit.group].append(hit)
        own_genuine = [
            AssembledHit(mid, score, gated.path, outcome.index, False)
            for mid, score in gated.ranked
            if score >= gated.pass_mark and not family(mid)
        ]
        if own_genuine:
            best = max(own_genuine, key=lambda hit: hit.score)
            (guaranteed_reranked if gated.path == RERANKED_PATH else guaranteed_cosine).append(best)

    def by_score(hits: list[AssembledHit]) -> list[AssembledHit]:
        return sorted(hits, key=lambda hit: -hit.score)

    def first_per_id(hits: list[AssembledHit]) -> list[AssembledHit]:
        seen: set[str] = set()
        out = []
        for hit in hits:
            if hit.memory_id not in seen:
                seen.add(hit.memory_id)
                out.append(hit)
        return out

    heads = by_score(guaranteed_cosine)
    head_ids = {hit.memory_id for hit in heads}
    groups = {
        RERANKED_GENUINE: by_score(members[RERANKED_GENUINE]),
        RERANKED_FAMILY: by_score(members[RERANKED_FAMILY]),
        COSINE_GENUINE: heads
        + [hit for hit in by_score(members[COSINE_GENUINE]) if hit.memory_id not in head_ids],
        COSINE_FAMILY: first_per_id(by_score(members[COSINE_FAMILY])),
    }
    return groups, by_score(guaranteed_reranked) + heads


def assemble_paragraph_results(search: ParagraphSearch, cap: int) -> list[AssembledHit]:
    """The semantic results alone, in the final order, at most `cap`
    (`paragraph_result_groups` then `order_result_groups` with no keyword
    hits): the guaranteed results first, the rest in group order."""
    return assemble_with_guarantees(search, cap)[0]


def assemble_with_guarantees(search: ParagraphSearch, cap: int) -> tuple[list[AssembledHit], list[str]]:
    """`assemble_paragraph_results`, plus every guaranteed id in the S64
    order (whether or not the cap kept it)."""
    groups, guaranteed = paragraph_result_groups(search)
    by_id: dict[str, AssembledHit] = {}
    for name in result_group_order():
        for hit in groups.get(name, ()):
            by_id.setdefault(hit.memory_id, hit)
    guaranteed_ids = [hit.memory_id for hit in guaranteed]
    ids = order_result_groups(
        {name: [hit.memory_id for hit in hits] for name, hits in groups.items()},
        cap,
        guaranteed_ids,
    )
    return [by_id[mid] for mid in ids], guaranteed_ids


def merge_with_keyword(
    semantic: Sequence[tuple[Memory, str]],
    keyword: Sequence[Memory],
    *,
    cap: int,
    guaranteed: Sequence[str] = (),
) -> list[Memory]:
    """The final result list of a turn or a tool call with semantic results
    (spec §4, S97, S98; I4): `semantic` is `(memory, group)` in the semantic
    order, `keyword` the keyword hits in their own final order (name hits
    first, genuine before monologue-family). A keyword hit that is also a
    semantic result is dropped from the keyword side (dedup favors semantic,
    I4); the rest split into the keyword genuine and keyword family groups.
    The six groups are then ordered by `order_result_groups`, at most `cap`,
    the `guaranteed` semantic ids (in their S64 order) kept first."""
    by_id: dict[str, Memory] = {}
    groups: dict[str, list[str]] = {name: [] for name in RESULT_GROUPS}
    for memory, group in semantic:
        if memory.id not in by_id:
            by_id[memory.id] = memory
            groups[group].append(memory.id)
    for memory in keyword:
        if memory.id not in by_id:
            by_id[memory.id] = memory
            groups[KEYWORD_FAMILY if is_monologue_family(memory) else KEYWORD_GENUINE].append(
                memory.id
            )
    semantic_ids = {memory.id for memory, _ in semantic}
    ids = order_result_groups(groups, cap, [mid for mid in guaranteed if mid in semantic_ids])
    return [by_id[mid] for mid in ids]


def run_semantic_recall(
    store: MemoryStore,
    persona_dir: Path,
    user_input: str,
    *,
    keyword_words: Collection[str] | None = None,
) -> SemanticRecallResult | None:
    """Attempt semantic-PRIMARY recall for one turn.

    Name-recall fix R6 (spec §3): the message is searched per paragraph
    (`search_paragraphs`: split with `str.splitlines()`, blank paragraphs and
    paragraphs with none of `keyword_words` dropped, every paragraph embedded
    in one batch via the shared process-cached embedding provider, each
    searched and cut on its own, the cuts merged, one per-message time budget
    for the embeds, scans and reranks) and the results assembled across
    paragraphs (`assemble_paragraph_results`, capped at `MAX_STANDOUT_COUNT`).
    A one-paragraph message behaves as before apart from its embed + scan
    time counting in the budget (S61). `keyword_words` is the caller's
    keyword token set plus the words of the known names in the message
    (`brain.chat.prompt`); `None` (direct callers) drops blank paragraphs
    only.

    Each query's coarse cut (the top `relevance.CANDIDATE_POOL` genuine
    memories by cosine plus the monologue-family memories in its plain
    top-`CANDIDATE_POOL`, `genuine_first_coarse_cut`, S77) is
    `rank_and_gate`d:

      - reranked path: a cross-encoder rerank of a per-message-width prefix
        (`reranker.rerank_for_recall`, anchors on top, >= 5 real
        candidates), gated by the CALIBRATED rerank floor
        (`store.get_reranker_floor`) on the anchor-normalized score;
      - cosine path (name-recall fix R2, spec §2): when fewer than 5 real
        candidates fit or exist, the reranker fails to load/score, or its
        normalization falls back, the coarse cut ranked by cosine, gated by
        the cosine floor (`store.get_cosine_floor`). Not keyword-only.

    Passive recall logs each query's calibration row(s), each stamped with its
    own true scale and carrying that query (the paragraph, not the whole
    message) as its text (a reranked query logs only the `normalized` row: its
    cosine tail, spec §4 S82, writes none, S84; a query whose rerank scored but
    whose rerank floor was unavailable logs the `normalized` row, then the
    cosine path's `cosine` row).

    Returns a populated `SemanticRecallResult` ONLY when at least one
    candidate clears the operative floor of the path taken. Returns `None`
    for every INCONCLUSIVE case:
      - nothing clears the floor,
      - an empty or sparse candidate pool (cold-start / idle backfill not
        caught up — "graceful warm-up"),
      - an embed failure, or no cosine floor yet on the cosine path (the
        bootstrap not computed yet, or failed and awaiting the next-lull retry: no
        gate is possible; never an ungated ranking),
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
    search: ParagraphSearch | None = None
    try:
        search = search_paragraphs(
            store, user_input, keyword_words=keyword_words, log_calibration=True
        )
        if search is None:
            return None
        assembled, guaranteed_ids = assemble_with_guarantees(search, MAX_STANDOUT_COUNT)
        tiers = _surfacing([(hit.memory_id, hit.score) for hit in assembled])
        if tiers is None:
            return None
        pool = search.pool
        full = [pool[mid][0] for mid in tiers.full_ids]
        snippet = [pool[mid][0] for mid in tiers.snippet_ids]
        hits = [
            SemanticHit(
                memory=pool[hit.memory_id][0],
                score=hit.score,
                path=hit.path,
                paragraph=hit.paragraph,
                monologue_family=hit.monologue_family,
            )
            for hit in assembled[: len(full) + len(snippet)]
        ]
        # The legacy single-ranking fields describe the first query that has
        # a gated ranking (the only one on a one-paragraph message); every
        # query's own ranking is in `paragraphs`.
        primary = next(o.gated for o in search.outcomes if o.gated is not None)
        tail = primary.tail
        return SemanticRecallResult(
            full=full,
            snippet=snippet,
            scores=dict(primary.ranked),
            hits=hits,
            path=primary.path,
            scale=primary.scale,
            pass_mark=primary.pass_mark,
            tail_scores=dict(tail.ranked) if tail is not None else {},
            tail_scale=tail.scale if tail is not None else None,
            tail_pass_mark=tail.pass_mark if tail is not None else None,
            paragraphs=search.outcomes,
            whole_message_fallback=search.whole_message_fallback,
            total_width=search.total_width,
            budget=search.budget,
            rerank_budget=search.rerank_budget,
            guaranteed_ids=tuple(guaranteed_ids),
        )
    except Exception:  # noqa: BLE001 — fail-soft: ANY failure demotes to lexical, never raises
        log.warning(
            "run_semantic_recall: semantic path failed — falling back to lexical",
            exc_info=True,
        )
        return None
    finally:
        # One diagnostics record per passive recall (spec section 6, S15, P-22),
        # fail-soft: `emit_recall_diagnostics` never raises.
        emit_recall_diagnostics(persona_dir, search, source=recall_diagnostics.SOURCE_PASSIVE)
