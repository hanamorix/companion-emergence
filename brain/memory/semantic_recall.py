"""semantic_recall.py — semantic-PRIMARY retrieval + option-4 surfacing.

Stage 3 of the local-semantic-retrieval build
(``~/.claude/plans/memory-dream-rework-semantic-retrieval-brief.md``,
decisions 3-5, DIRECTION CORRECTED 2026-09-09: semantic is PRIMARY, the
existing lexical/importance/hebbian/recency blend is the FALLBACK/backstop).

Recall runs semantic cosine as the FIRST retrieval attempt for a turn. When
it produces a CONCLUSIVE result (clear standouts, decision 5's first two
tiers) that result is surfaced and the existing lexical path never runs for
that turn. When semantic is INCONCLUSIVE — a bunched clump, no/sparse
candidate pool (cold-start / idle backfill hasn't caught up yet — the
"graceful warm-up" contract), or any embedding-infra failure — this module
returns ``None`` and the caller (``brain.chat.prompt._build_recall_block``)
falls through UNCHANGED to the existing lexical/blend retrieval, exactly as
it behaved before this stage. This module never touches that fallback path.

This module owns:
  - the semantic candidate-pool builder (active memories that already have a
    cached vector under the CURRENT model_id — never triggers a new embed
    for an uncached memory; that bulk-embed job is Stage 2's, off this hot
    path)
  - the query embed (the ONE allowed synchronous in-turn embed, decision 4)
  - cosine scoring
  - the option-4 relative-gap shape classifier, with a SANE COLD-START
    BOOTSTRAP floor/gap — Stage 4 (NOT this stage) will replace these with a
    per-persona auto-calibrated value recomputed on the weekly rollover; see
    `SemanticCalibration` for the seam it plugs into
  - the surfacing-tier decision (which candidate ids are "full" vs
    "snippet"). Rendering (actual body/snippet text, recall-counter ticks)
    stays owned by ``brain.chat.prompt``, mirroring how it already
    renders/bumps the lexical path — this module only decides WHICH ids go
    in which bucket, in cosine-SELECTION order; presentation order is the
    caller's call (spec: "cosine selects, the normal sort orders the
    presentation").

Does NOT touch: the lexical/blend fallback itself (untouched, reused
as-is), the embed-on-write / idle-backfill machinery (Stage 2, unaffected),
or clustering (Stage 5, unaffected).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from brain.memory.embeddings import build_embedding_cache, cosine_similarity, hash_content
from brain.memory.store import Memory, MemoryStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cold-start bootstrap calibration (Stage 4 plug-in seam)
# ---------------------------------------------------------------------------
#
# SANE DEFAULTS, chosen from the spec's own empirical sanity numbers
# (bge-small-en-v1.5, no-AVX2 dev VM, 2026-09-08 test): paraphrase 0.749 >
# keyword-overlap-decoy 0.602 > unrelated 0.424. These are deliberately
# conservative, NOT-tuned-per-persona cold-start values — Stage 4 (spec
# decision 5's "Calibration" bullet, explicitly out of scope for this stage)
# replaces them with a per-persona floor/gap derived from that persona's own
# current score distribution, recomputed on the weekly rollover
# (`brain/chat/rollover.py`'s weekly-cap trigger). Until Stage 4 lands, every
# persona uses these bootstrap constants.

# Minimum cosine for a candidate to count AT ALL. Set just above the spec's
# measured "unrelated" score (0.424) so a genuinely-unrelated memory can't
# enter the standout/clump judgment, while sitting below the measured
# "wrong but keyword-related" decoy score (0.602) so a same-topic-but-wrong
# match still gets weighed — it's the GAP logic below, not the floor, that
# should demote a decoy when a real match is present.
SEMANTIC_FLOOR_BOOTSTRAP = 0.45

# Minimum score DROP between two consecutively-ranked (floor-passing)
# candidates for the judgment to call that a real "cliff" — standouts above
# it, an undifferentiated remainder below/beyond it — rather than ordinary
# score jitter within one topic cluster. 0.08 is a modest fraction of the
# ~0.15-0.3 spread the spec's sanity numbers show between a real match and a
# decoy/unrelated score: small enough not to miss a genuine standout, large
# enough that jitter inside a cluster doesn't get misread as a cliff.
SEMANTIC_GAP_BOOTSTRAP = 0.08

# The largest standout cluster the surfacing rule ever recognises (decision
# 5: there is no tier for more than 9 clear standouts — 10+ IS a clump by
# definition, "bunched, ~>=10"). NOT a Stage-4 tunable, part of the fixed
# shape of the three-tier rule.
MAX_STANDOUT_COUNT = 9

# How many top-ranked candidates the cliff scan looks at. One MORE than
# MAX_STANDOUT_COUNT: confirming a cliff AFTER the 9th-ranked candidate (the
# largest possible standout cluster) requires seeing the 10th-ranked score
# too — a window capped at exactly 9 could never observe that drop and would
# misclassify a clean "9 standouts" case as a clump for lack of anything to
# compare the 9th candidate against.
_SCAN_WINDOW = MAX_STANDOUT_COUNT + 1

# Surfacing-tier boundaries (decision 5, option 4). Not a Stage-4 tunable —
# this is the fixed SHAPE of the three-tier rule itself; only the floor/gap
# that decide what counts as a "standout" are the calibration target.
FULL_INJECT_STANDOUT_MAX = 5  # <=5 clear standouts: all rendered in full


@dataclass(frozen=True)
class SemanticCalibration:
    """The floor/gap pair the shape classifier runs against.

    A plain value holder — deliberately NOT the calibration algorithm
    itself. `bootstrap()` is the cold-start default every persona uses until
    Stage 4 (per-persona auto-recalibration on the weekly rollover) lands
    and starts producing persona-specific values. Stage 4 plugs in by
    constructing this from ITS OWN recomputed floor/gap (e.g. loaded from a
    per-persona state file written on the weekly rollover) instead of
    calling `bootstrap()` — `classify_semantic_shape` and `run_semantic_recall`
    take this as a parameter for exactly that reason, rather than reading
    the module constants directly, so Stage 4 needs no change to either.
    """

    floor: float
    gap: float

    @staticmethod
    def bootstrap() -> SemanticCalibration:
        return SemanticCalibration(floor=SEMANTIC_FLOOR_BOOTSTRAP, gap=SEMANTIC_GAP_BOOTSTRAP)


ShapeKind = Literal["standouts", "clump", "none"]


@dataclass(frozen=True)
class ShapeResult:
    """The classified SHAPE of a sorted-descending semantic score list.

    kind="standouts": a clean cliff was found; `standout_count` candidates
        (ranked 1..standout_count) are the standout cluster.
    kind="clump": floor-passing candidates exist but no clean cliff was
        found within the scan window — INCONCLUSIVE, caller falls back.
    kind="none": no candidate passed the floor at all — INCONCLUSIVE
        (indistinguishable, for surfacing purposes, from "clump"; kept as
        its own value for diagnostics/tests).
    """

    kind: ShapeKind
    standout_count: int  # 0 for "clump"/"none"


def classify_semantic_shape(
    scored_desc: list[tuple[str, float]],
    *,
    calibration: SemanticCalibration,
) -> ShapeResult:
    """Classify the SHAPE of a sorted-descending (memory_id, cosine) list.

    Relative-gap judgment (decision 5): the top candidate must clear
    `calibration.floor` at all, or the shape is "none" (no semantic
    candidate is even worth considering). Otherwise, walk ranked pairs
    (rank i, rank i+1) from the top: the first pair where rank i+1 has
    EITHER fallen below the floor OR dropped by `>= calibration.gap` from
    rank i is a "cliff" — everything at-or-above rank i is the standout
    cluster (`standout_count = i + 1`), distinct from the undifferentiated
    remainder below the cliff. No cliff found within the scan (bounded to
    `MAX_STANDOUT_COUNT` possible standouts, decision 5's tier ceiling) =>
    a bunched clump (INCONCLUSIVE — the caller falls back to lexical). A
    single floor-passing candidate with nothing else in the pool is
    trivially a 1-item standout cluster (nothing to compare it against, so
    there is no clump to detect).

    A candidate below the floor is NOT pre-filtered out before scanning —
    its presence (or the absence of any further candidate at all) is
    exactly what the pairwise scan uses to detect where the standout
    cluster ends.
    """
    if not scored_desc or scored_desc[0][1] < calibration.floor:
        return ShapeResult(kind="none", standout_count=0)
    if len(scored_desc) == 1:
        return ShapeResult(kind="standouts", standout_count=1)

    window = scored_desc[:_SCAN_WINDOW]
    for i in range(len(window) - 1):
        score, next_score = window[i][1], window[i + 1][1]
        if next_score < calibration.floor or (score - next_score) >= calibration.gap:
            return ShapeResult(kind="standouts", standout_count=i + 1)
    return ShapeResult(kind="clump", standout_count=0)


@dataclass(frozen=True)
class SemanticSurfacing:
    """The surfacing-tier id split for a CONCLUSIVE ("standouts") shape.

    Both lists are in cosine-SELECTION order (highest cosine first) — cosine
    decides membership and ranking of the candidate pool; the caller decides
    PRESENTATION order for the snippet tier (spec: "cosine selects, the
    normal sort orders the presentation").
    """

    full_ids: list[str]
    snippet_ids: list[str]


def surfacing_tiers(
    scored_desc: list[tuple[str, float]], shape: ShapeResult
) -> SemanticSurfacing | None:
    """Split a conclusive standout shape into (full, snippet) id tiers.

    Returns None for a non-"standouts" shape (clump/none) — the caller must
    fall back to the lexical path; this function only decides surfacing for
    an already-conclusive semantic result.

    `scored_desc` must be the SAME sorted-descending list `shape` was
    classified from (this function trusts `shape.standout_count` as an
    index into it).
    """
    if shape.kind != "standouts":
        return None
    standout_ids = [mid for mid, _ in scored_desc[: shape.standout_count]]
    if shape.standout_count <= FULL_INJECT_STANDOUT_MAX:
        return SemanticSurfacing(full_ids=standout_ids, snippet_ids=[])
    # 6-9 (SEMANTIC_SCAN_WINDOW caps standout_count at 9): top 5 full, the
    # rest snippet.
    return SemanticSurfacing(
        full_ids=standout_ids[:FULL_INJECT_STANDOUT_MAX],
        snippet_ids=standout_ids[FULL_INJECT_STANDOUT_MAX:],
    )


def build_semantic_candidate_pool(
    store: MemoryStore, embeddings_cache
) -> dict[str, tuple[Memory, np.ndarray]]:
    """Active memories that already have a cached vector under THIS cache's
    model_id, paired with that vector.

    Deliberately NEVER computes a new embedding for an uncached memory —
    that would be exactly the forbidden hot-path bulk embed. An active
    memory with no cached vector yet (the idle backfill hasn't reached it)
    simply isn't a semantic candidate this turn. This IS the "graceful
    warm-up" contract from the spec: semantic coverage grows as the corpus
    embeds; a cold/sparse persona degrades to the lexical fallback (empty
    pool here -> `run_semantic_recall` returns None) until backfill catches
    up.

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
        vec = hash_to_vector.get(hash_content(mem.content))
        if vec is not None:
            pool[mem.id] = (mem, vec)
    return pool


@dataclass(frozen=True)
class SemanticRecallResult:
    """A CONCLUSIVE semantic-primary recall — the caller renders this and
    skips the lexical fallback entirely for this turn.

    `full` / `snippet` are Memory lists in cosine-SELECTION order; `scores`
    maps memory_id -> cosine similarity for callers that want the raw
    number (tests, logging).
    """

    full: list[Memory]
    snippet: list[Memory]
    scores: dict[str, float]


def run_semantic_recall(
    store: MemoryStore,
    persona_dir: Path,
    user_input: str,
    *,
    calibration: SemanticCalibration | None = None,
) -> SemanticRecallResult | None:
    """Attempt semantic-PRIMARY recall for one turn.

    Embeds `user_input` (~34ms, synchronous — the ONE allowed in-turn embed,
    spec decision 4) via this persona's embedding cache/provider, cosines it
    against the model_id-scoped candidate pool, and classifies the result
    shape (decision 5).

    Returns a populated `SemanticRecallResult` ONLY when the shape is
    conclusive (a standout cluster, decision 5's first two tiers). Returns
    `None` for every INCONCLUSIVE case:
      - a bunched clump / no clean cliff,
      - an empty or sparse candidate pool (cold-start / idle backfill not
        caught up — "graceful warm-up"),
      - ANY failure ANYWHERE in this function — constructing the local
        embedding provider/cache, embedding the query, building the
        candidate pool, cosine scoring, or shape classification/surfacing
        (fail-soft: a broken/missing local model, or a transient store
        error such as a locked sqlite db during the background backfill,
        must never break recall — it only demotes this turn to
        lexical-primary, matching the spec's warm-up contract). The whole
        body is wrapped in a broad `except Exception` for exactly this
        reason: earlier revisions only caught the cache-open and query-embed
        steps, leaving pool-build/scoring/classification exceptions to
        propagate straight out — this function's own contract (and this
        docstring) always said "ANY failure", so the catch now actually
        matches it.

    Never renders anything and never bumps `recall_count` itself — the
    caller (`brain.chat.prompt._build_recall_block`) owns rendering and the
    recall-counter ticks, exactly as it already does for the lexical path.
    On `None`, the caller falls through to that EXISTING lexical/blend path,
    unchanged.
    """
    calibration = calibration or SemanticCalibration.bootstrap()
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

            scored = [(mid, cosine_similarity(query_vec, vec)) for mid, (_, vec) in pool.items()]
            scored.sort(key=lambda pair: -pair[1])

            shape = classify_semantic_shape(scored, calibration=calibration)
            tiers = surfacing_tiers(scored, shape)
            if tiers is None:
                return None

            full = [pool[mid][0] for mid in tiers.full_ids]
            snippet = [pool[mid][0] for mid in tiers.snippet_ids]
            return SemanticRecallResult(full=full, snippet=snippet, scores=dict(scored))
        except Exception:  # noqa: BLE001 — fail-soft: ANY failure demotes to lexical, never raises
            log.warning(
                "run_semantic_recall: semantic path failed after opening the embedding cache "
                "— falling back to lexical",
                exc_info=True,
            )
            return None
    finally:
        embeddings_cache.close()
