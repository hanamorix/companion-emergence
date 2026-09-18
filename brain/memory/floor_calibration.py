"""Reranker abstention-floor derivation for F2a's daily calibration tick
(#250 §7, inc7).

Runs ENTIRELY inside `brain.bridge.supervisor._run_calibration_tick` (the
once-daily idle-gated cadence, after inc6's judge-labeling pass), never on
the per-turn recall path — mirrors `relevance_judge.py`'s
"offline-only orchestration module" shape.

Replaces the hardcoded `brain.memory.semantic_recall.RERANK_FLOOR` constant
with a DB-adaptive floor, per persona (i.e. per the ONE memories.db this
process is running against) and per reranker model id (so a jina fp32<->fp16
precision flip, or any future reranker swap, never mixes floors fit against
different score scales):

  1. `fit_threshold_fbeta` — the raw-logit cutoff fit (Youden's-J / F-beta
     family, spec's cited research basis), recall-leaning via `beta > 1`.
  2. `ema_update` — smooths the newly-fit floor against the previous run's
     persisted floor, so no single day's fit can jump the floor outright.
  3. `stability_gate_accepts` — a numpy-only bootstrap-CI check: resamples
     the day's labeled pairs, refits the threshold on each resample, and
     only accepts the update if the PRIOR floor still falls inside the
     resulting 95% CI (today's data is statistically consistent with
     established history) — a genuinely noisy/outlier day HOLDS instead of
     swinging the floor.
  4. `derive_and_persist_floor` — the ONE orchestration entry point the
     daily tick calls: reads labeled pairs off `calibration_log`
     (`MemoryStore.labeled_calibration_pairs`), decides cold-start vs a real
     fit (OUTCOME-based on accumulated usable labeled pairs, never a day
     count), runs the gate, and persists the result via
     `MemoryStore.write_reranker_floor` / `get_reranker_floor` (I1: a table
     in memories.db, never a side file).

This module ALSO derives `store.CALIBRATION_LOG_RETENTION_WINDOW_DAYS`'s
FINAL value (spec Section 5's "MUST before ship" pruning window, left
PROVISIONAL at 14.0 by inc5 pending this increment's sample-size tunable) —
see `RETENTION_WINDOW_DAYS_DEFAULT` below. `store.py` imports that constant
directly rather than re-deriving it, so the tunable key
(`calibration.retention_window_days`) stays owned by `store.py`/
`prune_calibration_log` (inc5's contract), only its DEFAULT VALUE moves here.

Cold-start (spec: "until enough daily logs have accumulated ... serves a
small bootstrap floor rather than having no floor or crashing") reuses
`reranker.py`'s own `_FP16_GATE_PAIRS` — the same small bundled
representative (query, doc) set §2's fp16-vs-fp32 self-check already ships —
scored through the LIVE production reranker provider, so the bootstrap floor
sits on the same scale real per-turn scores will use.

HOT-PATH no-persisted-floor bootstrap (F2a inc8, #250 §7 UPDATED, Roy
2026-09-18): `get_bootstrap_floor` below is the OTHER cold-start path this
module owns — distinct from `derive_and_persist_floor`'s own cold-start
branch (which runs ONLY inside the once-daily tick and PERSISTS its result).
`get_bootstrap_floor` is called SYNCHRONOUSLY from
`MemoryStore.get_reranker_floor` whenever no `reranker_floor_calibration`
row exists yet for a model_id — i.e. it can fire on the per-turn hot path,
on the very FIRST no-row recall/precision-check of the process. It is
computed ONCE per model_id and cached process-wide (never persisted to
`memories.db` — a transient, in-memory-only fallback that a real persisted
row always supersedes, see that function's docstring), and it deliberately
builds its own reranker provider via `reranker._bootstrap_reranker_provider`
rather than `reranker.build_reranker_provider` — going through the latter
would recurse back into `MemoryStore.get_reranker_floor` via the fp16/fp32
precision self-check (see `_bootstrap_reranker_provider`'s docstring for the
full loop this avoids). Never touches the §6 torch-backed relevance judge —
jina (ONNX, via `reranker.CrossEncoderProvider`) is the only model involved.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from brain import tunables

if TYPE_CHECKING:
    from brain.memory.reranker import RerankerProvider
    from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (I7/I3) — every numeric literal below is either registered here
# (operator-overridable via tunables.json, mirroring relevance_judge.py's
# CALIBRATION_SAMPLE_ROWS/AMBIGUOUS_BAND_HALF_WIDTH pattern) or derived from
# one of these at call time.
# ---------------------------------------------------------------------------

# F-beta value for the recall-leaning cutoff fit (spec Section 7:
# "Recall-leaning operating point ... carried forward from the prior design
# intent"). DEFAULT 2.0 [Planning-resolved 2026-09-17, Fixing FORK 1 — see
# the spec's "Open reconfirmations" section]: beta=2 is the canonical
# recall-leaning F-measure (recall weighted 2x precision), squarely within
# the signed-off "recall-lean" mandate and reversible via the tunable — not
# an owner-blocking pin.
FLOOR_FIT_BETA: float = tunables.register("calibration.floor_fit_beta", 2.0)

# EMA smoothing / drift-responsiveness window, in DAYS (spec Section 7: "the
# newly-computed floor is EMA-smoothed against the previous run's floor").
# alpha = 2 / (N + 1) is the standard N-period EMA weighting; N=7 (one
# calendar week) smooths day-of-week query-mix variation (weekday vs
# weekend usage patterns) while staying responsive within roughly a week of
# actual corpus drift. This SAME N also anchors the retention window's
# drift-responsiveness bound below (RETENTION_WINDOW_DAYS_DEFAULT) — one
# shared "how many days of history matter" judgment call, not two
# independently-argued magic numbers.
FLOOR_EMA_WINDOW_DAYS: float = tunables.register("calibration.floor_ema_window_days", 7.0)

# Lower bound on labeled (reranker_score, label) PAIRS required before a
# REAL fit is attempted (below this, the tick serves the cold-start
# bootstrap floor instead — spec Section 7's cold-start bootstrap). The
# spec's own citation: a cutoff fit is "robust at hundreds of pairs" — 200
# is the upper end of "hundreds," chosen for a comfortable margin over the
# bare minimum rather than the smallest number that technically qualifies.
# This SAME constant also drives the retention window's sample-drawable-days
# lower bound below.
FLOOR_FIT_MIN_LABELED_PAIRS: int = tunables.register("calibration.floor_fit_min_labeled_pairs", 200)

# Stability gate's confidence level (spec Section 7 red-team / #250 §7:
# "gated by a stability/noise check ... bootstrap-CI vs previous EMA ...
# 95% CI" — the spec pins this number literally).
FLOOR_STABILITY_CI: float = tunables.register("calibration.floor_stability_ci", 0.95)

# Bootstrap resample count for the stability gate's CI. Percentile-estimate
# error shrinks roughly like 1/sqrt(iterations); 200 balances a reasonably
# smooth CI estimate against the once-daily idle compute this adds on the
# no-AVX2 potato baseline (mirrors CALIBRATION_SAMPLE_ROWS's own
# "bounded-for-potato" reasoning in relevance_judge.py) — this is pure numpy
# resampling of already-labeled scores, not a model forward pass, so 200
# resamples is cheap even on modest hardware.
FLOOR_STABILITY_BOOTSTRAP_ITERATIONS: int = tunables.register(
    "calibration.floor_stability_bootstrap_iterations", 200
)


def _worst_case_daily_labeled_pairs() -> int:
    """Worst-case daily yield of judged (score, label) pairs the tick can
    add to the labeled pool, for the retention-window derivation below.

    Lazily imports `relevance_judge.CALIBRATION_SAMPLE_ROWS` (rather than a
    top-level import) purely to keep this module's own import graph
    independent of import ORDER — by the time this function is actually
    CALLED (module load, see `RETENTION_WINDOW_DAYS_DEFAULT` below, or a
    test), `relevance_judge` is safe to import regardless of which module
    got there first (see relevance_judge.py's own TYPE_CHECKING-gated
    MemoryStore import for the matching half of this fix). Mirrors that
    constant's own "worst case: 1 candidate/row" reasoning — a persistently
    narrow rerank pool is the floor of what one labeled ROW yields in
    labeled PAIRS.
    """
    from brain.memory.relevance_judge import CALIBRATION_SAMPLE_ROWS

    return int(CALIBRATION_SAMPLE_ROWS)


def derive_retention_window_days(
    *,
    min_labeled_pairs: int | None = None,
    ema_window_days: float | None = None,
    worst_case_daily_yield: int | None = None,
) -> float:
    """Finalized `calibration_log` retention window (spec Section 5's "MUST
    before ship" pruning requirement / acceptance 5b), replacing inc5's
    provisional 14.0-day placeholder.

    `max(sample-drawable-days, drift-responsiveness)` per the spec's own
    wording:
      - sample-drawable-days: how many days, at the WORST-CASE daily labeled
        yield, it takes to accumulate `FLOOR_FIT_MIN_LABELED_PAIRS` — the
        floor fit's own statistical lower bound (§7).
      - drift-responsiveness: `FLOOR_EMA_WINDOW_DAYS` — reusing the EMA
        smoothing window as the "how many days of history stay relevant
        before they're stale" bound (§5's "short enough to stay responsive
        to corpus drift... to bound potato storage"), rather than arguing a
        second, independent number for the same underlying judgment call.

    All three inputs default to the live tunables (operator-overridable);
    explicit args are exposed for tests that want to pin the formula's
    shape without depending on the registered defaults.
    """
    if min_labeled_pairs is None:
        min_labeled_pairs = tunables.get_tunable(
            "calibration.floor_fit_min_labeled_pairs", FLOOR_FIT_MIN_LABELED_PAIRS
        )
    if ema_window_days is None:
        ema_window_days = tunables.get_tunable(
            "calibration.floor_ema_window_days", FLOOR_EMA_WINDOW_DAYS
        )
    if worst_case_daily_yield is None:
        worst_case_daily_yield = _worst_case_daily_labeled_pairs()
    sample_drawable_days = math.ceil(min_labeled_pairs / max(1, worst_case_daily_yield))
    return float(max(sample_drawable_days, ema_window_days))


# Computed once at import time (mirrors store.py's own prior provisional
# constant being a plain module-level float) — `store.py` imports THIS
# value as its tunable's default rather than re-deriving it, keeping the
# derivation's one home here while the tunable KEY/registration stays owned
# by store.py/prune_calibration_log per inc5's existing contract.
RETENTION_WINDOW_DAYS_DEFAULT: float = derive_retention_window_days()


# ---------------------------------------------------------------------------
# Threshold fit — Youden's-J / F-beta family on raw (unbounded) logits.
# ---------------------------------------------------------------------------


def fit_threshold_fbeta(pairs: list[tuple[float, str]], *, beta: float) -> float:
    """Return the raw-logit threshold maximizing F-beta over judged
    'relevant' vs 'irrelevant' pairs (spec Section 7: "a cutoff-fitting
    method in the Youden's-J / F-beta family on the raw logits").

    `beta > 1` biases the fit toward RECALL (accepting some false-positive
    surfacing over missing a true positive) — the spec's "recall-leaning
    operating point." Scans every midpoint between consecutive sorted
    unique scores (plus the two extremes) as a candidate threshold — the
    exhaustive, exact search for a 1-D cutoff (no gradient method needed:
    the candidate set is finite and small at the pair counts this fits
    against). Ties broken toward the LOWER (more inclusive, more
    recall-favoring) threshold, since `>` (not `>=`) is required to replace
    `best_threshold` and candidates are scanned in ascending order.

    Degenerate inputs (every pair judged the SAME class — no separation to
    fit) fall back to a threshold strictly below/above every observed score
    (serves everything, or nothing) rather than raising: a single-class
    sample is a real possibility (e.g. a very small or very lopsided day's
    sample) and must not crash the tick.
    """
    if not pairs:
        raise ValueError("fit_threshold_fbeta requires at least one labeled pair")
    scores = np.array([s for s, _ in pairs], dtype=np.float64)
    is_relevant = np.array([label == "relevant" for _, label in pairs], dtype=bool)

    if not is_relevant.any():
        # Nothing judged relevant — no threshold can produce a true
        # positive; set the floor above every score so nothing surfaces.
        return float(scores.max() + 1.0)
    if is_relevant.all():
        # Everything judged relevant — set the floor below every score so
        # everything surfaces (maximal recall, the only meaningful choice
        # with no irrelevant examples to separate from).
        return float(scores.min() - 1.0)

    sorted_scores = np.sort(scores)
    candidates = np.concatenate((
        [sorted_scores[0] - 1.0],
        (sorted_scores[:-1] + sorted_scores[1:]) / 2.0,
        [sorted_scores[-1] + 1.0],
    ))
    beta_sq = beta * beta
    best_f_beta = -1.0
    best_threshold = float(candidates[0])
    for threshold in candidates:
        predicted_relevant = scores >= threshold
        tp = int(np.sum(predicted_relevant & is_relevant))
        fp = int(np.sum(predicted_relevant & ~is_relevant))
        fn = int(np.sum(~predicted_relevant & is_relevant))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = beta_sq * precision + recall
        f_beta = (1.0 + beta_sq) * precision * recall / denom if denom > 0 else 0.0
        if f_beta > best_f_beta:
            best_f_beta = f_beta
            best_threshold = float(threshold)
    return best_threshold


# ---------------------------------------------------------------------------
# EMA smoothing + bootstrap stability gate.
# ---------------------------------------------------------------------------


def ema_update(prior_ema_floor: float | None, raw_floor: float, *, window_days: float) -> float:
    """EMA-smooth `raw_floor` (today's fresh fit) against `prior_ema_floor`
    (the previously persisted floor) — spec Section 7. `alpha = 2 /
    (window_days + 1)` is the standard N-period EMA weighting. No prior
    floor (the very first REAL derivation, exiting cold-start) returns
    `raw_floor` unchanged — there is nothing to smooth against yet."""
    if prior_ema_floor is None:
        return raw_floor
    alpha = 2.0 / (window_days + 1.0)
    return alpha * raw_floor + (1.0 - alpha) * prior_ema_floor


def _bootstrap_ci(
    pairs: list[tuple[float, str]],
    *,
    beta: float,
    ci: float,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap-resample `pairs` `iterations` times (with replacement),
    refit the threshold on each resample, and return the `ci`-confidence
    percentile interval of the resulting threshold distribution — numpy
    only, no scipy dependency."""
    n = len(pairs)
    thresholds = np.empty(iterations, dtype=np.float64)
    indices = np.arange(n)
    for i in range(iterations):
        sample_idx = rng.choice(indices, size=n, replace=True)
        sample = [pairs[j] for j in sample_idx]
        thresholds[i] = fit_threshold_fbeta(sample, beta=beta)
    tail = (1.0 - ci) / 2.0
    lo, hi = np.percentile(thresholds, [tail * 100.0, (1.0 - tail) * 100.0])
    return float(lo), float(hi)


def stability_gate_accepts(
    prior_ema_floor: float | None,
    pairs: list[tuple[float, str]],
    *,
    beta: float,
    ci: float,
    iterations: int,
    rng: np.random.Generator,
) -> bool:
    """Spec Section 7's stability/noise gate: "gated by a stability/noise
    check so a single noisy day's distribution cannot swing the floor — if
    the check trips, the update is held rather than applied."

    Bootstrap-resamples `pairs`, refits a threshold per resample, and
    accepts the update only if `prior_ema_floor` falls INSIDE the resulting
    CI — i.e. today's labeled sample is statistically consistent with the
    established trend. A day whose bootstrap distribution excludes the
    established floor is treated as the noisy/outlier case the spec
    describes, and the update is held (this function returns False; the
    caller must then NOT write a new floor this tick).

    No prior floor (nothing established yet to compare against — the first
    real derivation) trivially accepts; this mirrors `ema_update`'s own
    "nothing to smooth against yet" base case.
    """
    if prior_ema_floor is None:
        return True
    lo, hi = _bootstrap_ci(pairs, beta=beta, ci=ci, iterations=iterations, rng=rng)
    return lo <= prior_ema_floor <= hi


# ---------------------------------------------------------------------------
# Cold-start bootstrap fit — reuses reranker.py's §2 bundled pairs.
# ---------------------------------------------------------------------------


def _cold_start_pairs(reranker_provider: RerankerProvider) -> list[tuple[float, str]]:
    """Score `reranker.py`'s bundled `_FP16_GATE_PAIRS` through the LIVE
    production reranker provider, labeled per THAT module's own grouping
    comment ("# genuine (clearly relevant)" first 3, "# decoy (clearly
    irrelevant)" next 3) — the borderline last 4 are excluded here since
    they carry no clean ground-truth label for a threshold fit (they exist
    in reranker.py for the fp16/fp32 AGREEMENT check, which never needs a
    label, only agreement between two providers).

    Reuses this set (rather than shipping a second, separate bootstrap set)
    per spec Section 7's cold-start bootstrap: "reuse §2's bundled fp16-gate
    pairs for the same fit." Scored through the SAME provider real per-turn
    recall uses, so the bootstrap floor sits on the correct scale even
    before any real corpus data exists.
    """
    from brain.memory.reranker import _FP16_GATE_PAIRS

    labeled_slice = _FP16_GATE_PAIRS[:6]
    labels = ["relevant"] * 3 + ["irrelevant"] * 3
    pairs: list[tuple[float, str]] = []
    for (query, doc), label in zip(labeled_slice, labels, strict=True):
        (score,) = reranker_provider.rerank(query, [doc])
        pairs.append((float(score), label))
    return pairs


# ---------------------------------------------------------------------------
# Hot-path bootstrap floor (F2a inc8, #250 §7 UPDATED, Roy 2026-09-18) —
# the DEFAULT `get_reranker_floor` serves when NO persisted row exists yet,
# so semantic recall's existence is decoupled from the daily tick ever
# having fired. Process-wide cache, keyed by model_id, computed ONCE.
# ---------------------------------------------------------------------------

# model_id -> the bootstrap floor dict last derived for it (same shape as
# `MemoryStore.get_reranker_floor`'s persisted-row dict). Process-wide,
# mirrors `reranker.py`'s `_provider_cache`/`_latency_cache`/
# `_precision_decision_cache` pattern: computed once per model_id, reused by
# every later caller in the process, reset only by tests
# (`_reset_bootstrap_floor_cache`, wired into `tests/conftest.py`'s autouse
# fixture alongside the reranker module's own resets).
_bootstrap_floor_cache: dict[str, dict[str, Any]] = {}
_bootstrap_floor_cache_lock = threading.Lock()


def get_bootstrap_floor(reranker_model_id: str) -> dict[str, Any] | None:
    """Derived DEFAULT floor for `reranker_model_id`, served by
    `MemoryStore.get_reranker_floor` whenever no persisted
    `reranker_floor_calibration` row exists yet (spec Section 7, UPDATED
    2026-09-18 — Roy's bootstrap-floor ruling, F2a inc8): "no floor ->
    lexical" permanently coupled semantic recall's EXISTENCE to the daily
    calibration tick ever having fired for a given reranker model_id (an
    operator disabling calibration, or recall running before the tick's
    first idle moment, would silently and PERMANENTLY demote to lexical-only
    even with embeddings present) — this decouples the two by always having
    a servable floor.

    DERIVATION (I3-clean, not a magic number): scores `reranker.py`'s own
    bundled `_FP16_GATE_PAIRS[:6]` (the same relevant/decoy set §2's
    fp16-vs-fp32 gate uses — reused rather than shipping a second bundled
    set, exactly like `_cold_start_pairs` above) through a RAW provider for
    `reranker_model_id` specifically (`reranker._bootstrap_reranker_
    provider` — NOT `reranker.build_reranker_provider`, see that function's
    docstring for why: going through the precision self-check would recurse
    back into THIS function), then fits via the SAME `fit_threshold_fbeta`
    (Youden's-J / F-beta, recall-leaning) every other floor in this module
    uses — no separate/duplicated fitting logic. Scored through whichever
    model_id the caller asked about, so the result sits on that exact
    model's score scale (fp32 or fp16, whichever is the runtime reranker).

    COMPUTE CONSTRAINT (load-bearing, spec Section 7): computed ONCE per
    model_id (this cache) and NEVER involves the §6 torch-backed relevance
    judge — only jina (ONNX, via `CrossEncoderProvider`) scores the bundled
    pairs, so no torch import and no extra latency beyond this one-time cost
    ever touch the hot path. In practice this is very often a cache HIT on
    the underlying reranker PROVIDER too (not just this floor cache): the
    production call sites (`run_semantic_recall`, `_semantic_top_k`, the
    precision self-check) all resolve/construct their own reranker provider
    for this exact model_id via `reranker.build_reranker_provider` BEFORE
    ever calling `get_reranker_floor`, which already populated
    `reranker._provider_cache[model_id]` — `_bootstrap_reranker_provider`
    reads that same cache, so the ONNX session is typically already warm.

    NEVER PERSISTED: this is a transient, in-memory-only fallback — the
    caller (`MemoryStore.get_reranker_floor`) always checks the PERSISTED
    `reranker_floor_calibration` row first and only reaches this function on
    a miss, so a real corpus-derived floor (once the daily tick writes one)
    permanently supersedes this cache for that model_id with no way for a
    stale bootstrap value to shadow it.

    FAIL-SOFT (spec: the bootstrap computation itself must never crash a
    turn): any failure constructing the provider or fitting the threshold
    (a reranker load error, an empty/degenerate pairs list, ...) is caught,
    logged, and returns `None` — `get_reranker_floor` then degrades to the
    PRE-ruling contract (`None` -> caller falls back to lexical), the
    bootstrap's own last-resort failure path.
    """
    cached = _bootstrap_floor_cache.get(reranker_model_id)
    if cached is not None:
        return dict(cached)
    with _bootstrap_floor_cache_lock:
        cached = _bootstrap_floor_cache.get(reranker_model_id)
        if cached is not None:
            return dict(cached)
        try:
            from brain.memory.reranker import _bootstrap_reranker_provider

            provider = _bootstrap_reranker_provider(reranker_model_id)
            pairs = _cold_start_pairs(provider)
            beta = tunables.get_tunable("calibration.floor_fit_beta", FLOOR_FIT_BETA)
            floor = fit_threshold_fbeta(pairs, beta=beta)
        except Exception:  # noqa: BLE001 — fail-soft: must never break a recall/self-check
            logger.exception(
                "floor_calibration: bootstrap floor computation failed for %s -> "
                "get_reranker_floor degrades to the pre-ruling None/lexical-fallback contract",
                reranker_model_id,
            )
            return None
        result: dict[str, Any] = {
            "reranker_model_id": reranker_model_id,
            "floor": floor,
            "raw_fit_floor": floor,
            "sample_pairs": len(pairs),
            "is_cold_start": True,
            "updated_at": None,
        }
        _bootstrap_floor_cache[reranker_model_id] = result
        return dict(result)


def _reset_bootstrap_floor_cache() -> None:
    """Test-only: clear the cached bootstrap floor(s).

    Wired into `tests/conftest.py`'s autouse `_reset_reranker_provider_cache`
    fixture alongside `reranker._reset_reranker_provider_cache` et al. — same
    rationale: a test that calls the real `get_bootstrap_floor` (directly or
    via `MemoryStore.get_reranker_floor`) must not read or leak a value a
    prior/later test's call happened to cache for the same model_id.
    """
    with _bootstrap_floor_cache_lock:
        _bootstrap_floor_cache.clear()


# ---------------------------------------------------------------------------
# Orchestration — the ONE entry point the daily calibration tick calls.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorDerivationOutcome:
    """Result of one `derive_and_persist_floor` call."""

    accepted: bool
    """True iff a floor was ACTUALLY WRITTEN this call (a cold-start fit, or
    a real fit that cleared the stability gate). False means the stability
    gate tripped and held — the persisted floor is UNCHANGED. Callers
    (`_run_calibration_tick`) must only invalidate the reranker precision
    cache when this is True — nothing changed on a held cycle."""

    floor: float
    """The floor now in effect (freshly written if `accepted`, otherwise
    the prior persisted value, held unchanged)."""

    raw_fit_floor: float
    """This cycle's raw (pre-EMA) fitted threshold, for diagnostics/tests —
    distinct from `floor`, which is EMA-smoothed once past cold-start."""

    sample_pairs: int
    """How many labeled pairs backed THIS cycle's fit (cold-start pairs
    when `is_cold_start`, real accumulated `calibration_log` pairs
    otherwise)."""

    is_cold_start: bool
    """True while real accumulated labeled pairs stay below
    `FLOOR_FIT_MIN_LABELED_PAIRS` — an OUTCOME-based state (driven by how
    much usable labeled data has accumulated, never a day count)."""

    held_for_stability: bool
    """True iff a real fit was attempted but the stability gate held it."""


def derive_and_persist_floor(
    store: MemoryStore,
    reranker_model_id: str,
    *,
    reranker_provider: RerankerProvider | None = None,
    rng: np.random.Generator | None = None,
) -> FloorDerivationOutcome:
    """Derive this cycle's floor for `reranker_model_id` and persist it to
    `memories.db` (`MemoryStore.write_reranker_floor` — I1) if accepted.

    Cold-start vs real fit is decided by how many USABLE labeled pairs
    (`MemoryStore.labeled_calibration_pairs`, already filtered to
    `reranker_model_id` and to definitively "relevant"/"irrelevant" labels)
    have accumulated in `calibration_log` — an OUTCOME check, never a day
    count, per spec Section 7's cold-start bootstrap.

    `reranker_provider` / `rng` are test-injection points (mirrors
    `relevance_judge.label_calibration_sample`'s `judge`/`provider`
    parameters) — production leaves both `None`: `reranker_provider`
    defaults to `reranker.build_reranker_provider()` (only constructed when
    actually needed, i.e. only on a cold-start cycle) and `rng` defaults to
    a fresh `numpy.random.default_rng()`.
    """
    beta = tunables.get_tunable("calibration.floor_fit_beta", FLOOR_FIT_BETA)
    ema_window_days = tunables.get_tunable("calibration.floor_ema_window_days", FLOOR_EMA_WINDOW_DAYS)
    min_labeled_pairs = tunables.get_tunable(
        "calibration.floor_fit_min_labeled_pairs", FLOOR_FIT_MIN_LABELED_PAIRS
    )
    ci = tunables.get_tunable("calibration.floor_stability_ci", FLOOR_STABILITY_CI)
    iterations = tunables.get_tunable(
        "calibration.floor_stability_bootstrap_iterations", FLOOR_STABILITY_BOOTSTRAP_ITERATIONS
    )
    if rng is None:
        rng = np.random.default_rng()

    prior = store.get_reranker_floor(reranker_model_id)
    real_pairs = store.labeled_calibration_pairs(reranker_model_id)

    if len(real_pairs) >= min_labeled_pairs:
        raw_floor = fit_threshold_fbeta(real_pairs, beta=beta)
        # Only compare against a PRIOR REAL (non-cold-start) floor — a
        # cold-start floor was never a genuine EMA history point, so the
        # first real derivation always starts a fresh EMA/gate history
        # (outcome-based cold-start exit, spec Section 7).
        prior_ema = prior["floor"] if (prior is not None and not prior["is_cold_start"]) else None

        if not stability_gate_accepts(prior_ema, real_pairs, beta=beta, ci=ci, iterations=iterations, rng=rng):
            held_floor = prior["floor"] if prior is not None else raw_floor
            logger.info(
                "floor calibration: stability gate held the update for %s "
                "(raw_fit=%.4f, held floor=%.4f, sample_pairs=%d)",
                reranker_model_id, raw_floor, held_floor, len(real_pairs),
            )
            return FloorDerivationOutcome(
                accepted=False,
                floor=held_floor,
                raw_fit_floor=raw_floor,
                sample_pairs=len(real_pairs),
                is_cold_start=False,
                held_for_stability=True,
            )

        new_floor = ema_update(prior_ema, raw_floor, window_days=ema_window_days)
        store.write_reranker_floor(
            reranker_model_id,
            floor=new_floor,
            raw_fit_floor=raw_floor,
            sample_pairs=len(real_pairs),
            is_cold_start=False,
        )
        logger.info(
            "floor calibration: wrote real floor=%.4f (raw_fit=%.4f) for %s, sample_pairs=%d",
            new_floor, raw_floor, reranker_model_id, len(real_pairs),
        )
        return FloorDerivationOutcome(
            accepted=True,
            floor=new_floor,
            raw_fit_floor=raw_floor,
            sample_pairs=len(real_pairs),
            is_cold_start=False,
            held_for_stability=False,
        )

    # Cold-start: not enough real labeled data yet — serve/persist the
    # bundled-pairs bootstrap floor (recomputed fresh each cycle; no EMA
    # history to smooth against while still in cold-start).
    if reranker_provider is None:
        from brain.memory.reranker import build_reranker_provider

        reranker_provider = build_reranker_provider(store=store)
    cold_pairs = _cold_start_pairs(reranker_provider)
    raw_floor = fit_threshold_fbeta(cold_pairs, beta=beta)
    store.write_reranker_floor(
        reranker_model_id,
        floor=raw_floor,
        raw_fit_floor=raw_floor,
        sample_pairs=len(cold_pairs),
        is_cold_start=True,
    )
    logger.info(
        "floor calibration: cold-start floor=%.4f for %s (%d real labeled pairs, need %d to exit)",
        raw_floor, reranker_model_id, len(real_pairs), min_labeled_pairs,
    )
    return FloorDerivationOutcome(
        accepted=True,
        floor=raw_floor,
        raw_fit_floor=raw_floor,
        sample_pairs=len(cold_pairs),
        is_cold_start=True,
        held_for_stability=False,
    )
