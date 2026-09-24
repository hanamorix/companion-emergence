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
     UNCHANGED by the pre-flip revision (Change 1) below.
  2. `derive_and_persist_floor` — the ONE orchestration entry point the
     daily tick calls: reads the MOST RECENTLY COMPLETED DAY's labeled
     pairs off `calibration_log` (`MemoryStore.labeled_calibration_pairs`,
     day-scoped — see that method's docstring for the exact boundary) and,
     if that day cleared `FLOOR_FIT_MIN_LABELED_PAIRS`, fits and persists
     the raw threshold DIRECTLY — no smoothing layer sits between the fit
     and the persisted value. Below that threshold, a data-starvation
     BACKSTOP holds the previously-persisted floor unchanged (or, if no
     floor has ever been persisted yet, writes nothing at all) — see
     `FloorDerivationOutcome` and `derive_and_persist_floor`'s own
     docstring for the exact contract.

Pre-flip revision Change 1 (2026-09-23, "nimble floor"): the daily fit used
to run through two additional smoothing layers stacked on top of
`fit_threshold_fbeta` — an EMA blend against the previous floor
(`ema_update`) and a bootstrap-CI stability gate
(`stability_gate_accepts`) that could HOLD an update indefinitely against a
stale anchor. An independent code verify confirmed the stability gate had
NO recovery path: once it held, every later day was compared against the
same never-updated stale anchor, so a sustained large/fast shift (a model
swap, a precision flip, a big corpus change) locked the floor permanently.
Change 1 REMOVES both layers entirely (`ema_update`, `stability_gate_
accepts`, and the bootstrap-CI helper `_bootstrap_ci` they shared are gone
— see the changelog in the pre-flip revision spec) and re-points the fit
from the full multi-day retention-window pool to the single most recently
completed day (`MemoryStore.labeled_calibration_pairs`'s new day filter) —
a single day's volume (~800-2,500 labeled pairs, per the revision's own
data-volume verify) is comfortably past the noise-robustness threshold on
its own, which is what makes dropping the smoothing layers safe. The OLD
`derive_and_persist_floor` cold-start branch (which, below
`FLOOR_FIT_MIN_LABELED_PAIRS`, scored `reranker.py`'s bundled
`_FP16_GATE_PAIRS` to synthesize and unconditionally persist a floor) is
ALSO removed by Change 1, replaced by the data-starvation backstop
described above — see `derive_and_persist_floor`'s docstring for why this
means `derive_and_persist_floor` no longer needs a `reranker_provider`
argument, or an `rng` argument (no more bootstrap resampling either).

This module ALSO derives `store.CALIBRATION_LOG_RETENTION_WINDOW_DAYS`'s
value — see `RETENTION_WINDOW_DAYS_DEFAULT` and `derive_retention_window_
days` below. `store.py` imports that constant directly rather than
re-deriving it, so the tunable key (`calibration.retention_window_days`)
stays owned by `store.py`/`prune_calibration_log` (inc5's contract), only
its DEFAULT VALUE moves here. Change 1 shrinks this window's derivation
too (see that function's docstring): the fit no longer pools across days,
so retention no longer needs to cover `FLOOR_FIT_MIN_LABELED_PAIRS`'s
worst-case multi-day accumulation, only the current day plus a small
safety margin.

HOT-PATH no-persisted-floor bootstrap (F2a inc8, #250 §7 UPDATED, Roy
2026-09-18): `get_bootstrap_floor` below is the module's OTHER, still-
standing cold-start path — since Change 1 removes `derive_and_persist_
floor`'s own bundled-pairs cold-start branch (see above), this hot path is
now the ONLY place `reranker.py`'s bundled `_FP16_GATE_PAIRS` still gets
scored to synthesize a floor (via the shared `_cold_start_pairs` helper
below). It serves the true first-ever-install case, before any tick has
run: `get_bootstrap_floor` is called SYNCHRONOUSLY from
`MemoryStore.get_reranker_floor` whenever no `reranker_floor_calibration`
row exists yet for a model_id — i.e. it can fire on the per-turn hot path,
on the very FIRST no-row recall of the process. It is computed ONCE per
model_id and cached process-wide (never persisted to `memories.db` — a
transient, in-memory-only fallback that a real persisted row always
supersedes, see that function's docstring), and it deliberately builds its
own reranker provider via `reranker._bootstrap_reranker_provider` rather
than `reranker.build_reranker_provider` — the latter resolves its OWN
model_id from the `reranker.precision` tunable (ignoring any
caller-specified id), whereas this hot path must score the bundled pairs
through the EXACT `model_id` `get_reranker_floor`'s caller is asking about
(see `_bootstrap_reranker_provider`'s own docstring, current as of Change
2's removal of the fp16/fp32 precision self-check this used to also dodge
recursion through). Never touches the §6 torch-backed relevance judge —
jina (ONNX, via `reranker.CrossEncoderProvider`) is the only model
involved.

F2b §5b (#276 inc3, UNCHANGED by Change 1): both this hot-path bootstrap
AND (formerly) the now-removed `derive_and_persist_floor` cold-start
branch fit on the per-query ANCHOR-NORMALIZED score
(`raw - median(anchor_scores)`, `reranker.normalize_bundled_pairs_against_
anchors` — the FULL curated anchor pool, off the hot path), not the raw
reranker score, so the fit lands on the same scale the per-recall floor
gate (`reranker.normalize_against_anchors`) compares against. See
`_cold_start_pairs` below for the shared normalization mechanism.
"""

from __future__ import annotations

import logging
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

# Lower bound on labeled (reranker_score, label) PAIRS required, in the
# MOST RECENTLY COMPLETED DAY, before a real fit is attempted (below this,
# Change 1's data-starvation backstop holds the previously-persisted floor
# instead — see `derive_and_persist_floor`). The spec's own citation: a
# cutoff fit is "robust at hundreds of pairs" — 200 is the upper end of
# "hundreds," chosen for a comfortable margin over the bare minimum rather
# than the smallest number that technically qualifies. NOT reopened/re-
# derived by the pre-flip revision (Change 1's Open Reconfirmations
# explicitly leave this one alone) — reused as-is from the original F2a
# build.
FLOOR_FIT_MIN_LABELED_PAIRS: int = tunables.register("calibration.floor_fit_min_labeled_pairs", 200)

# Pre-flip revision Change 1, retention-window Open Reconfirmation: the
# safety buffer (in DAYS) added on top of the 2 calendar days
# (`_RETENTION_TODAY_AND_YESTERDAY_DAYS` below) the day-scoped fit
# structurally needs to always have "yesterday" (the most recently
# completed day it actually reads — `MemoryStore.labeled_calibration_
# pairs`) still inside the retention window. One buffer day covers a
# late-running tick or a timezone-straddling day boundary (the tick fires
# just after local midnight but before/after the `day_bucket` column's own
# UTC-anchored rollover) without collapsing retention to the bare 2-day
# minimum the spec's own hard rule forbids ("must retain at least
# today+yesterday+buffer, NOT collapse to <2 days"). Operator-overridable
# like every other tunable here, but 1.0 is not an arbitrary pick — see
# `derive_retention_window_days`'s docstring for the full derivation.
FLOOR_RETENTION_SAFETY_BUFFER_DAYS: float = tunables.register(
    "calibration.floor_retention_safety_buffer_days", 1.0
)

# Structural (not operator-tunable) minimum: TODAY's in-progress
# `day_bucket` plus YESTERDAY's (the most recently COMPLETED day the fit
# actually reads — see `MemoryStore.labeled_calibration_pairs`). This is
# not a judgment call the way the safety buffer above is — retention
# collapsing below this would let the exact day the fit needs age out of
# the window before the fit ever reads it, which is a correctness bug, not
# a tuning choice — so it stays a plain named constant rather than a
# tunable an operator could override into breakage.
_RETENTION_TODAY_AND_YESTERDAY_DAYS: float = 2.0


def derive_retention_window_days(*, safety_buffer_days: float | None = None) -> float:
    """`calibration_log` retention window (spec Section 5 / acceptance 5b),
    Change 1's SHRUNK derivation (pre-flip revision, Open Reconfirmation
    "the retention-window shrink value").

    Before Change 1, the daily fit pooled every labeled pair across the
    FULL retention window, so the window had to be sized to cover
    `FLOOR_FIT_MIN_LABELED_PAIRS`' worst-case multi-day accumulation (inc7:
    `max(ceil(200/worst_case_daily_yield), FLOOR_EMA_WINDOW_DAYS)`, both
    inputs now gone — the EMA window along with `ema_update` itself, the
    pooled read along with the day filter that replaced it). Change 1's fit
    reads ONLY the single most recently completed day
    (`MemoryStore.labeled_calibration_pairs`), so retention no longer needs
    to cover an accumulation window at all — only enough days for that one
    day's data to still be sitting in the table when the fit goes looking
    for it:

      `_RETENTION_TODAY_AND_YESTERDAY_DAYS` (2.0, structural — today's
      in-progress bucket plus yesterday's, the day the fit reads) +
      `safety_buffer_days` (`FLOOR_RETENTION_SAFETY_BUFFER_DAYS`, 1.0 by
      default, operator-tunable — a late-running tick or a timezone-
      straddling day boundary).

    Default total: 3.0 days — small, but never below the spec's own hard
    floor of "today+yesterday+buffer, not <2 days" (I3/I7: a derived, named
    constant, not a re-typed magic number; changing the tunable changes the
    result, proving it is a live derivation).
    """
    if safety_buffer_days is None:
        safety_buffer_days = tunables.get_tunable(
            "calibration.floor_retention_safety_buffer_days", FLOOR_RETENTION_SAFETY_BUFFER_DAYS
        )
    return float(_RETENTION_TODAY_AND_YESTERDAY_DAYS + safety_buffer_days)


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
# Cold-start bootstrap fit — reuses reranker.py's bundled pairs. Pre-flip
# revision Change 1: the ONLY caller left is `get_bootstrap_floor` below —
# `derive_and_persist_floor`'s own cold-start branch (the daily-tick path)
# is removed; see the module docstring.
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

    F2b §5b (#276 inc3): each pair's score is now the per-query
    ANCHOR-NORMALIZED score (`raw - median(anchor_scores)`,
    `reranker.normalize_bundled_pairs_against_anchors` — the SAME shared
    `_median_normalize` core `normalize_against_anchors` (the per-recall
    gate) uses, scored against the FULL curated anchor pool rather than the
    gate's latency-limited `k`-subset — see that function's docstring and
    the spec's §5b "FULL pool P, NOT the per-recall k" requirement), not the
    raw reranker score. This is what makes this cold-start path land on the
    same normalized scale the per-recall floor gate compares against
    (`normalize_against_anchors` in `semantic_recall.py` /
    `search_memories.py`) — before the original §5b fix, this source fit a
    raw-scale floor while the gate compared a normalized score against it,
    an under-abstention scale mismatch.

    Pre-flip revision Change 1: the ONLY caller of this helper is now
    `get_bootstrap_floor` below — `derive_and_persist_floor`'s own
    cold-start branch (the daily-tick path this helper used to ALSO back)
    is removed; see the module docstring.
    """
    from brain.memory.reranker import _FP16_GATE_PAIRS, normalize_bundled_pairs_against_anchors

    labeled_slice = _FP16_GATE_PAIRS[:6]
    labels = ["relevant"] * 3 + ["irrelevant"] * 3
    normalized_scores = normalize_bundled_pairs_against_anchors(reranker_provider, labeled_slice)
    return list(zip((float(s) for s in normalized_scores), labels, strict=True))


# ---------------------------------------------------------------------------
# Hot-path bootstrap floor (F2a inc8, #250 §7 UPDATED, Roy 2026-09-18) —
# the DEFAULT `get_reranker_floor` serves when NO persisted row exists yet,
# so semantic recall's existence is decoupled from the daily tick ever
# having fired. Process-wide cache, keyed by model_id, computed ONCE.
# ---------------------------------------------------------------------------

# model_id -> the bootstrap floor dict last derived for it (same shape as
# `MemoryStore.get_reranker_floor`'s persisted-row dict). Process-wide,
# mirrors `reranker.py`'s `_provider_cache`/`_latency_cache` pattern
# (pre-flip revision Change 2 removed the sibling `_precision_decision_
# cache` this comment used to also list, along with the precision
# self-check it backed): computed once per model_id, reused by every later
# caller in the process, reset only by tests (`_reset_bootstrap_floor_
# cache`, wired into `tests/conftest.py`'s autouse fixture alongside the
# reranker module's own resets).
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
    bundled `_FP16_GATE_PAIRS[:6]` (the small relevant/decoy set that used
    to also back the now-removed fp16-vs-fp32 self-check, F2a spec §2 —
    reused rather than shipping a second bundled set, exactly like
    `_cold_start_pairs` above) through a provider built for `reranker_
    model_id` SPECIFICALLY (`reranker._bootstrap_reranker_provider` — NOT
    `reranker.build_reranker_provider`, see that function's docstring for
    why: `build_reranker_provider` resolves its OWN model_id from the
    `reranker.precision` tunable, ignoring any caller-specified id, so it
    could resolve to a DIFFERENT model_id than the one THIS function was
    actually asked about), then fits via the SAME `fit_threshold_fbeta`
    (Youden's-J / F-beta, recall-leaning) every other floor in this module
    uses — no separate/duplicated fitting logic. Scored through whichever
    model_id the caller asked about, so the result sits on that exact
    model's score scale (fp32 or fp16, whichever is the runtime reranker) —
    F2b §5b (#276 inc3): via `_cold_start_pairs`, this is now the
    NORMALIZED scale (`raw - median(anchor_scores)`, scored against the
    FULL curated anchor pool), matching what the per-recall gate compares
    against.

    COMPUTE CONSTRAINT (load-bearing, spec Section 7): computed ONCE per
    model_id (this cache) and NEVER involves the §6 torch-backed relevance
    judge — only jina (ONNX, via `CrossEncoderProvider`) scores the bundled
    pairs, so no torch import and no extra latency beyond this one-time cost
    ever touch the hot path. In practice this is very often a cache HIT on
    the underlying reranker PROVIDER too (not just this floor cache): the
    production call sites (`run_semantic_recall`, `_semantic_top_k`) all
    resolve/construct their own reranker provider for this exact model_id
    via `reranker.build_reranker_provider` BEFORE ever calling `get_
    reranker_floor`, which already populated
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
    """Result of one `derive_and_persist_floor` call.

    Pre-flip revision Change 1 ("nimble floor"): this cycle is now always
    exactly one of two outcomes — a direct raw fit over the most recently
    completed day's labeled pairs (`accepted=True`), or a data-starvation
    BACKSTOP (`accepted=False`, `held_for_data_starvation=True`) when that
    day had fewer than `FLOOR_FIT_MIN_LABELED_PAIRS` labeled pairs. There is
    no EMA blend and no stability gate anymore — an accepted cycle's
    `floor` IS its `raw_fit_floor`, unchanged.
    """

    accepted: bool
    """True iff a floor was ACTUALLY WRITTEN this call (the most recently
    completed day cleared `FLOOR_FIT_MIN_LABELED_PAIRS` and its raw fit was
    persisted directly). False means the data-starvation backstop held —
    either the previously-persisted floor stands unchanged
    (`floor`/`raw_fit_floor` carry its values through), or — the no-prior-
    row edge case — NOTHING was ever persisted and nothing was written this
    cycle either (`floor`/`raw_fit_floor` are `None`)."""

    floor: float | None
    """The floor now in effect after this cycle: the freshly fit value when
    `accepted`; the previously-persisted floor, carried through unchanged,
    when the backstop held against an EXISTING prior row; or `None` when
    the backstop held with NO prior row at all (a fresh deploy still inside
    Change 1's data-starvation ramp — nothing is in effect from THIS
    module for that case; per-turn recall still gets a served floor from
    the separate, transient `get_bootstrap_floor` hot path — see
    `MemoryStore.get_reranker_floor`)."""

    raw_fit_floor: float | None
    """This cycle's raw fitted threshold when a fit actually ran (always
    equal to `floor` when `accepted` — no smoothing sits between them
    anymore); the prior row's own `raw_fit_floor`, carried through
    unchanged, when the backstop held against an existing prior; `None`
    when there is no prior and no fit ran this cycle."""

    sample_pairs: int
    """How many day-scoped labeled pairs (the most recently completed
    day's, for `reranker_model_id` — `MemoryStore.labeled_calibration_
    pairs`) THIS cycle read, even on a held/no-write cycle, so a
    data-starvation hold stays visible in logs/diagnostics."""

    is_cold_start: bool
    """Always False now. Pre-flip revision Change 1 removes the bundled-
    pairs cold-start branch this field used to distinguish (a fit over
    synthetic bundled pairs, unconditionally persisted) — `derive_and_
    persist_floor` only ever fits real accumulated `calibration_log` data
    now, or holds. Kept as a field (rather than removed) because the
    persisted `reranker_floor_calibration.is_cold_start` column and its
    downstream observability readers (`semantic_recall.py`,
    `search_memories.py`) still exist; this module just never writes
    `True` through it anymore. `floor_calibration.get_bootstrap_floor` is
    a wholly SEPARATE, still-standing mechanism that DOES still report
    `is_cold_start=True` on its own transient, never-persisted dict —
    unrelated to this field."""

    held_for_data_starvation: bool
    """True iff the most recently completed day's usable labeled-pair
    count fell below `FLOOR_FIT_MIN_LABELED_PAIRS` this cycle (whether or
    not a prior row existed to hold) — replaces the removed stability
    gate's `held_for_stability`. Holds carry NO memory: the very next day
    that clears the threshold fits fresh from that day alone, with no
    dependence on however many prior holds preceded it (unlike the removed
    stability gate, which compared every later day against the same
    never-updated stale anchor once it first held)."""


def derive_and_persist_floor(store: MemoryStore, reranker_model_id: str) -> FloorDerivationOutcome:
    """Derive this cycle's floor for `reranker_model_id` from the MOST
    RECENTLY COMPLETED DAY's labeled pairs and persist it to `memories.db`
    (`MemoryStore.write_reranker_floor` — I1) directly, with no smoothing
    (pre-flip revision Change 1, "nimble floor").

    THE MECHANISM (spec Change 1, Roy's ruling restated: "each day, fit
    from the most recent day's hits/misses, use directly, no EMA, no
    gate"):
      1. Read `MemoryStore.labeled_calibration_pairs(reranker_model_id)` —
         already day-scoped to the most recently completed day (the
         MAX `day_bucket` with any usable labeled row for this model_id;
         see that method's own docstring for the exact boundary this
         module reuses rather than re-deriving).
      2. If that day has `>= FLOOR_FIT_MIN_LABELED_PAIRS` usable pairs: fit
         `fit_threshold_fbeta` and persist the RAW result DIRECTLY —
         `floor == raw_fit_floor`, always, for an accepted cycle. No EMA
         blend, no stability-gate comparison against history.
      3. Otherwise (the data-starvation BACKSTOP — not a smoothing layer,
         a genuine "not enough data to fit anything today" floor):
           - a prior persisted row EXISTS: HOLD it — the persisted row is
             left byte-for-byte untouched, nothing is (re)written this
             cycle, and this cycle's outcome reports the prior's own
             `floor`/`raw_fit_floor` values (unchanged) so callers/logs
             can see what is still in effect.
           - NO prior persisted row exists (the no-prior-row edge case —
             a fresh deploy still inside the ramp): WRITE NOTHING. Do not
             crash on the `None` prior, do not synthesize a floor just to
             have something to persist — `floor`/`raw_fit_floor` in the
             returned outcome are `None`. Recall stays served meanwhile by
             the separate hot-path `get_bootstrap_floor`
             (`MemoryStore.get_reranker_floor`'s own no-persisted-row
             fallback, untouched by this revision). Once some day
             accumulates `>= FLOOR_FIT_MIN_LABELED_PAIRS` real pairs, this
             function fits and persists normally from that point on, per
             step 2 above.
      Holds carry NO memory across cycles either way: the very next day
      that clears the threshold fits FRESH from that day alone, with no
      damping or partial-move from however many holds preceded it — see
      `FloorDerivationOutcome.held_for_data_starvation`'s docstring.

    Takes no `reranker_provider`/`rng` injection points anymore (mirrors
    `relevance_judge.label_calibration_sample`'s old `judge`/`provider`
    shape less now, on purpose): the removed cold-start branch was the
    only thing here that ever needed a live reranker provider to score
    bundled pairs, and the removed stability gate was the only thing that
    ever needed a bootstrap RNG. Neither exists in this function anymore —
    see the module docstring's "What is removed."
    """
    beta = tunables.get_tunable("calibration.floor_fit_beta", FLOOR_FIT_BETA)
    min_labeled_pairs = tunables.get_tunable(
        "calibration.floor_fit_min_labeled_pairs", FLOOR_FIT_MIN_LABELED_PAIRS
    )

    real_pairs = store.labeled_calibration_pairs(reranker_model_id)

    if len(real_pairs) >= min_labeled_pairs:
        raw_floor = fit_threshold_fbeta(real_pairs, beta=beta)
        store.write_reranker_floor(
            reranker_model_id,
            floor=raw_floor,
            raw_fit_floor=raw_floor,
            sample_pairs=len(real_pairs),
            is_cold_start=False,
        )
        logger.info(
            "floor calibration: wrote raw floor=%.4f for %s, sample_pairs=%d (no EMA, no gate)",
            raw_floor, reranker_model_id, len(real_pairs),
        )
        return FloorDerivationOutcome(
            accepted=True,
            floor=raw_floor,
            raw_fit_floor=raw_floor,
            sample_pairs=len(real_pairs),
            is_cold_start=False,
            held_for_data_starvation=False,
        )

    # Data-starvation backstop: the most recently completed day did not
    # clear FLOOR_FIT_MIN_LABELED_PAIRS. Read the PERSISTED-ONLY prior row
    # (never the transient bootstrap `get_reranker_floor` would otherwise
    # serve on a miss) so the no-prior-row edge case below is judged on
    # whether a REAL row exists, not on whether SOME floor is servable.
    prior = store.get_persisted_reranker_floor(reranker_model_id)
    if prior is not None:
        logger.info(
            "floor calibration: data-starvation backstop held the floor for %s "
            "(sample_pairs=%d < min=%d; holding prior floor=%.4f unchanged, no refit attempted)",
            reranker_model_id, len(real_pairs), min_labeled_pairs, prior["floor"],
        )
        return FloorDerivationOutcome(
            accepted=False,
            floor=prior["floor"],
            raw_fit_floor=prior["raw_fit_floor"],
            sample_pairs=len(real_pairs),
            is_cold_start=False,
            held_for_data_starvation=True,
        )

    logger.info(
        "floor calibration: data-starvation backstop, no prior floor row for %s "
        "(sample_pairs=%d < min=%d) — writing nothing; recall stays served by "
        "get_bootstrap_floor in the meantime",
        reranker_model_id, len(real_pairs), min_labeled_pairs,
    )
    return FloorDerivationOutcome(
        accepted=False,
        floor=None,
        raw_fit_floor=None,
        sample_pairs=len(real_pairs),
        is_cold_start=False,
        held_for_data_starvation=True,
    )
