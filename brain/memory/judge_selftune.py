"""Weekly judge self-tune — OFFLINE cadence + gate + hardware-tiered scaffold.

F2c inc2 (spec: `f2c-judge-selftune-spec.md` §1 [hardware-tiered mechanism],
§2 [cadence + gate], §7 [scope]). Builds ONLY the scaffold this increment
owns: the weekly cadence tick, the per-persona consumed-cursor marker, the
>handful gate, runtime RAM tier-detection, and the cgroup-aware OOM-safety
downgrade. The actual tuning (knob-refit fit / LoRA adapter retrain / full
fine-tune, the 2/3-1/3 eval split, champion/challenger + rollback) is NOT
built here — inc3+ fills in the `# TODO(F2c inc3+)` marker in
`_run_judge_selftune_tick` below. Entirely offline (I6): this tick runs only
from `supervisor.run_folded`'s weekly cadence block, never on the per-turn
recall path — no torch/sentence_transformers import here.

Cadence wiring mirrors `brain.engines.interest_sweep` EXACTLY (own cadence
file, own interval constant, own fault-isolated tick function that owns
neither cadence nor throttle — the caller in `supervisor.run_folded` does),
NOT the daily calibration tick's startup-catch-up shape: this is a weekly,
gated cadence with no catch-up-at-boot step (spec §2 "wiring" bullet).

See the durable Haiku-oracle note directly above `_run_judge_selftune_tick`
below (spec §6) for what this module's future training code converges
toward.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

import numpy as np

from brain import tunables
from brain.bridge.model_tier import MODEL_RELEVANCE_JUDGE
from brain.memory.reranker import _available_ram_headroom_bytes

logger = logging.getLogger(__name__)

JUDGE_TUNE_CADENCE_FILE = "judge_selftune_cadence.json"
JUDGE_TUNE_INTERVAL_HOURS = 168.0

# --- Tune-grade identifiers (spec §1 table) -------------------------------
# The weight-retrain GRADE selected by RAM tier and, if needed, downgraded
# by the cgroup-aware OOM-safety guard. Plain strings (not an enum) so they
# serialize directly into logging/marker state without a translation layer.
TUNE_GRADE_KNOB_REFIT = "knob_refit"  # weak box — always-safe floor
TUNE_GRADE_LORA = "lora"  # mid box
TUNE_GRADE_FULL_FT = "full_ft"  # beefy box
_TUNE_GRADE_ORDER = (TUNE_GRADE_KNOB_REFIT, TUNE_GRADE_LORA, TUNE_GRADE_FULL_FT)

# --- Tunables (I3/I7 — every threshold below is a registered tunable, never
# a bare hardcoded constant) --------------------------------------------

# ">a handful" gate (spec §2): the weekly tick fires only when MORE new
# non-None `haiku_label` POSITIONS (one Haiku decision each — spec §2's
# pinned counting unit, red-team fix F-2) than this have accumulated across
# UNCONSUMED `calibration_log` rows (`MemoryStore.count_new_haiku_
# decisions`). Provisional default — "a handful" ~5, so "more than a
# handful" starts just past it. Unlike the RAM-tier/footprint tunables
# below, this one isn't hardware-dependent, so it is a build-time judgment
# call rather than a dry-run-derived figure (spec's Open Reconfirmations
# list it alongside the RAM thresholds as a build-time derivation, not an
# owner fork).
JUDGE_TUNE_GATE_HANDFUL_DECISIONS: int = tunables.register(
    "judge_selftune.gate_handful_decisions", 20
)

# RAM tier thresholds (spec §1: "set by the dry-run... NOT hand-picked").
# ⚠ PROVISIONAL until F2c inc5's timed LoRA dry-run on deploy-class
# no-AVX2 hardware (spec Open Reconfirmations) actually measures them —
# these defaults are placeholders only. Spec §1's own data point: the dev
# VM (9.5 GB RAM / 512 MB swap) OOM-crashed merely LOADING three CPU models
# simultaneously, an early signal the knob-refit-only floor may extend
# further up than these placeholder cutoffs assume; inc5 confirms where
# LoRA actually becomes viable. Bytes (not GB) to match
# `_read_total_ram_bytes`'s / `_available_ram_headroom_bytes`'s unit.
JUDGE_TUNE_RAM_TIER_LORA_MIN_BYTES: float = tunables.register(
    "judge_selftune.ram_tier_lora_min_bytes", 16.0 * (1024**3)
)
JUDGE_TUNE_RAM_TIER_FULL_FT_MIN_BYTES: float = tunables.register(
    "judge_selftune.ram_tier_full_ft_min_bytes", 64.0 * (1024**3)
)

# Per-tune-grade memory FOOTPRINT estimates (spec §1 ⚠ cgroup-aware
# OOM-safety guard) — how much memory one epoch of that grade's
# weight-retrain needs. ⚠ PROVISIONAL, same inc5 dry-run populates real
# measured figures. Knob-refit's is a couple of interpretable scalar
# parameters (a deterministic Platt/threshold fit, no weight training at
# all), deliberately tiny so it fits on any box that can run the brain in
# the first place — this is what makes it the "always-safe floor" the spec
# calls for.
JUDGE_TUNE_FOOTPRINT_KNOB_REFIT_BYTES: float = tunables.register(
    "judge_selftune.footprint_knob_refit_bytes", 256.0 * (1024**2)
)
JUDGE_TUNE_FOOTPRINT_LORA_BYTES: float = tunables.register(
    "judge_selftune.footprint_lora_bytes", 6.0 * (1024**3)
)
JUDGE_TUNE_FOOTPRINT_FULL_FT_BYTES: float = tunables.register(
    "judge_selftune.footprint_full_ft_bytes", 24.0 * (1024**3)
)


def _read_total_ram_bytes() -> float | None:
    """Host TOTAL physical RAM (`/proc/meminfo`'s `MemTotal:` line) — the
    RAM-tier detector's raw signal (spec §1: "RAM is the proxy for overall
    capability... an 8 GB box implies a modest CPU, 32 GB a real CPU").

    Mirrors `brain.memory.reranker._proc_meminfo_available_bytes`'s SHAPE
    exactly (same fail-soft posture: `None` on non-Linux or any read/parse
    error, never a fabricated figure) but reads `MemTotal`, not
    `MemAvailable` — tier detection is about the box's overall CAPABILITY
    CLASS, not this instant's free memory (that is the separate OOM-safety
    guard's job, via `_available_ram_headroom_bytes`, reused unchanged from
    `reranker.py` below). A deliberate near-duplicate of the meminfo read,
    not a refactor-to-share: `reranker.py` is out of this increment's scope
    (it owns the per-turn recall-path memory read; this module owns the
    offline weekly-tick one), and the two reads answer different questions.

    Read at RUNTIME on every call, never cached (spec §1 "detect just prior
    to doing the thing... to account for any hardware upgrades in the past
    week") — callers must not memoize this across ticks.
    """
    if sys.platform != "linux":
        return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) * 1024.0
        return None
    except Exception:  # noqa: BLE001 — fail-soft: a tier probe must never crash the tick
        logger.exception("judge_selftune: /proc/meminfo MemTotal read failed")
        return None


def _select_tune_grade_by_ram(total_ram_bytes: float | None) -> str:
    """RAM tier -> tune-grade select (spec §1 table), evaluated against the
    CURRENT tunable thresholds (re-read every call, so a live tunables.json
    override takes effect on the very next tick).

    `total_ram_bytes is None` (detection failed / non-Linux) fails toward
    the always-safe floor — knob-refit — never toward a grade that might
    not fit; this mirrors `_available_ram_headroom_bytes`'s own "never
    assume you have more than you can prove" posture.
    """
    lora_min = tunables.get_tunable(
        "judge_selftune.ram_tier_lora_min_bytes", JUDGE_TUNE_RAM_TIER_LORA_MIN_BYTES
    )
    full_ft_min = tunables.get_tunable(
        "judge_selftune.ram_tier_full_ft_min_bytes", JUDGE_TUNE_RAM_TIER_FULL_FT_MIN_BYTES
    )
    if total_ram_bytes is None:
        return TUNE_GRADE_KNOB_REFIT
    if total_ram_bytes >= full_ft_min:
        return TUNE_GRADE_FULL_FT
    if total_ram_bytes >= lora_min:
        return TUNE_GRADE_LORA
    return TUNE_GRADE_KNOB_REFIT


def _downgrade_for_oom_safety(tune_grade: str, effective_headroom_bytes: float | None) -> str:
    """Cgroup-aware OOM-safety guard (spec §1 ⚠, Planning-adopted,
    load-bearing). Host-total RAM (`_select_tune_grade_by_ram` above) picks
    the CAPABILITY tier; this pairs it with a cgroup-aware EFFECTIVE-memory
    check — REUSING the corrected `reranker._available_ram_headroom_bytes`
    unchanged (no duplicate cgroup read: that helper already walks
    `/proc/self/cgroup` to resolve the CALLING PROCESS's own cgroup v2/v1
    limit, falling back to host-wide `/proc/meminfo` only when no cgroup
    limit applies) — and downgrades to the largest grade whose FOOTPRINT
    actually fits, so a cgroup-capped box (host-total RAM looks capable,
    but the container is capped far below it) never selects a tier that
    would OOM. Knob-refit's footprint is tiny by construction and always
    fits, so it is the floor this can never downgrade past.

    `effective_headroom_bytes is None` (every headroom source unavailable —
    the same fail-soft contract `_available_ram_headroom_bytes` documents:
    non-Linux, or every cgroup/meminfo read failed) means there is no
    signal to downgrade BY, but also no signal to trust the RAM-tier pick
    WITH — this fails toward the safe floor too, the same posture
    `_select_tune_grade_by_ram`'s own `None` branch takes, rather than
    treating "unknown" as "unlimited" (exactly the fp16-width OOM
    `_available_ram_headroom_bytes` itself was fixed to prevent).
    """
    footprints = {
        TUNE_GRADE_KNOB_REFIT: tunables.get_tunable(
            "judge_selftune.footprint_knob_refit_bytes", JUDGE_TUNE_FOOTPRINT_KNOB_REFIT_BYTES
        ),
        TUNE_GRADE_LORA: tunables.get_tunable(
            "judge_selftune.footprint_lora_bytes", JUDGE_TUNE_FOOTPRINT_LORA_BYTES
        ),
        TUNE_GRADE_FULL_FT: tunables.get_tunable(
            "judge_selftune.footprint_full_ft_bytes", JUDGE_TUNE_FOOTPRINT_FULL_FT_BYTES
        ),
    }
    if effective_headroom_bytes is None:
        return TUNE_GRADE_KNOB_REFIT
    start_idx = _TUNE_GRADE_ORDER.index(tune_grade)
    for idx in range(start_idx, -1, -1):
        grade = _TUNE_GRADE_ORDER[idx]
        if footprints[grade] <= effective_headroom_bytes:
            return grade
    return TUNE_GRADE_KNOB_REFIT  # last-resort: footprint estimates misconfigured; the floor still fits


# ---------------------------------------------------------------------------
# F2c inc3 — the knob-refit fit itself (spec §5: "fit the threshold/Platt
# slope+intercept on the (judge-raw-score, effective-label) pairs"). Weak-
# tier training code: deterministic, NO torch, NO randomness — mirrors
# `floor_calibration.fit_threshold_fbeta`'s posture (pure numpy, exact/
# reproducible, fails toward a safe degenerate answer rather than raising
# on a single-class sample) but fits TWO parameters (slope + intercept)
# instead of one (a bare threshold), because §5 explicitly steers toward
# Platt scaling as the DERIVED generalization of the existing fixed knob:
# `relevance_judge.label_for_score` already applies a sigmoid centered at
# 0.5 (equivalent to slope=1.0, intercept=0.0) — Platt scaling is exactly
# that same sigmoid family, re-centered and re-scaled by a proper
# maximum-likelihood fit instead of hand-picked at (1.0, 0.0). A bare
# threshold would have discarded the existing knob's shape rather than
# generalizing it (I3: derived, not hand-picked).
# ---------------------------------------------------------------------------


def _platt_targets(labels: list[str]) -> np.ndarray:
    """Lin-Lin-Weng (2007) regularized target probabilities for Platt
    scaling — the standard, published fix for Platt's original method's
    known divergence failure mode on perfectly separable data (a real
    possibility with a small weekly batch: e.g. every "relevant" pair
    scoring higher than every "irrelevant" one). Bounds each target
    strictly inside (0, 1) instead of at the raw 0/1 label, so the
    maximum-likelihood fit below cannot chase an unreachable exact
    boundary out to +/-infinity. A published, well-known recipe (not a
    hand-picked heuristic): `t_relevant = (n_pos + 1) / (n_pos + 2)`,
    `t_irrelevant = 1 / (n_neg + 2)`.
    """
    n_pos = sum(1 for label in labels if label == "relevant")
    n_neg = len(labels) - n_pos
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    return np.array([t_pos if label == "relevant" else t_neg for label in labels], dtype=np.float64)


def fit_platt_knob(pairs: list[tuple[float, str]]) -> tuple[float, float]:
    """Fit `(slope, intercept)` such that `sigmoid(slope * raw_score +
    intercept)` best separates `pairs` into their `"relevant"`/
    `"irrelevant"` effective labels (spec §5, AC2) — the knob-refit's core
    fit, consumed by `relevance_judge.label_for_score`'s `slope`/
    `intercept` params.

    DETERMINISTIC (AC2/AC8): Newton-Raphson with backtracking line search
    on the (Lin-Lin-Weng regularized-target) binary cross-entropy loss —
    always the SAME fixed starting point `(slope=0.0, intercept=0.0)`, a
    fixed iteration cap, no random initialization and no randomness
    anywhere in the loop, so the SAME `pairs` (in any order — the loss is a
    sum, order-invariant) always produces the SAME output. Pure numpy, no
    torch/sentence_transformers import anywhere in this module (AC8) — a
    couple of interpretable scalar parameters, the "always-safe floor"
    tier's whole point (spec §1).

    BEST-SEPARATING (AC2): Newton-Raphson on a strictly concave
    log-likelihood (the regularized targets above keep it strictly concave
    even for separable data) converges to the actual maximum-likelihood
    (slope, intercept) for this 1-D logistic-regression family — the
    best-separating member of the sigmoid family the existing fixed knob
    already belongs to, not an arbitrary or hand-tuned pick.

    Degenerate input (mirrors `fit_threshold_fbeta`'s posture): a
    single-class sample (every pair the SAME label — no separation to fit)
    returns the IDENTITY mapping `(1.0, 0.0)` — `label_for_score`'s exact
    fixed-default behavior — rather than fitting a meaningless direction
    from zero contrast. An EMPTY `pairs` raises `ValueError` (no data
    exists to train on at all) — `_run_judge_selftune_tick` treats that as
    a "no train" fault (spec AC3's consume=trained-on-never-fired-on
    contract): the calling tick's own try/except catches it, and the rows
    that would have fed this fit stay unconsumed for next week.

    MONOTONICITY GUARD (Opus cold-review, LOW): the judge's raw score is
    positively-correlated-with-relevance BY CONSTRUCTION (the existing
    fixed knob is `sigmoid(raw_score) >= 0.5` = `"relevant"` — higher
    score always means more relevant). A fitted `slope <= 0.0` would
    INVERT that direction (higher score -> `"irrelevant"`), which is never
    a valid refit of this judge — it only arises from bad/insufficient/
    anti-correlated weekly data. Such a fit falls back to the IDENTITY
    mapping `(1.0, 0.0)` too, same posture as the single-class case above,
    rather than applying an inverting knob.
    """
    if not pairs:
        raise ValueError("fit_platt_knob requires at least one labeled pair")
    scores = np.array([score for score, _ in pairs], dtype=np.float64)
    labels = [label for _, label in pairs]
    if not any(label == "relevant" for label in labels) or not any(
        label == "irrelevant" for label in labels
    ):
        # Single-class sample: no separation to fit — identity mapping,
        # same posture as fit_threshold_fbeta's degenerate-input fallback.
        return 1.0, 0.0

    targets = _platt_targets(labels)
    slope, intercept = 0.0, 0.0

    def _loss(s: float, i: float) -> float:
        z = s * scores + i
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-12, 1.0 - 1e-12)
        return float(-np.sum(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)))

    prev_loss = _loss(slope, intercept)
    for _ in range(100):
        z = slope * scores + intercept
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-12, 1.0 - 1e-12)
        grad_slope = float(np.sum((p - targets) * scores))
        grad_intercept = float(np.sum(p - targets))
        w = p * (1.0 - p)
        h_ss = float(np.sum(w * scores * scores)) + 1e-12
        h_si = float(np.sum(w * scores))
        h_ii = float(np.sum(w)) + 1e-12
        det = h_ss * h_ii - h_si * h_si
        if abs(det) < 1e-12:
            break
        d_slope = (h_ii * grad_slope - h_si * grad_intercept) / det
        d_intercept = (h_ss * grad_intercept - h_si * grad_slope) / det
        if abs(d_slope) < 1e-10 and abs(d_intercept) < 1e-10:
            break

        step = 1.0
        new_slope, new_intercept, new_loss = slope, intercept, prev_loss
        for _ in range(30):  # backtracking line search, fixed cap, deterministic
            new_slope = slope - step * d_slope
            new_intercept = intercept - step * d_intercept
            new_loss = _loss(new_slope, new_intercept)
            if new_loss <= prev_loss + 1e-12:
                break
            step *= 0.5

        if abs(prev_loss - new_loss) < 1e-12:
            slope, intercept = new_slope, new_intercept
            break
        slope, intercept, prev_loss = new_slope, new_intercept, new_loss

    if slope <= 0.0:
        # Monotonicity guard (Opus cold-review, LOW): the judge's raw score
        # is positively-correlated-with-relevance BY CONSTRUCTION (the
        # existing fixed knob is sigmoid(raw_score) >= 0.5 = "relevant" —
        # higher score always means more relevant). A non-positive fitted
        # slope would INVERT that direction (higher score -> "irrelevant"),
        # which is never a valid refit of this judge — it only happens on
        # bad/insufficient/contradictory data (e.g. anti-correlated pairs
        # from a noisy or too-small weekly batch). Fail toward the identity
        # mapping instead of applying an inverting knob, mirroring the
        # single-class fallback immediately above this fit loop.
        return 1.0, 0.0

    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# F2c (durable note, spec §6): Haiku is the effective relevance ORACLE this
# module's future training code converges the local judge toward. The
# TODO(F2c inc3+) placeholder below is where that training will run: it
# will fit the judge's score-to-label mapping (knob-refit / LoRA / full
# fine-tune) against the accumulated Haiku tie-break decisions logged in
# calibration_log, the same decisions relevance_judge.label_calibration_
# sample already treats as ground truth over the local judge's own
# provisional label at ambiguous positions (see that module's own durable
# note at its orchestration entry point, spec §6). If a relevance-quality
# problem shows up downstream later, this is one of the two places to
# look first: what the judge converges toward is Haiku's own labeling
# behavior, not an independently verified ground truth, so a systematic
# Haiku bias would propagate into the judge rather than being caught by
# it.
# ---------------------------------------------------------------------------


def _run_judge_selftune_tick(*, store, now: datetime) -> dict:
    """One weekly judge self-tune tick (F2c inc2 cadence/gate scaffold +
    inc3 knob-refit). Caller owns cadence + throttle — mirrors
    `interest_sweep.run_sweep_tick`'s contract exactly (a leaf engine call,
    not a supervisor `_run_X_tick` wrapper by naming convention alone; this
    function's own docstring states the same "caller owns cadence +
    throttle" contract the BUILD instructions name it by). Never raises.

    Counts UNCONSUMED non-None `haiku_label` positions across
    `calibration_log` (the >handful gate, spec §2/§3, pinned counting unit
    — `MemoryStore.count_new_haiku_decisions`) and, ONLY when the gate
    fires: runtime-detects the RAM tune-grade, applies the cgroup-aware
    OOM-safety downgrade, then runs the knob-refit (spec §5 — ALWAYS runs
    on every tier, weak/mid/beefy alike; `tune_grade` selects only whether
    a LoRA/full-FT weight-retrain ALSO runs, not built here — TODO(F2c
    inc5/6) below) — assembling `(judge_raw_score, effective_label)` pairs
    from `row_ids`'s rows (`MemoryStore.judge_knob_refit_pairs`), fitting
    `fit_platt_knob`, and persisting the result
    (`MemoryStore.write_judge_knob_calibration`) — and ONLY once that
    fit has completed successfully marks every row this tick scanned as
    consumed (`MemoryStore.mark_selftune_consumed`). Spec §2's "consume =
    trained-on, never fired-on" pinned contract: if the assembly or fit
    raises (e.g. `fit_platt_knob`'s `ValueError` on zero usable pairs — a
    real possibility for legacy rows logged before F2c inc1 added the raw-
    score column), this function's own try/except below catches it,
    `result["fired"]` stays `False`, and `row_ids` are left UNCONSUMED —
    next week's gate recounts them rather than silently losing that Haiku
    signal.

    F2c INC4a (built, elsewhere): loading a persona's fitted knob params
    INTO the live judge pass — `relevance_judge.label_calibration_sample`
    now reads what this tick persists (via `store.get_judge_knob_
    calibration`) and applies it to every `label_for_score` call in its
    pass, so a completed refit here takes effect on this persona's very
    next daily calibration tick.
    ⚠ F2c INC5/6, NOT built here: the LoRA (mid) / full fine-tune (beefy)
    weight-retrain tiers, the 2/3-train/1/3-test champion/challenger split
    + rollback (spec §4) — `tune_grade` below is computed and returned so
    a caller can see which grade WOULD run once those increments land, but
    only the knob-refit executes regardless of `tune_grade`'s value.

    Returns a caller-facing result dict (ignored by the current
    `supervisor.run_folded` wiring below, mirrors `run_sweep_tick`'s own
    ignored-return-value contract):
    ``{"fired": bool, "tune_grade": str | None, "new_decisions": int,
    "error": str | None}``.
    """
    result: dict = {"fired": False, "tune_grade": None, "new_decisions": 0, "error": None}
    try:
        gate_handful = tunables.get_tunable(
            "judge_selftune.gate_handful_decisions", JUDGE_TUNE_GATE_HANDFUL_DECISIONS
        )
        count, row_ids = store.count_new_haiku_decisions()
        result["new_decisions"] = count
        if count <= gate_handful:
            return result  # not yet MORE than a handful — no fire, nothing consumed

        total_ram = _read_total_ram_bytes()
        tune_grade = _select_tune_grade_by_ram(total_ram)
        effective_headroom = _available_ram_headroom_bytes()
        tune_grade = _downgrade_for_oom_safety(tune_grade, effective_headroom)

        # Knob-refit ALWAYS runs on the weak tier (spec §5) and is the
        # natural finishing step after any weight-retrain on mid/beefy
        # tiers too — inc5/6 will re-run it after their own weight update,
        # not built here. Any exception here (assembly or fit) propagates
        # to this function's own except block below, which is exactly
        # spec AC3's "fault before training completes" case: `row_ids`
        # stays unconsumed.
        pairs = store.judge_knob_refit_pairs(row_ids)
        slope, intercept = fit_platt_knob(pairs)
        store.write_judge_knob_calibration(MODEL_RELEVANCE_JUDGE, slope=slope, intercept=intercept)

        # Consume = trained-on, never fired-on (spec §2 pinned contract):
        # reaching this line means the knob-refit above ran to completion,
        # so `row_ids` — the exact set `judge_knob_refit_pairs` was built
        # from — is now safe to retire.
        store.mark_selftune_consumed(row_ids, consumed_at=now)
        store.write_judge_selftune_state(MODEL_RELEVANCE_JUDGE, last_trained_at=now)
        result["fired"] = True
        result["tune_grade"] = tune_grade
    except Exception as exc:  # noqa: BLE001 — fault-isolated, mirrors run_sweep_tick
        logger.warning("judge self-tune tick failed: %s", exc)
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
