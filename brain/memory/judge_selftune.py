"""Weekly judge self-tune — OFFLINE cadence + gate + hardware-tiered scaffold.

F2c inc2 (spec: `f2c-judge-selftune-spec.md` §1 [hardware-tiered mechanism],
§2 [cadence + gate], §7 [scope]); the tuning itself landed inc3–inc6. This
module owns: the weekly cadence tick, the per-persona consumed-cursor marker,
the >handful gate, runtime RAM tier-detection, and the cgroup-aware
OOM-safety downgrade — plus the tune dispatch: the weak-tier Platt knob-refit
(inc3, `fit_platt_knob`/`_weak_knob_refit_and_consume`, deterministic, no
torch) and the mid/beefy weight-retrain lifecycle (inc5b-2 LoRA + inc6 full
fine-tune, `_run_weight_retrain`; since inc7 ONE model lineage per persona:
every update is applied on top of the persona's current model, the tier picks
only this week's method via `_select_retrain`, and the persona keeps one plain
checkpoint behind one pointer; the 2/3-1/3 split + champion/challenger live in
the torch-scoped sibling modules `judge_eval`/`judge_lora`/`judge_full_ft`).
Entirely offline (I6): this tick runs only from `supervisor.run_folded`'s
weekly cadence block, never on the per-turn recall
path — no top-level torch/sentence_transformers import here (the weight-retrain
tiers import them LAZILY inside the sibling modules).

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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

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


def _weak_knob_refit_and_consume(store, now: datetime, row_ids: list[int], knob_key: str) -> None:
    """The weak-tier / degenerate-floor / revert knob-refit (spec §5, inc3
    path): fit the Platt knob on the FULL LOGGED `(raw_score,
    effective_label)` set (`judge_knob_refit_pairs`, doc-AGNOSTIC — trains on
    every row incl. legacy doc-absent ones), persist it under `knob_key` (the
    key of the model SERVING this tick, `relevance_judge.judge_knob_key`;
    F2c inc9 — the logged scores came from that model), then consume the
    FULL firing `row_ids` (every row was trained on by this doc-agnostic fit,
    so all are retired — Planning consume-semantics 2B). Any fault (e.g.
    `fit_platt_knob`'s `ValueError` on zero usable pairs) propagates to the
    caller's try/except, leaving `row_ids` UNCONSUMED (fail-safe)."""
    pairs = store.judge_knob_refit_pairs(row_ids)
    slope, intercept = fit_platt_knob(pairs)
    store.write_judge_knob_calibration(knob_key, slope=slope, intercept=intercept)
    store.write_judge_selftune_state(MODEL_RELEVANCE_JUDGE, last_trained_at=now)
    store.mark_selftune_consumed(row_ids, consumed_at=now)


def _training_start(current: Path | None) -> str:
    """What this week's weight update is applied ON TOP OF (spec §1, AC12;
    F2c inc7): the persona's CURRENT model — its own plain checkpoint, or the
    base judge's model id when it has never been weight-tuned. The single home
    of this rule. A REVERTED week never moves the pointer, so the week after a
    rejected challenger starts from the model that never got the rejected
    adjustment [OWNER 2026-09-25]."""
    return str(current) if current is not None else MODEL_RELEVANCE_JUDGE


def _keep_after_accept(new_name: str) -> list[str]:
    """Which stored checkpoint dirs survive an ACCEPTED swap (F2c inc7 ruling
    Q1; spec §5 knob-first order, inc9): only the new one. The previous
    checkpoint (and, separately, its knob row) is deleted right after the
    swap that commits the new one; if the OS refuses (files still
    open/memory-mapped, Windows — I13), the failure is logged, not raised,
    and `judge_lora.reap_unreferenced` retries at the start of the next tick."""
    return [new_name]


def _reap_orphan_knob_rows(store, persona_dir: Path) -> int:
    """Delete every tuned-checkpoint knob row (`<base id>@<name>`) except the
    one for the checkpoint the persona's `current` pointer names (F2c inc9,
    spec §5: "a knob row left by a staged checkpoint that never got swapped in
    ... is orphaned and cleaned up on the next tick"; also a superseded
    checkpoint's row whose post-swap delete faulted). Runs at the start of
    every weekly tick, beside `judge_lora.reap_unreferenced`.

    Never deletes the plain base-id row (the base judge's own knob) or a row
    of any other model id. SAFETY, mirroring `reap_unreferenced`: the pointer
    is read raw; if it exists but cannot be read, or is empty, nothing is
    reaped this tick. An absent pointer means no tuned checkpoint serves, so
    every `<base id>@...` row is an orphan. Returns the rows deleted."""
    from brain.memory import judge_lora
    from brain.memory.relevance_judge import judge_knob_key

    pointer = judge_lora.pointer_file(judge_lora.champion_dir(persona_dir))
    keep: str | None = None
    try:
        if pointer.exists():
            name = pointer.read_text(encoding="utf-8").strip()
            if not name:
                logger.warning("judge self-tune: %s is empty; skipping the knob-row reap", pointer)
                return 0
            keep = judge_knob_key(MODEL_RELEVANCE_JUDGE, name)
    except OSError:
        logger.warning("judge self-tune: cannot read %s; skipping the knob-row reap", pointer, exc_info=True)
        return 0
    prefix = f"{MODEL_RELEVANCE_JUDGE}@"
    deleted = 0
    for key in store.list_judge_knob_keys():
        if key.startswith(prefix) and key != keep:
            deleted += store.delete_judge_knob_calibration(key)
    return deleted


def _select_retrain(tune_grade: str, start: str, staged: Path) -> Callable[..., Callable[[tuple[str, str]], str]]:
    """This week's update METHOD (spec §1 table): LORA → a LoRA trained on
    `start` and merged in memory (`judge_lora.build_lora_retrain_fn`); FULL_FT
    → a full fine-tune continued from `start` (`judge_full_ft.
    build_full_ft_retrain_fn`). Both save a PLAIN checkpoint to `staged`.
    Module functions are attribute-referenced at CALL time so a test's
    monkeypatch is honored."""
    from brain.memory import judge_full_ft, judge_lora

    if tune_grade == TUNE_GRADE_FULL_FT:
        return judge_full_ft.build_full_ft_retrain_fn(start, save_full_dir=staged)
    return judge_lora.build_lora_retrain_fn(
        start,
        target_modules=judge_lora.BGE_RERANKER_LORA_TARGET_MODULES,
        modules_to_save=judge_lora.BGE_RERANKER_LORA_MODULES_TO_SAVE,
        save_dir=staged,
    )


def _run_weight_retrain(
    store, now: datetime, persona_dir: Path, row_ids: list[int], tune_grade: str, info: dict
) -> tuple[bool | None, bool]:
    """The LoRA/full-FT weight-retrain lifecycle (spec §4/§5, AC5/6/12/13) for
    a mid/beefy ACCEPT-or-REVERT tick. ONE model lineage per persona (F2c
    inc7): the challenger is trained ON TOP OF the persona's current model
    (`_training_start`), the champion it must beat is that same current model,
    and an accepted challenger replaces it in the ONE per-persona store. The
    tier (`tune_grade`) only picks this week's method (`_select_retrain`).
    Returns `(accepted, checkpoint_persisted)` — `accepted` is `None` when the
    DEGENERATE FLOOR fell through to a plain knob-refit. `info` receives the
    H4 observability fields (`current`, `start`). Raises on any fault (the
    caller leaves `row_ids` unconsumed).

    DURABLE HAIKU-ORACLE NOTE (spec §6): the champion/challenger below scores
    both judges against the accumulated HAIKU tie-break labels as the oracle,
    and the accepted challenger's knob is re-fit on Haiku-derived effective
    labels — Haiku is the effective relevance ORACLE this weight-retrain
    converges the local judge toward, NOT an independently verified ground
    truth. A systematic Haiku bias would propagate into which judge this
    accepts and how its knob is centered; if a relevance-quality problem shows
    up downstream, this dispatch is one of the two places to look (alongside
    the knob-refit's training data).

    ORDER + crash-safety (spec §4; §5 knob-first order, F2c inc9): every knob
    row is keyed to the model it was fit on (`relevance_judge.judge_knob_key`),
    and the serve path reads only the served model's key, so the pointer and
    the knob cannot diverge.
      - ACCEPT: re-score the knob on the STAGED model → write it under the
        STAGED checkpoint's key → swap the pointer (the single COMMIT) →
        delete the previous checkpoint and its knob row (best-effort) →
        persist state → CONSUME the doc-HAVING subset (2B) LAST. No rollback
        step exists: a fault before the swap leaves the previous checkpoint
        serving with its own knob (the staged dir is discarded; its knob row,
        if written, is an orphan reaped next tick by `_reap_orphan_knob_rows`);
        a fault after the swap leaves the new checkpoint serving with its own
        knob (rows unconsumed, re-eligible; leftovers reaped next tick).
      - REVERT: the pointer is never swapped and no knob row is written for
        the rejected staged checkpoint; reap the discarded staged dir; knob
        from LOGGED scores (inc3 path) under the SERVING model's key; CONSUME
        the FULL `row_ids` LAST.
    """
    from brain.memory import judge_eval, judge_full_ft, judge_lora, relevance_judge
    from brain.memory.relevance_judge import judge_knob_key

    resolved_min_n = tunables.get_tunable(
        "judge_selftune.eval_min_test_n", judge_eval.JUDGE_EVAL_MIN_TEST_N_DEFAULT
    )

    current = judge_lora.resolve_current_checkpoint(persona_dir)
    start = _training_start(current)
    info["current"] = current.name if current is not None else "base"
    info["start"] = current.name if current is not None else "base"
    # The knob key of the model serving this tick (floor / revert write here).
    served_key = judge_knob_key(MODEL_RELEVANCE_JUDGE, current)

    triples = store.judge_lora_training_triples(row_ids)
    # DEGENERATE FLOOR (spec §4, finding 7 — reuse the pinned `eval_min_test_n`
    # tunable, no second threshold): too few Haiku (query, doc, label) triples
    # to split + evaluate → keep the champion, run the always-on weak knob-refit
    # on the logged pairs, consume the full set. No champion/challenger.
    if len(triples) < resolved_min_n:
        _weak_knob_refit_and_consume(store, now, row_ids, served_key)
        return (None, False)

    root = judge_lora.champion_dir(persona_dir)
    # The pointer's raw name (even an unusable legacy one): after the swap its
    # knob row is deleted; `current` above is what it resolves to.
    prior_name = judge_lora.read_pointer_name(root)

    # Champion = the model currently SERVING this persona (spec §1: serving is
    # tier-independent): its own checkpoint, else the base judge.
    if current is not None:
        champ_scorer = judge_full_ft.load_full_scorer(current)

        def champion(item):
            label, _amb = relevance_judge.label_for_score(float(champ_scorer(item)))
            return label
    else:
        base_judge = relevance_judge.build_judge_provider()

        def champion(item):
            label, _amb = relevance_judge.label_for_score(
                float(base_judge.score(item[0], item[1]))
            )
            return label

    train, test = judge_eval.split_train_test(triples)
    train_items = [(q, d, label) for (q, d, label) in train]
    test_items = [((q, d), label) for (q, d, label) in test]

    staged = judge_lora.staged_adapter_path(root)
    retrain_fn = _select_retrain(tune_grade, start, staged)

    committed = False
    try:
        cc = judge_eval.run_champion_challenger(
            champion=champion,
            retrain_fn=retrain_fn,
            train_items=train_items,
            test_items=test_items,
            # No-op rollback: the champion is never mutated during eval
            # (retrain writes to the fresh staged subdir); crash-safety is
            # owned here via the knob-first order + atomic pointer swap.
            rollback=judge_eval.RollbackHandle(),
        )
        if cc.accepted:
            # RE-SCORE the knob on the TUNED model (§5 pin): forward-pass the
            # doc-HAVING re-score set through the staged checkpoint, loaded by
            # the same loader that will serve it. Raises if empty → handler,
            # rows unconsumed.
            scorer = judge_full_ft.load_full_scorer(staged)
            rescore = store.judge_knob_refit_rescore_items(row_ids)
            pairs = [(float(scorer((q, d))), label) for (q, d, label) in rescore]
            slope, intercept = fit_platt_knob(pairs)

            # Knob FIRST, keyed to the staged checkpoint (spec §5, inc9): it
            # exists before the pointer can name that checkpoint.
            store.write_judge_knob_calibration(
                judge_knob_key(MODEL_RELEVANCE_JUDGE, staged), slope=slope, intercept=intercept
            )
            judge_lora.swap_champion_pointer(root, staged)
            committed = True  # the single commit point: new checkpoint + its own knob serve
            # Q1: the previous checkpoint and its knob row are deleted right
            # after the swap; both deletes are best-effort, and a refused or
            # faulted one is retried next tick (reap_unreferenced /
            # _reap_orphan_knob_rows).
            judge_lora.cleanup_stale_adapters(root, keep_names=_keep_after_accept(staged.name))
            if prior_name is not None:
                try:
                    store.delete_judge_knob_calibration(judge_knob_key(MODEL_RELEVANCE_JUDGE, prior_name))
                except Exception:  # noqa: BLE001 — best-effort; the next tick's reap retries
                    logger.warning("judge self-tune: deleting the previous checkpoint's knob row faulted", exc_info=True)
            store.write_judge_selftune_state(MODEL_RELEVANCE_JUDGE, last_trained_at=now)
            # ACCEPT consume = doc-HAVING subset only (2B): a legacy
            # doc-absent row was NOT trained here (excluded from both the
            # train set and the re-score knob), so it stays re-eligible.
            doc_having = store.rows_with_doc_snapshot(row_ids)
            store.mark_selftune_consumed(doc_having, consumed_at=now)
            return (True, True)

        # REVERT: pointer untouched; discard the rejected staged dir (no knob
        # row was ever written for it). Knob from LOGGED scores (resulting
        # model == champion == logged-score model → no re-score, inc3 path),
        # under the serving model's key. Consume the FULL row_ids (2B).
        judge_lora.discard_stored_dir(staged)
        _weak_knob_refit_and_consume(store, now, row_ids, served_key)
        return (False, False)
    except Exception:
        # No rollback step (knob-first): before the commit the previous
        # checkpoint still serves with its own knob, so only the staged dir is
        # discarded (a staged knob row, if written, is reaped next tick); after
        # the commit the new checkpoint serves with its own knob and the
        # leftovers are reaped next tick. The original fault propagates to the
        # tick's handler (rows stay UNCONSUMED — fail-safe, re-eligible).
        if not committed:
            try:
                judge_lora.discard_stored_dir(staged)
            except Exception:  # noqa: BLE001 — cleanup is best-effort; never mask the original
                logger.warning("judge self-tune: discarding the staged checkpoint faulted", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# F2c (durable note, spec §6): Haiku is the effective relevance ORACLE this
# module's training code converges the local judge toward. The tick below
# (and the `_run_weight_retrain` helper above) fit the judge's score-to-label
# mapping (knob-refit / LoRA / full fine-tune) against the accumulated Haiku
# tie-break decisions logged in calibration_log, the same decisions
# relevance_judge.label_calibration_sample already treats as ground truth over
# the local judge's own provisional label at ambiguous positions (see that
# module's own durable note at its orchestration entry point, spec §6). If a
# relevance-quality problem shows up downstream later, this is one of the two
# places to look first: what the judge converges toward is Haiku's own
# labeling behavior, not an independently verified ground truth, so a
# systematic Haiku bias would propagate into the judge rather than being
# caught by it.
# ---------------------------------------------------------------------------


def _run_judge_selftune_tick(*, store, now: datetime, persona_dir: Path | None = None) -> dict:
    """One weekly judge self-tune tick (F2c inc2 cadence/gate + inc3 knob-refit
    + inc5b-2/inc6/inc7 weight-retrain lifecycle). Caller owns cadence +
    throttle — mirrors `interest_sweep.run_sweep_tick`'s contract exactly.
    Never raises.

    First (before the gate, when `persona_dir` is given) it reaps stored
    checkpoint dirs the pointer does not name (`judge_lora.reap_unreferenced`):
    the retry path for a previous checkpoint whose post-swap delete the OS
    refused, and the cleanup for a crash-orphaned staged dir (F2c inc7, ruling
    Q1). Then it reaps knob rows of checkpoints the pointer does not name
    (`_reap_orphan_knob_rows`, F2c inc9). Each is fault-isolated; neither ever
    blocks the tick.

    Then it counts UNCONSUMED non-None `haiku_label` positions across
    `calibration_log` (the >handful gate, spec §2/§3,
    `count_new_haiku_decisions`) and, ONLY when the gate fires: runtime-detects
    the RAM tune-grade + applies the cgroup-aware OOM-safety downgrade. The
    tier picks only this week's update METHOD (spec §1):

    - **WEAK tier** (`knob_refit`): the inc3 path unchanged —
      `_weak_knob_refit_and_consume` fits the Platt knob on the FULL logged
      set and consumes the full `row_ids`. The served model is untouched.
    - **LoRA / full-FT tiers** (mid/beefy): `_run_weight_retrain` trains this
      week's update ON TOP OF the persona's current model, runs the 2/3-1/3
      champion/challenger (McNemar, AC6) against that same current model, and
      on ACCEPT swaps the new plain checkpoint into the persona's one pointer
      (re-scored knob, previous checkpoint deleted). A DEGENERATE FLOOR (too
      few Haiku triples) falls through to the weak knob-refit.

    Spec §2's "consume = trained-on, never fired-on": consume is the LAST
    durable step on every path, so any fault before it (caught here) leaves
    `row_ids` UNCONSUMED and re-eligible next week.

    F2c inc8 (spec §3 "Retention vs the weekly tick", AC14): the gate counts,
    and so the tune trains on, only Haiku decisions logged within one weekly
    cadence of `now`; after a FIRED tick, `MemoryStore.clear_selftune_held_rows`
    deletes the rows kept only by the one-week hold that this tick trained on
    (or that rolled past a week), in its own fault guard (`cleared` /
    `clear_error`).

    Returns ``{"fired", "tune_grade", "method", "new_decisions", "current",
    "start", "accepted", "adapter_persisted", "error", "cleared",
    "clear_error"}`` (caller-facing; the
    `supervisor.run_folded` wiring ignores it). `method` = the update method
    that ran (`knob_refit` / `lora` / `full_ft`); `current` / `start` = the
    checkpoint name serving before the tick and the one this week trained
    from (`"base"` = the shared base judge); `adapter_persisted` (historical
    key name) = a new checkpoint was swapped in.
    """
    result: dict = {
        "fired": False,
        "tune_grade": None,
        "method": None,
        "new_decisions": 0,
        "current": None,
        "start": None,
        "accepted": None,
        "adapter_persisted": False,
        "error": None,
        "cleared": 0,
        "clear_error": None,
    }
    if persona_dir is not None:
        try:
            from brain.memory import judge_lora

            judge_lora.reap_unreferenced(judge_lora.champion_dir(persona_dir))
        except Exception:  # noqa: BLE001 — reaping is best-effort; never blocks the tick
            logger.warning("judge self-tune: leftover-checkpoint reap faulted", exc_info=True)
        try:
            _reap_orphan_knob_rows(store, persona_dir)
        except Exception:  # noqa: BLE001 — reaping is best-effort; never blocks the tick
            logger.warning("judge self-tune: orphan knob-row reap faulted", exc_info=True)
    try:
        gate_handful = tunables.get_tunable(
            "judge_selftune.gate_handful_decisions", JUDGE_TUNE_GATE_HANDFUL_DECISIONS
        )
        # F2c inc8: only Haiku decisions logged within one weekly cadence of
        # `now` are counted (and so trained on — every training read is scoped
        # to `row_ids`).
        count, row_ids = store.count_new_haiku_decisions(now=now)
        result["new_decisions"] = count
        if count <= gate_handful:
            return result  # not yet MORE than a handful — no fire, nothing consumed

        total_ram = _read_total_ram_bytes()
        tune_grade = _select_tune_grade_by_ram(total_ram)
        effective_headroom = _available_ram_headroom_bytes()
        tune_grade = _downgrade_for_oom_safety(tune_grade, effective_headroom)
        result["tune_grade"] = tune_grade

        if tune_grade == TUNE_GRADE_KNOB_REFIT or persona_dir is None:
            # Weak tier: knob-refit only (spec §5), inc3 behavior unchanged.
            # `persona_dir is None` (no per-persona store available) is a safe
            # floor: without a place to persist a checkpoint, a mid/beefy box
            # cannot run the weight-retrain, so it degrades to the always-safe
            # knob-refit (the supervisor always passes persona_dir; None only
            # arises in unit tests exercising the weak path or the gate/tier
            # logic).
            if tune_grade != TUNE_GRADE_KNOB_REFIT and persona_dir is None:
                logger.info(
                    "judge self-tune: tune_grade=%s but no persona_dir — floor to knob-refit",
                    tune_grade,
                )
            result["method"] = TUNE_GRADE_KNOB_REFIT
            from brain.memory.relevance_judge import judge_knob_key

            cur = None
            if persona_dir is not None:
                from brain.memory import judge_lora

                cur = judge_lora.resolve_current_checkpoint(persona_dir)
                result["current"] = cur.name if cur is not None else "base"
                # A knob week tunes (only the knob of) the serving model itself.
                result["start"] = result["current"]
            # F2c inc9: the knob is keyed to the model serving (the persona's
            # current checkpoint, else the plain base id).
            _weak_knob_refit_and_consume(store, now, row_ids, judge_knob_key(MODEL_RELEVANCE_JUDGE, cur))
        else:
            result["method"] = tune_grade
            info: dict = {}
            try:
                accepted, persisted = _run_weight_retrain(
                    store, now, persona_dir, row_ids, tune_grade, info
                )
            finally:
                result["current"] = info.get("current")
                result["start"] = info.get("start")
            result["accepted"] = accepted
            result["adapter_persisted"] = persisted
        result["fired"] = True
        logger.info(
            "judge self-tune: fired method=%s current=%s start=%s accepted=%s new_decisions=%d",
            result["method"],
            result["current"],
            result["start"],
            result["accepted"],
            count,
        )
    except Exception as exc:  # noqa: BLE001 — fault-isolated, mirrors run_sweep_tick
        logger.warning("judge self-tune tick failed: %s", exc)
        result["error"] = f"{type(exc).__name__}: {exc}"
    if result["fired"]:
        # F2c inc8 (spec §3): the accumulated rows are cleared once the weekly
        # self-tune has trained on them. Runs AFTER the tune's own end-of-tick
        # consume and OUTSIDE the handler above, in its own guard: the rows are
        # already trained on, so a clear fault must never mark the tune as
        # failed (`error`/`fired` untouched) — the next daily prune deletes
        # consumed held rows anyway.
        try:
            result["cleared"] = store.clear_selftune_held_rows(now=now)
        except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
            logger.warning("judge self-tune: clearing trained-on held rows faulted: %s", exc)
            result["clear_error"] = f"{type(exc).__name__}: {exc}"
    return result
