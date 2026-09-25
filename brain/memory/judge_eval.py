"""Champion/challenger evaluation + rollback infrastructure for F2c's
weight-retrain tiers (LoRA / full fine-tune) — F2c inc4b (spec
`f2c-judge-selftune-spec.md` §4 [eval + guardrail], §5 [tiered
weight-retrain build detail], AC5 [eval split], AC6 [champion/challenger +
rollback]).

TORCH-FREE, STANDALONE (I6/AC8): this module builds the eval SPLIT + the
AC6 statistical bar + the champion/challenger ORCHESTRATION shape only. It
is deliberately NOT wired into `judge_selftune._run_judge_selftune_tick`'s
tier dispatch (that wiring, and the real torch LoRA/full-FT `retrain_fn`,
are F2c inc5/6 — see that TODO marker in `judge_selftune.py`). Every judge
here is an injected plain callable (`item -> label`); `run_champion_
challenger` below never imports or constructs a real model, so importing —
or even fully exercising — this module never pulls torch/sentence_
transformers into `sys.modules`.

Lives in its own sibling module to `judge_selftune.py` rather than inside
it: `judge_selftune.py` already owns the cadence/gate/knob-refit (the
ALWAYS-ON, no-guard mechanism, spec §5); this module owns a DIFFERENT
concern — the guarded weight-retrain's statistical accept/revert bar and
its orchestration shape — that only the LoRA/full-FT tiers ever invoke, and
that inc5/6 will import from here as a unit (`split_train_test` +
`challenger_not_worse` + `run_champion_challenger` + `RollbackHandle`)
without pulling in `judge_selftune.py`'s cadence/RAM-tier-detection
machinery, which the champion/challenger step doesn't need.

Durable Haiku-oracle note (spec §6, mirrors the note in `judge_selftune.py`
directly above `_run_judge_selftune_tick`, and the one at F2a's judge/label
site in `relevance_judge.py`): the held-out test-set "oracle" this module's
`run_champion_challenger` scores champion and challenger against is the
accumulated HAIKU tie-break decisions, not an independently verified
ground truth. A systematic Haiku bias would propagate into which judge
this module ACCEPTS, the same caveat the knob-refit's training data
carries — if a relevance-quality problem shows up downstream later, the
champion/challenger accept/revert decision built here is one of the places
to look, alongside the knob-refit's training data.

Production NOTE for inc5/6: the unit this module's `split_train_test`/
`run_champion_challenger` must be fed is STRICTLY the non-None
`haiku_label` POSITIONS from `calibration_log` — the SAME counting unit
spec §2 pins for the weekly gate ("the accumulated Haiku decisions" =
individual Haiku-labeled (query, doc) pairs). This is NARROWER than
`MemoryStore.judge_knob_refit_pairs`'s pairs, which also include positions
where only the LOCAL judge labeled (Haiku never adjudicated) — that
broader effective-label set is correct for the knob-refit (more training
signal, no guard needed) but would leak non-oracle-backed positions into
the eval split if reused here unchanged. F2c inc5b-1 built exactly this
extraction — `MemoryStore.judge_lora_training_triples(row_ids)` in
`brain/memory/store.py`, `judge_knob_refit_pairs`-shaped but Haiku-only,
returning `(query, doc, label)` triples straight off `candidate_docs` (no
re-fetch) — for `judge_lora.build_lora_retrain_fn` to train on; inc5b-2
still owes wiring `split_train_test`/`run_champion_challenger` to call it.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from brain import tunables

logger = logging.getLogger(__name__)

# --- Tunables (I3/I7 — every threshold below is a registered tunable, never
# a bare hand-picked constant baked into the logic). Shares the
# `judge_selftune.*` key namespace with `judge_selftune.py`'s own tunables
# — both belong to the ONE F2c feature (mirrors `floor_calibration.py` and
# `relevance_judge.py` sharing the `calibration.*` namespace despite living
# in different files: the namespace tracks the FEATURE, not the module). ---

# One-sided McNemar-exact significance threshold (spec §4, PINNED
# 2026-09-24): "a standard significance convention... not a model-pinned
# magic number" — a documented, widely-used default, not hand-picked for
# this codebase.
JUDGE_EVAL_ALPHA_DEFAULT: float = tunables.register("judge_selftune.eval_alpha", 0.05)

# Held-out test-set size floor (spec §4 "DEGENERATE FLOOR"): below this,
# there is too little data to evaluate the challenger at all, so the
# orchestration REVERTS unconditionally rather than deploying on thin data.
# A build-time judgment call (same posture as `judge_selftune.
# JUDGE_TUNE_GATE_HANDFUL_DECISIONS`'s own docstring: not hardware-derived,
# so a dry-run doesn't set it either) — 30 is the conventional rule-of-thumb
# floor for a binomial-family test to have any meaningful power at all.
JUDGE_EVAL_MIN_TEST_N_DEFAULT: int = tunables.register("judge_selftune.eval_min_test_n", 30)


# ---------------------------------------------------------------------------
# AC5 — 2/3 train / 1/3 test split, no leakage, deterministic.
# ---------------------------------------------------------------------------


def split_train_test[T](
    items: Sequence[T],
    *,
    train_frac: float = 2.0 / 3.0,
    seed: int = 0,
) -> tuple[list[T], list[T]]:
    """Deterministic 2/3-train / 1/3-test split of `items` (spec §4 AC5).

    `items` is whatever unit the caller is splitting — in production, the
    non-None `haiku_label` POSITIONS (see this module's docstring above),
    the same unit spec §2 pins for the weekly gate. This function is
    content-agnostic: it partitions by POSITION in `items`, so whatever
    `items` holds (tuples, dicts, dataclasses...), each element lands on
    exactly one side — NO LEAKAGE by construction, since train/test are
    built from two disjoint index sets covering `items` exactly once each
    (verified in tests via id-sets, not just index math, since the
    production caller cares about the CONTENT not repeating across sides).

    DETERMINISTIC (spec: "seeded... so re-runs are reproducible"): a fixed
    `seed` (default 0) drives a `random.Random` instance private to this
    call — never the global `random` module — so the SAME `items` (in the
    same order) always produces the SAME split, independent of anything
    else the process has done with `random` elsewhere. Re-ordering `items`
    changes the split (position IS the identity here); callers that need
    the split to survive a re-fetch in a different row order should sort
    `items` by a stable key (e.g. the source row id) before calling.

    `train_frac` controls the split point via `round(n * train_frac)` —
    the default 2/3 matches spec §4 exactly. An `items` too short to make
    both sides non-empty still splits (e.g. n=1 puts it in train) — the
    DEGENERATE FLOOR on the TEST side is `challenger_not_worse`'s
    `min_n` job (§4), not this function's: splitting and "is there enough
    to evaluate on" are separate concerns.
    """
    n = len(items)
    n_train = round(n * train_frac)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    train_idx = sorted(order[:n_train])
    test_idx = sorted(order[n_train:])
    return [items[i] for i in train_idx], [items[i] for i in test_idx]


# ---------------------------------------------------------------------------
# AC6 — the pinned "not measurably worse" bar: one-sided McNemar EXACT test
# on the discordant pairs (spec §4, PINNED 2026-09-24).
# ---------------------------------------------------------------------------


def _discordant_counts(
    champion_correct: Sequence[bool], challenger_correct: Sequence[bool]
) -> tuple[int, int]:
    """`(b, c)` — `b` = champion-correct-AND-challenger-wrong pairs, `c` =
    challenger-correct-AND-champion-wrong pairs (spec §4's own naming).
    Raises `ValueError` on a length mismatch — the two sequences must be
    PAIRED (same held-out test set, same item order), never independently
    sized.
    """
    if len(champion_correct) != len(challenger_correct):
        raise ValueError(
            "champion_correct and challenger_correct must be the same length "
            f"(paired on the same test set): got {len(champion_correct)} vs "
            f"{len(challenger_correct)}"
        )
    b = sum(
        1
        for champ_ok, chall_ok in zip(champion_correct, challenger_correct, strict=True)
        if champ_ok and not chall_ok
    )
    c = sum(
        1
        for champ_ok, chall_ok in zip(champion_correct, challenger_correct, strict=True)
        if chall_ok and not champ_ok
    )
    return b, c


def mcnemar_exact_p(b: int, c: int) -> float:
    """One-sided EXACT McNemar p-value: `P(X >= b)`, `X ~ Binomial(b+c,
    0.5)` (spec §4, PINNED 2026-09-24) — "the challenger is measurably
    worse iff `b` significantly exceeds `c`".

    EXACT, not normal-approx (spec: "robust at small discordant counts"):
    computed as `sum(math.comb(m, k) for k in range(b, m+1)) / 2**m` —
    `math.comb` returns an exact Python arbitrary-precision integer for
    both the individual binomial coefficients and their sum, so the ONLY
    floating-point operation in the whole computation is the single final
    true-division of two exact integers, which Python's `/` operator on
    ints computes as the correctly-rounded nearest `float` regardless of
    how large the integers are (arbitrary-precision numerator/denominator,
    not an intermediate float that could lose precision). No scipy
    dependency, no chi-square normal approximation.

    `b <= 0` (challenger never loses a discordant pair, or there are no
    discordant pairs at all) returns `1.0` (certainly not worse). `b > m`
    is unreachable by construction (`b` is one of the `m = b + c`
    discordant pairs) but returns `0.0` defensively rather than raising.
    """
    m = b + c
    if b <= 0:
        return 1.0
    if b > m:
        return 0.0
    tail = sum(math.comb(m, k) for k in range(b, m + 1))
    return tail / (2**m)


def challenger_not_worse(
    champion_correct: Sequence[bool],
    challenger_correct: Sequence[bool],
    *,
    alpha: float = JUDGE_EVAL_ALPHA_DEFAULT,
    min_n: int = JUDGE_EVAL_MIN_TEST_N_DEFAULT,
) -> bool:
    """The AC6 accept/revert bar, as a PURE function (spec §4, PINNED
    2026-09-24): `True` ("accept" — adopt the challenger) iff the held-out
    test set is at least `min_n` AND the challenger is NOT measurably
    worse than the champion (one-sided McNemar exact `p >= alpha` on the
    discordant pairs, per `mcnemar_exact_p` above); `False` ("revert" —
    keep the champion) otherwise.

    `champion_correct[i]` / `challenger_correct[i]` = whether that judge's
    label on held-out test item `i` matched the Haiku ORACLE label — the
    caller (`run_champion_challenger` below, or a real inc5/6 caller)
    computes these by scoring both judges against the SAME test set; this
    function takes only the resulting booleans, no scores/labels, so it
    stays independent of what a "judge" or "item" even is.

    Ties AND improvements are KEPT (spec: "only a statistically-significant
    regression reverts") — `p >= alpha` covers both `b == c` (tie) and
    `c > b` (challenger better) as well as a `b > c` regression too small
    to be significant at `alpha`.

    `alpha`/`min_n` PURE-FUNCTION KEYWORD DEFAULTS reference this module's
    own `tunables.register`-ed constants directly (`JUDGE_EVAL_ALPHA_
    DEFAULT`/`JUDGE_EVAL_MIN_TEST_N_DEFAULT` above) rather than
    re-hardcoding the numbers here — ONE source of truth for the default
    value, no duplicate magic number (mirrors `fit_platt_knob`'s sibling
    pure functions in `judge_selftune.py`, which also take plain
    parameters and leave TUNABLE-OVERRIDE RESOLUTION to the caller: this
    function itself never calls `tunables.get_tunable` — an orchestrating
    caller like `run_champion_challenger` below resolves the live
    tunables.json override and passes the resolved value in, so this stays
    a pure, deterministically-testable function of its arguments alone).
    """
    n = len(champion_correct)
    if len(challenger_correct) != n:
        raise ValueError(
            "champion_correct and challenger_correct must be the same length "
            f"(paired on the same test set): got {n} vs {len(challenger_correct)}"
        )
    if n < min_n:
        return False  # DEGENERATE FLOOR (spec §4): too little data — revert regardless of b/c.
    b, c = _discordant_counts(champion_correct, challenger_correct)
    p = mcnemar_exact_p(b, c)
    return not (p < alpha)


# ---------------------------------------------------------------------------
# Champion/challenger + rollback orchestration (spec §4/AC6).
# ---------------------------------------------------------------------------


class RollbackHandle:
    """Rollback abstraction `run_champion_challenger` calls against; this
    module only defines the shape and a scripted/testing default. The
    production tick (`judge_selftune._run_weight_retrain`) passes this no-op
    default: its challenger is written to a fresh staged checkpoint and the
    persona's one pointer is swapped only on ACCEPT (F2c inc7), so a revert
    has nothing to restore.

    `record()` is called UNCONDITIONALLY, before `retrain_fn` runs, and
    must capture whatever is needed to put the CURRENT (pre-retrain,
    champion) state back — it returns an opaque snapshot token (whatever
    `restore` needs). `restore(snapshot)` is called ONLY when the AC6 bar
    rejects the challenger, never on accept (spec §4: "keep the new
    weights only if... not measurably worse... else revert with the
    recorded handle" — accept means keep what `retrain_fn` produced,
    nothing to restore).

    This default implementation is a plain in-memory no-op pair (`record`
    returns `None`, `restore` does nothing) — correct for `run_champion_
    challenger`'s functional contract in THIS increment (where
    `retrain_fn` returns a brand-new challenger label fn rather than
    mutating the champion in place, so there is nothing in-process to
    restore). Tests use a scripted subclass that RECORDS calls, to prove `restore`
    fires on revert and never on accept.
    """

    def record(self) -> Any:
        return None

    def restore(self, snapshot: Any) -> None:  # noqa: ARG002 — no-op default; see class docstring
        return None


@dataclass(frozen=True)
class ChampionChallengerResult:
    """`run_champion_challenger`'s return — the AC6 decision plus enough of
    the underlying McNemar arithmetic (`b`, `c`, `p_value`) for the caller
    to log/persist WHY, not just the bool.
    """

    accepted: bool  # True = challenger adopted (next champion); False = reverted
    judge: Callable[[Any], str]  # the challenger if accepted, else the original champion
    b: int
    c: int
    p_value: float
    n_test: int
    champion_correct: list[bool]
    challenger_correct: list[bool]


def run_champion_challenger(
    *,
    champion: Callable[[Any], str],
    retrain_fn: Callable[[Sequence[Any]], Callable[[Any], str]],
    train_items: Sequence[Any],
    test_items: Sequence[tuple[Any, str]],
    rollback: RollbackHandle,
    alpha: float | None = None,
    min_n: int | None = None,
) -> ChampionChallengerResult:
    """The champion/challenger + rollback orchestration (spec §4/AC6):
    record the rollback handle -> obtain the challenger via `retrain_fn` ->
    forward-only-evaluate BOTH champion and challenger on the SAME held-out
    test set against the Haiku oracle labels -> apply the AC6 bar -> ACCEPT
    (challenger becomes next champion) or REVERT (restore via the handle,
    champion unchanged).

    `champion`: the current judge, a plain callable `item -> label`.

    `retrain_fn`: produces the CHALLENGER from `train_items` — in this
    increment's tests, a scripted function; inc5/6 supplies the real torch
    LoRA/full-FT retrain there, wrapped to return a callable label fn with
    the SAME `item -> label` shape as `champion` (whatever "item" means is
    entirely up to the caller — this function never inspects it). Called
    EXACTLY ONCE, with `train_items` only — never `test_items` (AC5's
    no-leakage contract extends here: the eval step below must never let
    `retrain_fn` see the held-out set).

    `test_items`: `(item, haiku_oracle_label)` pairs — the held-out 1/3
    from `split_train_test` (paired with their oracle labels by the
    caller, since this module doesn't know how to look an oracle label up
    on its own).

    FORWARD-ONLY (spec §4: "no activations, no backward"): the eval step
    below calls `champion(item)`/`challenger(item)` — label-fn calls only
    — once per `test_items` entry per judge; it never calls `retrain_fn`
    again and never calls any training/backward entry point (there isn't
    one in this function's contract for it to call — `retrain_fn` is the
    ONLY training-shaped call in this whole orchestration, and it runs
    once, before the eval loop starts).

    `alpha`/`min_n` default to `None`, meaning "read the live
    `tunables.json` override (falling back to this module's registered
    default) at call time" — this is the ORCHESTRATION layer, so unlike
    `challenger_not_worse`'s own plain-parameter pure-function contract,
    THIS function is where a tunables override actually takes effect
    (mirrors `judge_selftune._run_judge_selftune_tick` resolving its own
    tunables via `tunables.get_tunable` right before calling the pure
    fit function it wraps).
    """
    resolved_alpha = (
        alpha
        if alpha is not None
        else tunables.get_tunable("judge_selftune.eval_alpha", JUDGE_EVAL_ALPHA_DEFAULT)
    )
    resolved_min_n = (
        min_n
        if min_n is not None
        else tunables.get_tunable("judge_selftune.eval_min_test_n", JUDGE_EVAL_MIN_TEST_N_DEFAULT)
    )

    snapshot = rollback.record()
    challenger = retrain_fn(train_items)

    champion_correct = [champion(item) == oracle_label for item, oracle_label in test_items]
    challenger_correct = [challenger(item) == oracle_label for item, oracle_label in test_items]

    accepted = challenger_not_worse(
        champion_correct, challenger_correct, alpha=resolved_alpha, min_n=resolved_min_n
    )
    b, c = _discordant_counts(champion_correct, challenger_correct)
    p_value = mcnemar_exact_p(b, c)

    if accepted:
        next_judge = challenger
    else:
        rollback.restore(snapshot)
        next_judge = champion

    logger.info(
        "judge champion/challenger: accepted=%s b=%d c=%d p=%.6g n_test=%d",
        accepted,
        b,
        c,
        p_value,
        len(test_items),
    )
    return ChampionChallengerResult(
        accepted=accepted,
        judge=next_judge,
        b=b,
        c=c,
        p_value=p_value,
        n_test=len(test_items),
        champion_correct=champion_correct,
        challenger_correct=challenger_correct,
    )
