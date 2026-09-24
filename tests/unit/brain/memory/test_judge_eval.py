"""Tests for `brain.memory.judge_eval` — F2c inc4b's champion/challenger +
rollback evaluation INFRASTRUCTURE for the weight-retrain tiers (spec
`f2c-judge-selftune-spec.md` §4 [eval + guardrail], AC5 [eval split], AC6
[champion/challenger + rollback]).

TORCH-FREE (I6/AC8): every judge below is a scripted plain callable; see
`test_run_champion_challenger_does_not_import_torch_or_sentence_
transformers` for the fresh-subprocess proof this module (and everything
it transitively imports) never pulls torch/sentence_transformers into
`sys.modules`, mirroring `test_judge_selftune.py`'s own sibling test.

This increment does NOT wire into `judge_selftune`'s tier dispatch or the
live tick (inc5/6 does) — these tests exercise `judge_eval`'s pure
functions and orchestration shape standalone, with scripted
train/eval/rollback stand-ins, never a real model or `MemoryStore`.
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from brain.memory import judge_eval
from brain.memory.judge_eval import (
    ChampionChallengerResult,
    RollbackHandle,
    challenger_not_worse,
    mcnemar_exact_p,
    run_champion_challenger,
    split_train_test,
)

# ---------------------------------------------------------------------------
# AC5 — split_train_test: ratio, no leakage, determinism.
# ---------------------------------------------------------------------------


def test_split_ratio_is_two_thirds_train_one_third_test() -> None:
    items = [f"item{i}" for i in range(30)]
    train, test = split_train_test(items)
    assert len(train) == 20  # round(30 * 2/3)
    assert len(test) == 10


def test_split_no_leakage_disjoint_id_sets() -> None:
    """A test item never appears in train (AC5) — checked by CONTENT
    identity (a set intersection), not merely index math, since a real
    caller cares that the same (query, doc) pair never lands on both
    sides.
    """
    items = [
        {"id": i, "haiku_label": "relevant" if i % 2 == 0 else "irrelevant"} for i in range(30)
    ]
    # dicts aren't hashable — split on the ids, which is what a real
    # caller would actually check disjointness on.
    train, test = split_train_test(items)
    train_ids = {d["id"] for d in train}
    test_ids = {d["id"] for d in test}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {d["id"] for d in items}
    assert len(train_ids) + len(test_ids) == len(items)  # every item on exactly one side


def test_split_is_deterministic_same_input_same_split() -> None:
    items = [f"item{i}" for i in range(41)]  # not evenly divisible — exercises rounding too
    train1, test1 = split_train_test(items)
    train2, test2 = split_train_test(items)
    assert train1 == train2
    assert test1 == test2


def test_split_different_seed_can_produce_a_different_split() -> None:
    items = [f"item{i}" for i in range(41)]
    train_a, _ = split_train_test(items, seed=0)
    train_b, _ = split_train_test(items, seed=1)
    assert train_a != train_b  # not a hard guarantee for all n, but true for this n/seed pair


def test_split_handles_tiny_input() -> None:
    train, test = split_train_test(["only"])
    assert train == ["only"]
    assert test == []


# ---------------------------------------------------------------------------
# AC6 — mcnemar_exact_p: known b/c values, exact not normal-approx.
# ---------------------------------------------------------------------------


def test_mcnemar_exact_p_strong_regression_b10_c0() -> None:
    # Spec §4 worked example: b=10, c=0 -> p = 0.5**10 ~= 0.000977 < 0.05 -> worse.
    p = mcnemar_exact_p(10, 0)
    assert p == pytest.approx(0.5**10, rel=1e-12)
    assert p < 0.05


def test_mcnemar_exact_p_mild_regression_b6_c2_kept() -> None:
    # Spec §4 worked example: b=6, c=2 -> p = P(X>=6 | Binomial(8,0.5)) = 37/256 ~= 0.144 > 0.05.
    p = mcnemar_exact_p(6, 2)
    assert p == pytest.approx(37.0 / 256.0, rel=1e-12)
    assert p > 0.05


def test_mcnemar_exact_p_tie_b5_c5_not_worse() -> None:
    p = mcnemar_exact_p(5, 5)
    assert p > 0.05  # ties are never "measurably worse"


def test_mcnemar_exact_p_no_discordant_pairs_is_certain_not_worse() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_exact_p_zero_b_is_certain_not_worse_regardless_of_c() -> None:
    assert mcnemar_exact_p(0, 8) == 1.0


def test_mcnemar_swapping_b_and_c_flips_the_decision() -> None:
    """Mutation-style bite: swapping b<->c must flip which side "loses" —
    a McNemar implementation that accidentally used `c` where it meant
    `b` (or vice versa) would pass every single-argument test above but
    fail this one.
    """
    p_b_worse = mcnemar_exact_p(8, 0)  # champion much better -> challenger worse
    p_c_worse = mcnemar_exact_p(0, 8)  # challenger much better -> challenger NOT worse
    assert p_b_worse < 0.05
    assert p_c_worse == 1.0
    assert p_b_worse != p_c_worse


def test_mcnemar_exact_differs_from_normal_approximation_at_small_n() -> None:
    """Verifies EXACT, not normal-approx (spec §4): b=4, c=0 (m=4
    discordant pairs) is a small-n case where the two disagree on the
    DECISION itself, not merely the numeric p-value — this is exactly why
    the spec bars the chi-square/normal approximation.

    Exact: p = P(X>=4 | Binomial(4,0.5)) = 1/16 = 0.0625 > 0.05 -> NOT
    worse (accept).
    Normal approx (no continuity correction): z = (b-c)/sqrt(b+c) = 4/2 =
    2.0, one-sided p = 1 - Phi(2.0) ~= 0.0228 < 0.05 -> WOULD say worse
    (revert) -- the wrong call an approximation makes at small n.
    """
    b, c = 4, 0
    exact_p = mcnemar_exact_p(b, c)
    assert exact_p == pytest.approx(1.0 / 16.0, rel=1e-12)
    assert exact_p > 0.05  # exact: not measurably worse

    m = b + c
    z = (b - c) / math.sqrt(m)
    normal_approx_p = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    assert normal_approx_p < 0.05  # normal approx: would (wrongly) call it worse
    assert exact_p != pytest.approx(normal_approx_p, rel=1e-3)


# ---------------------------------------------------------------------------
# AC6 — challenger_not_worse: the accept/revert bool, min-n floor.
# ---------------------------------------------------------------------------


def test_challenger_not_worse_reverts_on_significant_regression() -> None:
    champion_correct = [True] * 10 + [False] * 0
    challenger_correct = [
        False
    ] * 10  # every one of the 10 champion-correct pairs, challenger wrong
    assert challenger_not_worse(champion_correct, challenger_correct, min_n=1) is False


def test_challenger_not_worse_accepts_mild_regression_b6_c2() -> None:
    # 6 champion-only-correct, 2 challenger-only-correct, rest concordant.
    champion_correct = [True] * 6 + [False] * 2 + [True] * 2
    challenger_correct = [False] * 6 + [True] * 2 + [True] * 2
    assert challenger_not_worse(champion_correct, challenger_correct, min_n=1) is True


def test_challenger_not_worse_accepts_tie() -> None:
    champion_correct = [True] * 5 + [False] * 5
    challenger_correct = [False] * 5 + [True] * 5
    assert challenger_not_worse(champion_correct, challenger_correct, min_n=1) is True


def test_challenger_not_worse_accepts_pure_improvement() -> None:
    champion_correct = [False] * 8
    challenger_correct = [True] * 8
    assert challenger_not_worse(champion_correct, challenger_correct, min_n=1) is True


def test_challenger_not_worse_min_n_floor_reverts_regardless_of_counts() -> None:
    """DEGENERATE FLOOR (spec §4): even a challenger that is PERFECT where
    the champion is wrong everywhere must still revert if the held-out set
    is below the tunable minimum — never deploy on thin data.
    """
    champion_correct = [False] * 5
    challenger_correct = [True] * 5  # challenger looks flawlessly better...
    assert (
        challenger_not_worse(champion_correct, challenger_correct, min_n=1) is True
    )  # ...without the floor
    assert (
        challenger_not_worse(champion_correct, challenger_correct, min_n=10) is False
    )  # ...with it


def test_challenger_not_worse_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        challenger_not_worse([True, False], [True], min_n=1)


def test_challenger_not_worse_default_alpha_and_min_n_are_the_registered_tunable_defaults() -> None:
    import inspect

    sig = inspect.signature(challenger_not_worse)
    assert sig.parameters["alpha"].default == judge_eval.JUDGE_EVAL_ALPHA_DEFAULT
    assert sig.parameters["min_n"].default == judge_eval.JUDGE_EVAL_MIN_TEST_N_DEFAULT


# ---------------------------------------------------------------------------
# run_champion_challenger — orchestration: accept/revert, rollback wiring,
# forward-only contract.
# ---------------------------------------------------------------------------


class _CountingLabelFn:
    """A scripted label fn that counts calls and looks labels up from a
    plain dict, so tests can assert exactly how many times (and with what
    arguments) `run_champion_challenger` invokes it — the forward-only
    bite.
    """

    def __init__(self, labels: dict[str, str]) -> None:
        self.labels = labels
        self.calls: list[str] = []

    def __call__(self, item: str) -> str:
        self.calls.append(item)
        return self.labels[item]


class _RecordingRollback(RollbackHandle):
    def __init__(self, shared_log: list[str] | None = None) -> None:
        self.record_calls = 0
        self.restore_calls: list[object] = []
        self._shared_log = shared_log if shared_log is not None else []

    def record(self) -> object:
        self.record_calls += 1
        token = f"snapshot-{self.record_calls}"
        self._shared_log.append(f"record:{token}")
        return token

    def restore(self, snapshot: object) -> None:
        self.restore_calls.append(snapshot)
        self._shared_log.append(f"restore:{snapshot}")


def _make_retrain_fn(
    challenger_labels: dict[str, str], call_log: list[str] | None = None
) -> tuple[list[list[str]], object]:
    calls: list[list[str]] = []

    def retrain_fn(train_items: list[str]) -> _CountingLabelFn:
        calls.append(list(train_items))
        if call_log is not None:
            call_log.append("retrain")
        return _CountingLabelFn(challenger_labels)

    return calls, retrain_fn


def test_run_champion_challenger_reverts_on_significantly_worse_challenger() -> None:
    test_items = [(f"t{i}", "relevant") for i in range(8)]
    champion = _CountingLabelFn({f"t{i}": "relevant" for i in range(8)})  # champion always correct
    challenger_calls, retrain_fn = _make_retrain_fn({f"t{i}": "irrelevant" for i in range(8)})
    rollback = _RecordingRollback()

    result = run_champion_challenger(
        champion=champion,
        retrain_fn=retrain_fn,
        train_items=["train0", "train1"],
        test_items=test_items,
        rollback=rollback,
        alpha=0.05,
        min_n=1,
    )

    assert isinstance(result, ChampionChallengerResult)
    assert result.accepted is False
    assert result.judge is champion  # champion retained
    assert result.b == 8
    assert result.c == 0
    assert result.p_value == pytest.approx(1.0 / 256.0, rel=1e-12)
    assert rollback.record_calls == 1
    assert rollback.restore_calls == ["snapshot-1"]  # restore fired, with the recorded snapshot
    assert len(challenger_calls) == 1  # retrain_fn called exactly once
    assert challenger_calls[0] == ["train0", "train1"]  # called with train_items, never test_items


def test_run_champion_challenger_accepts_not_worse_challenger() -> None:
    test_items = [(f"t{i}", "relevant") for i in range(8)]
    champion = _CountingLabelFn({f"t{i}": "irrelevant" for i in range(8)})  # champion always wrong
    _calls, retrain_fn = _make_retrain_fn(
        {f"t{i}": "relevant" for i in range(8)}
    )  # challenger always right
    rollback = _RecordingRollback()

    result = run_champion_challenger(
        champion=champion,
        retrain_fn=retrain_fn,
        train_items=["train0"],
        test_items=test_items,
        rollback=rollback,
        alpha=0.05,
        min_n=1,
    )

    assert result.accepted is True
    assert result.judge is not champion  # the challenger, adopted as next champion
    assert result.b == 0
    assert result.c == 8
    assert rollback.record_calls == 1
    assert rollback.restore_calls == []  # never restored on accept


def test_run_champion_challenger_min_n_floor_reverts_even_a_clean_improvement() -> None:
    test_items = [(f"t{i}", "relevant") for i in range(5)]
    champion = _CountingLabelFn({f"t{i}": "irrelevant" for i in range(5)})
    _calls, retrain_fn = _make_retrain_fn({f"t{i}": "relevant" for i in range(5)})
    rollback = _RecordingRollback()

    result = run_champion_challenger(
        champion=champion,
        retrain_fn=retrain_fn,
        train_items=[],
        test_items=test_items,
        rollback=rollback,
        alpha=0.05,
        min_n=10,  # floor above the 5-item test set
    )

    assert result.accepted is False
    assert rollback.restore_calls  # reverted because of the floor, not the statistics


def test_run_champion_challenger_is_forward_only() -> None:
    """No training/backward call beyond the ONE `retrain_fn` invocation:
    `champion`/the returned challenger are called exactly once per test
    item each, and `retrain_fn` is called exactly once, with `train_items`
    only.
    """
    test_items = [(f"t{i}", "relevant") for i in range(6)]
    champion = _CountingLabelFn({f"t{i}": "relevant" for i in range(6)})
    challenger_calls, retrain_fn = _make_retrain_fn({f"t{i}": "relevant" for i in range(6)})
    rollback = _RecordingRollback()

    result = run_champion_challenger(
        champion=champion,
        retrain_fn=retrain_fn,
        train_items=["train0", "train1", "train2"],
        test_items=test_items,
        rollback=rollback,
        alpha=0.05,
        min_n=1,
    )

    assert len(challenger_calls) == 1  # retrain_fn: exactly once
    assert champion.calls == [f"t{i}" for i in range(6)]  # champion: once per test item, eval order
    assert result.judge.calls == [f"t{i}" for i in range(6)]  # challenger: once per test item too
    assert result.n_test == 6


def test_run_champion_challenger_records_rollback_before_retraining() -> None:
    """Ordering bite: `record()` must be called BEFORE `retrain_fn` runs —
    a rollback snapshot taken AFTER the challenger already exists would be
    a snapshot of nothing useful to restore.
    """
    shared_log: list[str] = []
    rollback = _RecordingRollback(shared_log)
    # 5 items, all champion-correct/challenger-wrong -> b=5, c=0, p=0.5**5 ~= 0.031 < 0.05,
    # a clean statistically-significant regression (not just the floor) forces the revert
    # path this test needs to observe `restore` firing.
    test_items = [(f"t{i}", "relevant") for i in range(5)]
    champion = _CountingLabelFn({f"t{i}": "relevant" for i in range(5)})

    def retrain_fn(_train_items: list[str]) -> _CountingLabelFn:
        assert shared_log == ["record:snapshot-1"], "retrain_fn ran before rollback.record()"
        shared_log.append("retrain")
        return _CountingLabelFn({f"t{i}": "irrelevant" for i in range(5)})  # deliberately worse

    run_champion_challenger(
        champion=champion,
        retrain_fn=retrain_fn,
        train_items=[],
        test_items=test_items,
        rollback=rollback,
        alpha=0.05,
        min_n=1,
    )
    assert shared_log == ["record:snapshot-1", "retrain", "restore:snapshot-1"]


def test_default_rollback_handle_is_a_safe_noop() -> None:
    handle = RollbackHandle()
    snapshot = handle.record()
    handle.restore(snapshot)  # must not raise


def test_run_champion_challenger_uses_tunable_defaults_when_alpha_min_n_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`alpha`/`min_n` default to `None` on the orchestration function,
    meaning "read the live tunable" -- override `judge_selftune.
    eval_min_test_n` via the tunables module directly and confirm it takes
    effect without passing `min_n` explicitly.
    """
    from brain import tunables

    monkeypatch.setattr(
        tunables, "_load_overrides_locked", lambda: {"judge_selftune.eval_min_test_n": 100}
    )
    test_items = [(f"t{i}", "relevant") for i in range(5)]
    champion = _CountingLabelFn({f"t{i}": "irrelevant" for i in range(5)})
    _calls, retrain_fn = _make_retrain_fn({f"t{i}": "relevant" for i in range(5)})
    rollback = _RecordingRollback()

    result = run_champion_challenger(
        champion=champion,
        retrain_fn=retrain_fn,
        train_items=[],
        test_items=test_items,
        rollback=rollback,
        # alpha/min_n omitted -> resolved from tunables.get_tunable at call time
    )
    assert result.accepted is False  # 5 < the overridden floor of 100


# ---------------------------------------------------------------------------
# AC8 — off hot path / torch scoping: a fresh subprocess proves exercising
# the champion/challenger path never imports torch/sentence_transformers.
# ---------------------------------------------------------------------------


def test_run_champion_challenger_does_not_import_torch_or_sentence_transformers() -> None:
    """Mirrors `test_judge_selftune.py`'s
    `test_tick_does_not_import_torch_or_sentence_transformers`: a FRESH
    subprocess (never inheriting this test process's own `sys.modules`)
    imports `judge_eval`, runs a full scripted `run_champion_challenger`
    pass (including a challenger REJECTION, which also exercises
    `rollback.restore`), and only then asserts neither package landed in
    `sys.modules`.
    """
    script = textwrap.dedent(
        """
        import sys

        from brain.memory.judge_eval import RollbackHandle, run_champion_challenger

        test_items = [(f"t{i}", "relevant") for i in range(8)]
        champion = lambda item: "relevant"
        labels = {f"t{i}": "irrelevant" for i in range(8)}

        def retrain_fn(train_items):
            return lambda item: labels[item]

        result = run_champion_challenger(
            champion=champion,
            retrain_fn=retrain_fn,
            train_items=["x"],
            test_items=test_items,
            rollback=RollbackHandle(),
            alpha=0.05,
            min_n=1,
        )
        assert result.accepted is False, result

        assert "torch" not in sys.modules, sorted(sys.modules)
        assert "sentence_transformers" not in sys.modules, sorted(sys.modules)
        print("SUBPROCESS_OK")
        """
    )
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SUBPROCESS_OK" in proc.stdout, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
