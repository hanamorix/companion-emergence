"""Name-recall fix R1-recovery: stuck-width recovery at a chat lull (spec §7;
S100, S113, S116; criteria C15a-C15h, C16d, CONC-8, CONC-9 in
`changes/name-recall-fix/R1-recovery/1.5-criteria.md`).

All offline: the scripted clock and cost-scripted fake reranker of
`test_rerank_cost_model.py` (a call costs overhead + rate x padded tokens; one
token per character), the runtime model id patched to the fake's, the full
latency budget patched through `tunables.get_tunable`. No model, no network.
Recovery counts are checked against independent computations with
`fractions.Fraction`.
"""

from __future__ import annotations

import ast
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import brain.memory.reranker as reranker_mod
from brain import tunables
from brain.bridge import supervisor
from brain.bridge.central_cadence import JobOutcome
from brain.memory import cost_period
from brain.memory.reranker import rerank_cost_estimate, rerank_for_recall
from tests.unit.brain.bridge.test_central_cadence import (
    _denying_slot,
    _granting_slot,
    _persona,
    _real_jobs,
)
from tests.unit.brain.memory.test_rerank_cost_model import (
    _MODEL,
    _QUERY,
    _Clock,
    _CostScriptedProvider,
    _docs,
)

_B = 4.0
# Pair size of every test document: longer than every anchor pair, so the
# longest pair of a 5 + 2 batch is a candidate and x = 7 * _L.
_L = 600
_X5 = 7 * _L
_REPO = Path(__file__).resolve().parents[4]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(reranker_mod, "_clock", c)
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _MODEL)
    real = tunables.get_tunable

    def _get(key, default):
        return _B if key == "reranker.latency_budget_seconds" else real(key, default)

    monkeypatch.setattr(tunables, "get_tunable", _get)
    return c


def _cached(monkeypatch: pytest.MonkeyPatch, provider) -> None:
    monkeypatch.setitem(reranker_mod._provider_cache, _MODEL, provider)


def _true_cost_provider(clock: _Clock, seconds_for_five: float, overhead: float = 0.0):
    """A fake whose 5 + 2 batch of `_L`-token pairs costs `seconds_for_five`."""
    return _CostScriptedProvider(clock, overhead=overhead, rate=(seconds_for_five - overhead) / _X5)


def _message(provider, *, budget: float = _B, docs=None):
    return rerank_for_recall(provider, _QUERY, docs or _docs(50, _L), budget_seconds=budget)


def _seven_doc_calls(provider) -> list[list[str]]:
    return [c for c in provider.calls if len(c) == 7]


# ---------------------------------------------------------------------------
# C15a: the record.
# ---------------------------------------------------------------------------


def test_c15a_a_budget_miss_records_the_query_and_its_first_five(clock: _Clock) -> None:
    reranker_mod._record_rerank_cost(_MODEL, 100, 1_000.0, None)  # inflated
    provider = _true_cost_provider(clock, 1.2)
    docs = _docs(50, _L)
    out = _message(provider, docs=docs)
    assert out.hand_off == "budget"
    miss = reranker_mod._budget_misses[_MODEL]
    assert miss.query == _QUERY
    assert list(miss.documents) == docs[:5]
    assert list(miss.candidate_tokens) == [_L] * 5
    assert list(miss.anchor_tokens) == [len(_QUERY) + len(a) for a in reranker_mod.ANCHOR_POOL]


def test_c15a_pool_allocation_and_sizes_hand_offs_write_no_record(clock: _Clock) -> None:
    reranker_mod._record_rerank_cost(_MODEL, 100, 1_000.0, None)
    provider = _true_cost_provider(clock, 1.2)
    assert rerank_for_recall(provider, _QUERY, _docs(4, _L), budget_seconds=_B).hand_off == "pool"
    assert (
        rerank_for_recall(provider, _QUERY, _docs(50, _L), budget_seconds=_B, max_real=4).hand_off
        == "allocation"
    )

    class _NoSizes(reranker_mod.FakeRerankerProvider):
        def pair_token_lengths(self, query, documents):
            return None

        def model_id(self) -> str:
            return _MODEL

    no_sizes = _NoSizes(default=0.0)
    assert rerank_for_recall(no_sizes, _QUERY, _docs(50, _L), budget_seconds=_B).hand_off == "sizes"
    assert _MODEL not in reranker_mod._budget_misses


# ---------------------------------------------------------------------------
# C15b: stuck is judged on the FULL budget, not the query's own share.
# ---------------------------------------------------------------------------


def _healthy_estimate(provider) -> None:
    """Two measured sizes of the true cost (overhead 0, the provider's rate)."""
    for x in (_X5, 2 * _X5):
        reranker_mod._record_rerank_cost(_MODEL, x, provider.rate * x, None)


def test_c15b_a_small_share_miss_on_a_healthy_model_is_not_stuck(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(i)/(ii): a paragraph share, or a message remainder after a long embed,
    too small for 5 while 5 fit the full budget: recorded, not stuck, not
    due, and the record is dropped by the due check."""
    provider = _true_cost_provider(clock, 1.2)
    _cached(monkeypatch, provider)
    _healthy_estimate(provider)
    for share in (_B / 4, 0.3):
        out = _message(provider, budget=share)
        assert out.hand_off == "budget"
        assert _MODEL in reranker_mod._budget_misses
        assert reranker_mod.stuck_width_due(1.0) is False
        assert _MODEL not in reranker_mod._budget_misses


def test_c15b_an_inflated_estimate_is_stuck(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _true_cost_provider(clock, 1.2)
    _cached(monkeypatch, provider)
    reranker_mod._record_rerank_cost(_MODEL, _X5, 40 * 1.2, None)
    assert _message(provider).hand_off == "budget"
    assert reranker_mod.stuck_width_due(1.0) is True


# ---------------------------------------------------------------------------
# C15c (a): a recall rerank within the full budget clears; over it does not.
# ---------------------------------------------------------------------------


def test_c15c_a_a_within_budget_rerank_clears_and_an_over_budget_one_does_not(
    clock: _Clock,
) -> None:
    fast = _true_cost_provider(clock, 1.2)
    _healthy_estimate(fast)
    miss = reranker_mod._BudgetMiss("old", ("a",) * 5, (_L,) * 5, (10,) * 8)
    reranker_mod._budget_misses[_MODEL] = miss
    # A measured rerank that takes MORE than B (the batch ran long this time).
    slow = _CostScriptedProvider(clock, overhead=_B + 1.0, rate=fast.rate)
    out = _message(slow)
    assert out.reranked and out.measured
    assert reranker_mod._budget_misses.get(_MODEL) is miss, "over-budget rerank proves nothing"
    out = _message(fast)
    assert out.reranked and out.measured
    assert _MODEL not in reranker_mod._budget_misses, "S116 (a): 5+ ran within B"


# ---------------------------------------------------------------------------
# C15d: due.
# ---------------------------------------------------------------------------


def _stuck(clock: _Clock, monkeypatch: pytest.MonkeyPatch, seconds_for_five: float = 1.2):
    provider = _true_cost_provider(clock, seconds_for_five)
    _cached(monkeypatch, provider)
    reranker_mod._record_rerank_cost(_MODEL, _X5, 40 * 1.2, None)
    assert _message(provider).hand_off == "budget"
    return provider


def test_c15d_due_needs_a_warmed_cached_provider(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _stuck(clock, monkeypatch)
    assert reranker_mod.stuck_width_due(1.0)
    monkeypatch.delitem(reranker_mod._provider_cache, _MODEL)
    assert not reranker_mod.stuck_width_due(1.0), "no cached provider: never builds one"
    _cached(monkeypatch, provider)
    with reranker_mod._cost_lock:
        reranker_mod._warmed_model_ids.discard(_MODEL)
    assert not reranker_mod.stuck_width_due(1.0), "not warmed: the session has not run"


def test_c15d_conc9_once_per_lull_per_model_id(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    # The estimate stays stuck after the run (one 1.2 s sample against a 40x
    # inflated one), and the run is within the budget (no "too long" verdict,
    # S127), so only the marker limits the next run.
    provider = _stuck(clock, monkeypatch)
    marker = float("-inf")  # a process that has seen no message
    assert reranker_mod.stuck_width_due(marker)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    # A run at the -inf marker (the record is cleared: 5 fit) ...
    assert reranker_mod.remeasure_stuck_width(marker)
    assert not reranker_mod._too_long_remembered(_MODEL)
    # ... then a tool call records a NEW miss without moving the marker.
    assert _message(provider).hand_off == "budget"
    assert not reranker_mod.stuck_width_due(marker), "one re-measure per lull, per model id"
    assert reranker_mod.stuck_width_due(5.0), "the marker moved: due again"


def test_c15d_the_record_key_is_the_cache_key_through_the_real_builder(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N3-6: `build_reranker_provider` caches the provider under the resolved
    model id and the provider reports that id, so a budget miss recorded by
    `rerank_for_recall` is the one the job finds (CrossEncoderProvider stubbed:
    no model files)."""
    monkeypatch.setattr(reranker_mod, "_register_fp16_reranker_model", lambda *a: None)
    built: list[str] = []

    def _stub(model_id, cache_dir):
        built.append(model_id)
        return _CostScriptedProvider(clock, 0.0, 1.2 / _X5, model=model_id)

    monkeypatch.setattr(reranker_mod, "CrossEncoderProvider", _stub)
    monkeypatch.setattr(reranker_mod, "_provider_cache", {})
    # The autouse conftest fixture replaced `build_reranker_provider` with a
    # fake; take the production function back from monkeypatch's undo record.
    real_build = next(
        value
        for obj, name, value in monkeypatch._setattr
        if obj is reranker_mod and name == "build_reranker_provider"
    )
    provider = real_build()
    assert built == [_MODEL] and provider.model_id() == _MODEL
    reranker_mod._record_rerank_cost(_MODEL, _X5, 40 * 1.2, None)
    assert _message(provider).hand_off == "budget"
    assert reranker_mod.stuck_width_due(1.0)


# ---------------------------------------------------------------------------
# C15e: the run.
# ---------------------------------------------------------------------------


def test_c15e_one_rerank_of_five_real_plus_two_anchors_recorded_at_seven_times_longest(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _stuck(clock, monkeypatch)
    miss = reranker_mod._budget_misses[_MODEL]
    before = reranker_mod._cost_sums[_MODEL]
    provider.calls.clear()
    assert reranker_mod.remeasure_stuck_width(1.0)
    assert len(provider.calls) == 1
    assert provider.calls[0] == [*miss.documents, *reranker_mod.ANCHOR_POOL[:2]]
    after = reranker_mod._cost_sums[_MODEL]
    assert after.n == before.n + 1 and after.sum_x == before.sum_x + _X5
    assert after.sum_y - before.sum_y == pytest.approx(1.2)
    assert _MODEL not in reranker_mod._budget_misses, "1.2 s <= B: cleared (S116 a)"


def test_c15e_run_rejudges_first_and_skips_a_no_longer_stuck_record(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _stuck(clock, monkeypatch)
    assert reranker_mod.stuck_width_due(1.0)
    # Between the due check and the run, chat reranks change the estimate.
    _healthy_estimate(provider)
    for _ in range(60):
        reranker_mod._record_rerank_cost(_MODEL, 2 * _X5, provider.rate * 2 * _X5, None)
    provider.calls.clear()
    assert reranker_mod.remeasure_stuck_width(1.0) is False
    assert provider.calls == []
    assert _MODEL not in reranker_mod._budget_misses


def test_c15e_stamp_before_the_ram_check(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _true_cost_provider(clock, 1.2)
    _cached(monkeypatch, provider)
    # One inflated sample at a SMALL size with an RSS delta: a RAM figure whose
    # high water (100 tokens) is far below the job's 5 + 2 batch.
    reranker_mod._record_rerank_cost(_MODEL, 100, 40 * 1.2 * 100 / _X5, 1e6)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    assert _message(provider).hand_off == "budget"
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0)
    provider.calls.clear()
    assert reranker_mod.remeasure_stuck_width(7.0) is False
    assert provider.calls == [], "RAM-skipped: no rerank"
    assert _MODEL in reranker_mod._budget_misses, "RAM-skipped: record kept"
    assert not reranker_mod.stuck_width_due(7.0), "stamped: not due again at the same marker"
    assert reranker_mod.stuck_width_due(8.0)


# ---------------------------------------------------------------------------
# C15f: recovery in three regimes, plus the S116 (b) drop.
# ---------------------------------------------------------------------------


def _drive(provider, *, limit: int = 200) -> int:
    """Alternate a message and a lull until a message reranks; return the
    number of job re-measures that took."""
    runs = 0
    for lull in range(limit):
        out = _message(provider)
        if out.reranked:
            return runs
        assert out.hand_off == "budget"
        assert reranker_mod.stuck_width_due(float(lull))
        assert reranker_mod.remeasure_stuck_width(float(lull))
        assert _MODEL not in reranker_mod._budget_misses
        runs += 1
    raise AssertionError("never recovered")


def _first_fit(predict, limit: int = 1000) -> int:
    for k in range(limit):
        if predict(k) <= _B:
            return k
    raise AssertionError


def test_c15f_1_first_period_single_size(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    t = Fraction(3, 10) * Fraction(_B)
    provider = _true_cost_provider(clock, float(t))
    _cached(monkeypatch, provider)
    reranker_mod._record_rerank_cost(_MODEL, _X5, float(40 * t), None)

    def predict(k: int) -> Fraction:  # ratio of sums at one size
        return (40 * t + k * t) / (k + 1)

    expected = _first_fit(predict)
    assert expected == 16
    assert predict(15) / _B > Fraction(101, 100) and predict(16) / _B < Fraction(99, 100)
    assert _drive(provider) == expected
    assert not reranker_mod.stuck_width_due(99.0)


def test_c15f_2_first_period_two_sizes(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    """The inflated sample at x1 = 2 * _X5, the re-measures at _X5: least
    squares (or its fallback) over the actual points, independently."""
    o, t5 = 0.2, 1.0
    provider = _true_cost_provider(clock, t5, overhead=o)
    _cached(monkeypatch, provider)
    x1 = 2 * _X5
    y1 = 40 * (o + provider.rate * x1)
    reranker_mod._record_rerank_cost(_MODEL, x1, y1, None)

    def predict(k: int) -> float:
        points = [(x1, y1)] + [(_X5, t5)] * k
        if k == 0:
            return y1 / x1 * _X5
        a = np.array([[1.0, x] for x, _ in points])
        b = np.array([y for _, y in points])
        (oo, rr), *_ = np.linalg.lstsq(a, b, rcond=None)
        if oo < 0 or rr <= 0:
            return sum(y for _, y in points) / sum(x for x, _ in points) * _X5
        return oo + rr * _X5

    expected = _first_fit(predict)
    assert predict(expected - 1) / _B > 1.01 and predict(expected) / _B < 0.99
    assert _drive(provider) == expected


def test_c15f_3_after_a_reset_the_rescaled_carry_recovers_in_one(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _true_cost_provider(clock, 1.2)
    _cached(monkeypatch, provider)
    # An inflated fit at two sizes, carried into the next period.
    reranker_mod._record_rerank_cost(_MODEL, _X5, 40 * 1.2, None)
    reranker_mod._record_rerank_cost(_MODEL, 2 * _X5, 80 * 1.2, None)
    cost_period.start_new_period()
    assert _drive(provider) == 1
    assert rerank_cost_estimate(_MODEL).seconds_per_token * _X5 == pytest.approx(1.2)


def test_c15f_4_an_accurate_too_long_record_is_dropped_after_one_run(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _true_cost_provider(clock, 5.0)  # 5 real + 2 anchors truly take 5 s > B
    _cached(monkeypatch, provider)
    _healthy_estimate(provider)
    assert _message(provider).hand_off == "budget"
    assert reranker_mod.stuck_width_due(1.0)
    assert reranker_mod.remeasure_stuck_width(1.0)
    assert _MODEL not in reranker_mod._budget_misses, "S116 (b): too long, dropped"
    assert not reranker_mod.stuck_width_due(2.0)
    # S127: the verdict is remembered, so the next budget miss (a new record)
    # does not bring the job back; see the S127 tests below.
    assert _message(provider).hand_off == "budget"
    assert not reranker_mod.stuck_width_due(2.0)


# ---------------------------------------------------------------------------
# S127: a "too long" verdict is remembered until the fitted cost model changes.
# ---------------------------------------------------------------------------


def _too_long(clock: _Clock, monkeypatch: pytest.MonkeyPatch):
    """An accurate slow host: 5 + 2 truly take 5 s > B, the estimate agrees
    (two sizes at that rate), one job run reached the S116 (b) verdict."""
    provider = _true_cost_provider(clock, 5.0)
    _cached(monkeypatch, provider)
    _healthy_estimate(provider)
    assert _message(provider).hand_off == "budget"
    assert reranker_mod.stuck_width_due(1.0)
    assert reranker_mod.remeasure_stuck_width(1.0)
    return provider


def test_s127_new_budget_misses_do_not_re_run_the_over_budget_rerank(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    assert len(_seven_doc_calls(provider)) == 1, "the one measured 5 + 2 rerank"
    held = clock.t
    for lull in range(2, 8):
        assert _message(provider).hand_off == "budget", "5 still never fit"
        assert _MODEL in reranker_mod._budget_misses, "a new stuck record"
        assert reranker_mod.stuck_width_due(float(lull)) is False
        assert reranker_mod.remeasure_stuck_width(float(lull)) is False
    assert len(_seven_doc_calls(provider)) == 1, "no further over-budget rerank (no lock hold)"
    assert clock.t == held, "no rerank time spent at any later lull"


def test_s127_the_run_itself_skips_while_the_verdict_is_remembered(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard in the run, apart from the due check: a run reached another
    way does no rerank and does not stamp the once-per-lull marker."""
    provider = _too_long(clock, monkeypatch)
    assert _message(provider).hand_off == "budget"
    calls = len(provider.calls)
    stamp = reranker_mod._remeasured_marker[_MODEL]
    assert reranker_mod.remeasure_stuck_width(9.0) is False
    assert len(provider.calls) == calls
    assert reranker_mod._remeasured_marker[_MODEL] == stamp
    assert _MODEL in reranker_mod._budget_misses, "the new record is kept"


def test_s127_the_verdict_ends_when_the_fitted_cost_model_changes(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    assert _message(provider).hand_off == "budget"
    assert not reranker_mod.stuck_width_due(2.0)
    before = rerank_cost_estimate(_MODEL)
    # Any measurement that moves the fit (here a slower batch, so the model
    # still says 5 do not fit and the record is still stuck).
    reranker_mod._record_rerank_cost(_MODEL, 4 * _X5, 4 * 6.0, None)
    after = rerank_cost_estimate(_MODEL)
    assert (after.overhead_seconds, after.seconds_per_token) != (
        before.overhead_seconds,
        before.seconds_per_token,
    )
    assert not reranker_mod._too_long_remembered(_MODEL)
    assert reranker_mod.stuck_width_due(2.0) is True, "the estimate changed: the job may run again"
    calls = len(_seven_doc_calls(provider))
    assert reranker_mod.remeasure_stuck_width(2.0) is True
    assert len(_seven_doc_calls(provider)) == calls + 1


def test_s127_a_sample_that_leaves_the_fit_unchanged_keeps_the_verdict(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Changes" is the fitted (overhead, rate), not the sample count: at one
    batch size the fit is the ratio of sums, so one more sample of the same
    cost leaves it exactly as it was."""
    provider = _true_cost_provider(clock, 5.0)
    _cached(monkeypatch, provider)
    reranker_mod._record_rerank_cost(_MODEL, _X5, 5.0, None)  # one size, 5 s: stuck
    assert _message(provider).hand_off == "budget"
    assert reranker_mod.remeasure_stuck_width(1.0)  # measures 5.0 s > B
    assert reranker_mod._too_long_remembered(_MODEL)
    before = rerank_cost_estimate(_MODEL)
    reranker_mod._record_rerank_cost(_MODEL, _X5, 5.0, None)
    after = rerank_cost_estimate(_MODEL)
    assert after.measured_batches == before.measured_batches + 1
    assert (after.overhead_seconds, after.seconds_per_token) == (
        before.overhead_seconds,
        before.seconds_per_token,
    )
    assert reranker_mod._too_long_remembered(_MODEL)


def test_s127_the_verdict_survives_the_compaction_tick_and_ends_at_the_next_fit_change(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    cost_period.start_new_period()  # the 24-hour tick: the fit in effect is carried
    assert reranker_mod._too_long_remembered(_MODEL)
    assert _message(provider).hand_off == "budget"
    assert not reranker_mod.stuck_width_due(3.0)
    # The new period's first measurement (one size) rescales the carry: a change.
    reranker_mod._record_rerank_cost(_MODEL, _X5, 6.0, None)
    assert not reranker_mod._too_long_remembered(_MODEL)


def test_s127_the_verdict_is_reached_under_the_fit_that_includes_the_jobs_own_sample(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimate was inflated (40 x 1.2 s at one size) and the job's own
    measurement (5 s > B) moves the fit; the verdict is still remembered, because
    it is stored under the fit that includes that sample."""
    provider = _stuck(clock, monkeypatch, seconds_for_five=5.0)
    before = rerank_cost_estimate(_MODEL)
    assert reranker_mod.remeasure_stuck_width(1.0)
    after = rerank_cost_estimate(_MODEL)
    assert after.seconds_per_token != before.seconds_per_token, "the job's sample moved the fit"
    assert reranker_mod._too_long_remembered(_MODEL)
    assert _message(provider).hand_off == "budget", "still stuck under the new fit"
    assert reranker_mod.stuck_width_due(2.0) is False


def test_s127_a_within_budget_run_remembers_nothing(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stuck(clock, monkeypatch)  # the run measures 1.2 s <= B
    assert reranker_mod.remeasure_stuck_width(1.0)
    assert not reranker_mod._too_long_remembered(_MODEL)


def test_s127_a_run_measuring_exactly_the_budget_is_not_too_long(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equality at B fits (S116 a clears on `<=`); only over B is the verdict."""
    provider = _CostScriptedProvider(clock, overhead=_B, rate=0.0)  # 5 + 2 take exactly B
    _cached(monkeypatch, provider)
    reranker_mod._record_rerank_cost(_MODEL, _X5, 40 * _B, None)  # inflated: stuck
    assert _message(provider).hand_off == "budget"
    began = clock.t
    assert reranker_mod.remeasure_stuck_width(1.0)
    assert clock.t - began == pytest.approx(_B)
    assert not reranker_mod._too_long_remembered(_MODEL)
    assert _MODEL not in reranker_mod._budget_misses


def test_s127_a_model_id_with_no_verdict_is_unaffected(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verdict is per model id: another id's verdict does not block this one."""
    with reranker_mod._cost_lock:
        reranker_mod._too_long_fit["some-other-reranker"] = (0.0, 1.0)
    provider = _stuck(clock, monkeypatch)
    assert reranker_mod.stuck_width_due(1.0) is True
    assert reranker_mod.remeasure_stuck_width(1.0) is True
    assert provider is not None


def test_s127_the_test_reset_forgets_verdicts() -> None:
    with reranker_mod._cost_lock:
        reranker_mod._too_long_fit[_MODEL] = (0.0, 1.0)
    reranker_mod._reset_rerank_cost_model()
    assert reranker_mod._too_long_fit == {}


# ---------------------------------------------------------------------------
# CONC-8: the job's clear/drop is identity-checked.
# ---------------------------------------------------------------------------


def test_conc8_a_new_record_during_the_run_survives(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _stuck(clock, monkeypatch)
    newer = reranker_mod._BudgetMiss("newer", ("n",) * 5, (_L,) * 5, (10,) * 8)

    def _chat_records_a_miss() -> None:
        # A chat thread's budget miss lands between the job's sample and its drop.
        def _record() -> None:
            with reranker_mod._cost_lock:
                reranker_mod._budget_misses[_MODEL] = newer

        t = threading.Thread(target=_record)
        t.start()
        t.join(5.0)

    monkeypatch.setattr(reranker_mod, "_remeasure_hook", _chat_records_a_miss)
    assert reranker_mod.remeasure_stuck_width(1.0)
    assert reranker_mod._budget_misses.get(_MODEL) is newer
    assert provider is not None


# ---------------------------------------------------------------------------
# C15g: the job in the table; C16d: the compaction job resets the period.
# ---------------------------------------------------------------------------


def _job(tmp_path, name):
    return {j.name: j for j in _real_jobs(_persona(tmp_path))}[name]


def test_c15g_the_job_forwards_the_real_marker_and_respects_the_slot(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, object]] = []
    monkeypatch.setattr(supervisor.cli_throttle, "chat_activity_marker", lambda: 42.5)
    monkeypatch.setattr(
        supervisor.reranker, "stuck_width_due", lambda m: seen.append(("due", m)) or True
    )
    monkeypatch.setattr(
        supervisor.reranker, "remeasure_stuck_width", lambda m: seen.append(("run", m)) or True
    )
    job = _job(tmp_path, "rerank_width_remeasure")
    assert job.has_work() is True
    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    assert job.run() is JobOutcome.COMPLETED
    assert seen == [("due", 42.5), ("run", 42.5)]
    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _denying_slot)
    assert job.run() is JobOutcome.SKIPPED
    assert seen == [("due", 42.5), ("run", 42.5)], "slot lost: no rerank"


@pytest.mark.parametrize(
    ("tick", "outcome", "resets"),
    [
        (lambda *a, **k: False, JobOutcome.COMPLETED, 1),
        (lambda *a, **k: True, JobOutcome.PAUSED, 0),
    ],
)
def test_c16d_the_compaction_job_starts_a_new_period_unless_paused(
    tmp_path, monkeypatch: pytest.MonkeyPatch, tick, outcome, resets
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(supervisor.cost_period, "start_new_period", lambda: calls.append(1))
    monkeypatch.setattr(supervisor, "_run_compaction_tick", tick)
    monkeypatch.setattr("brain.chat.compaction.build_compaction_provider", lambda _pd: None)
    assert _job(tmp_path, "compaction").run() is outcome
    assert len(calls) == resets


@pytest.mark.parametrize("where", ["provider", "tick"])
def test_c16d_a_raising_compaction_job_still_starts_a_new_period(
    tmp_path, monkeypatch: pytest.MonkeyPatch, where
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(supervisor.cost_period, "start_new_period", lambda: calls.append(1))

    def boom(*_a, **_k):
        raise RuntimeError("compaction failed")

    if where == "provider":
        monkeypatch.setattr("brain.chat.compaction.build_compaction_provider", boom)
    else:
        monkeypatch.setattr("brain.chat.compaction.build_compaction_provider", lambda _pd: None)
        monkeypatch.setattr(supervisor, "_run_compaction_tick", boom)
    with pytest.raises(RuntimeError):
        _job(tmp_path, "compaction").run()
    assert calls == [1]


def test_c16d_only_the_compaction_job_calls_start_new_period() -> None:
    """So a `nell chat --no-bridge` process (no supervisor) keeps its
    process-lifetime averages."""
    callers = []
    for path in sorted((_REPO / "brain").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "start_new_period"
            ):
                callers.append(path.relative_to(_REPO).as_posix())
    assert callers == ["brain/bridge/supervisor.py"]


# ---------------------------------------------------------------------------
# C15h: the NellFace label (placeholder).
# ---------------------------------------------------------------------------


def test_c15h_job_descriptor_is_a_marked_placeholder_without_em_dash() -> None:
    lines = (_REPO / "app" / "src" / "App.tsx").read_text(encoding="utf-8").splitlines()
    i = next(n for n, line in enumerate(lines) if "rerank_width_remeasure:" in line)
    assert lines[i].strip() == 'rerank_width_remeasure: "re-scoring cost check",'
    assert "PLACEHOLDER" in lines[i - 1]
    assert "—" not in lines[i] and "—" not in lines[i - 1]
