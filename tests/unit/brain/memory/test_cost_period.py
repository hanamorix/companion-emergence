"""Name-recall fix R1-recovery: cost series averaged per 24-hour compaction
period (spec §1 / §7; S101, S114, S118; criteria C16a-C16e, CONC-7, ADV-13 in
`changes/name-recall-fix/R1-recovery/1.5-criteria.md`).

Oracles are independent: numpy least squares (`np.linalg.lstsq`) or values
derived by hand in the test, never the implementation's own formula.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import brain.memory.cost_period as cp
import brain.memory.reranker as reranker_mod
from brain.memory.cost_period import LinearFit, PeriodCostSeries, PeriodSums, fit_linear


def _ols(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Independent (overhead, rate) by numpy least squares."""
    a = np.array([[1.0, float(x)] for x, _ in points])
    b = np.array([y for _, y in points])
    (overhead, rate), *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(overhead), float(rate)


def _state(points: list[tuple[int, float]], carried: LinearFit | None = None) -> PeriodSums:
    state = PeriodSums(carried=carried)
    for x, y in points:
        state = state.plus(x, y)
    return state


@pytest.fixture
def series():
    s = PeriodCostSeries("test series")
    yield s
    s.unregister()


# ---------------------------------------------------------------------------
# C16c: a first period (no carry) is a plain running fit, as R1.
# ---------------------------------------------------------------------------


def test_c16c_first_period_is_the_plain_least_squares_fit() -> None:
    points = [(1000, 0.7), (3000, 1.5), (2000, 1.1)]
    fit = _state(points).fit()
    overhead, rate = _ols(points)
    assert fit is not None and fit.samples == 3
    assert fit.overhead == pytest.approx(overhead) and fit.rate == pytest.approx(rate)
    one = _state([(1000, 0.5), (1000, 0.7)]).fit()
    assert one == LinearFit(0.0, 1.2 / 2000, 2), "one size, no carry: the ratio of sums"
    assert PeriodSums().fit() is None


# ---------------------------------------------------------------------------
# C16a / C16b: reset carries the fit in effect; one size rescales it (S114);
# two sizes hand over (S101).
# ---------------------------------------------------------------------------


def test_c16b_carry_then_rescale_at_one_size_then_new_fit_at_two_sizes() -> None:
    old = _state([(1000, 0.3), (3000, 0.7)])  # 0.1 + 0.0002 x
    carried = old.fit()
    assert carried is not None
    new = old.next_period()
    assert new.sums.n == 0
    assert new.fit() == carried, "no sample yet: the carried fit"

    # One sample at x0 = 2000: the carried fit predicts 0.5; the sample is 1.0.
    one = new.plus(2000, 1.0)
    fit = one.fit()
    assert fit is not None
    assert fit.predict(2000) == pytest.approx(1.0), "the rescaled prediction matches the sample"
    assert fit.overhead == pytest.approx(0.2) and fit.rate == pytest.approx(0.0004), (
        "both parameters scaled by one factor (1.0 / 0.5 = 2)"
    )
    # Several samples at the same size: matches their mean.
    three = one.plus(2000, 1.2).plus(2000, 1.4)
    assert three.fit().predict(2000) == pytest.approx(1.2)
    assert three.fit().rate / carried.rate == pytest.approx(1.2 / 0.5)

    # A second distinct size: the new period's own least squares.
    two = three.plus(4000, 2.0)
    points = [(2000, 1.0), (2000, 1.2), (2000, 1.4), (4000, 2.0)]
    overhead, rate = _ols(points)
    assert two.fit().overhead == pytest.approx(overhead)
    assert two.fit().rate == pytest.approx(rate)


def test_c16b_carry_waits_for_two_sizes_not_two_samples() -> None:
    carried = LinearFit(0.1, 0.0002, 9)
    state = PeriodSums(carried=carried).plus(2000, 1.0).plus(2000, 1.0)
    assert state.fit().predict(2000) == pytest.approx(1.0)
    assert state.fit().predict(4000) == pytest.approx(1.0 * (0.1 + 0.8) / 0.5), (
        "still the carried SHAPE, rescaled, after two samples of one size"
    )


def test_c16b_a_rescaled_carry_is_carried_again_by_the_next_tick() -> None:
    state = PeriodSums(carried=LinearFit(0.1, 0.0002, 9)).plus(2000, 1.0)
    in_effect = state.fit()
    assert state.next_period().carried == in_effect


def test_c16b_no_usable_factor_falls_back_to_the_new_ratio() -> None:
    state = PeriodSums(carried=LinearFit(0.0, 0.0, 1)).plus(1000, 0.5)
    assert state.fit() == LinearFit(0.0, 0.5 / 1000, 1)


def test_fit_linear_degenerate_cases() -> None:
    assert fit_linear(cp.LinearSums()) is None
    # A negative fitted overhead falls back to the ratio of sums.
    sums = cp.LinearSums().plus(1000, 0.1).plus(2000, 1.0)
    fit = fit_linear(sums)
    assert fit == LinearFit(0.0, 1.1 / 3000, 2)


# ---------------------------------------------------------------------------
# C16a on the reranker: the time sums reset and carry; RAM and warm-up kept.
# ---------------------------------------------------------------------------


def test_c16a_reranker_period_resets_time_sums_keeps_ram_and_warmup() -> None:
    model = "period-model"
    reranker_mod._record_rerank_cost(model, 1000, 0.3, 4096.0)
    reranker_mod._record_rerank_cost(model, 3000, 0.7, 8192.0)
    with reranker_mod._cost_lock:
        reranker_mod._warmed_model_ids.add(model)
    before = reranker_mod._cost_sums[model]
    estimate_before = reranker_mod.rerank_cost_estimate(model)

    cp.start_new_period()

    after = reranker_mod._cost_sums[model]
    assert (after.n, after.sum_x, after.sum_y, after.sum_xx) == (0, 0, 0.0, 0)
    assert (after.peak_x, after.ram_bytes, after.ram_x) == (
        before.peak_x,
        before.ram_bytes,
        before.ram_x,
    )
    assert model in reranker_mod._warmed_model_ids
    estimate_after = reranker_mod.rerank_cost_estimate(model)
    assert estimate_after == estimate_before, "the fit in effect is carried, RAM figure unchanged"


def test_c16a_c16e_every_registered_series_resets_on_one_call(series: PeriodCostSeries) -> None:
    second = PeriodCostSeries("stand-in for the S103 per-word cost")
    try:
        series.record("m", 100, 1.0)
        second.record("k", 10, 0.5)
        cp.start_new_period()
        assert series.state("m").sums.n == 0 and series.fit("m") == LinearFit(0.0, 0.01, 1)
        assert second.state("k").sums.n == 0 and second.fit("k") == LinearFit(0.0, 0.05, 1)
    finally:
        second.unregister()
    assert second.new_period not in cp._hooks


def test_c16e_a_hook_may_unregister_itself_while_called() -> None:
    calls: list[str] = []
    unregister = None

    def hook() -> None:
        calls.append("ran")
        unregister()

    unregister = cp.on_new_period(hook)
    done = threading.Event()
    threading.Thread(target=lambda: (cp.start_new_period(), done.set()), daemon=True).start()
    assert done.wait(5.0), "no deadlock: hooks run outside the list's lock"
    assert calls == ["ran"] and hook not in cp._hooks


def test_start_new_period_isolates_a_failing_hook(series: PeriodCostSeries) -> None:
    def boom() -> None:
        raise RuntimeError("series broke")

    unregister = cp.on_new_period(boom)
    try:
        series.record("m", 100, 1.0)
        cp.start_new_period()
        assert series.state("m").sums.n == 0
    finally:
        unregister()


# ---------------------------------------------------------------------------
# CONC-7: a reset injected between a record's read and write waits for it.
# ---------------------------------------------------------------------------


def _reset_during_record(monkeypatch, *, hook_target, record, read_new_n) -> tuple[int, bool]:
    inside = threading.Event()
    reset_done = threading.Event()

    def _hook() -> None:
        monkeypatch.setattr(*hook_target, None)
        inside.set()
        # The reset thread must not finish while we hold the guarded window.
        reset_done.wait(0.5)

    monkeypatch.setattr(*hook_target, _hook)
    t = threading.Thread(target=lambda: (inside.wait(5.0), cp.start_new_period(), reset_done.set()))
    t.start()
    record()
    t.join(5.0)
    return read_new_n(), reset_done.is_set()


def test_conc7_reset_waits_for_an_in_flight_series_record(
    monkeypatch: pytest.MonkeyPatch, series: PeriodCostSeries
) -> None:
    series.record("m", 100, 1.0)
    n, done = _reset_during_record(
        monkeypatch,
        hook_target=(cp, "_update_hook"),
        record=lambda: series.record("m", 300, 2.0),
        read_new_n=lambda: series.state("m").sums.n,
    )
    assert done
    assert n == 0, "the reset ran after the record: the new period holds no sample"
    carried = series.state("m").carried
    assert carried is not None and carried.samples == 2, "the carried fit includes the record"


def test_conc7_reset_waits_for_an_in_flight_rerank_record(monkeypatch: pytest.MonkeyPatch) -> None:
    model = "conc7-model"
    reranker_mod._record_rerank_cost(model, 100, 1.0, None)
    n, done = _reset_during_record(
        monkeypatch,
        hook_target=(reranker_mod, "_cost_update_hook"),
        record=lambda: reranker_mod._record_rerank_cost(model, 300, 2.0, None),
        read_new_n=lambda: reranker_mod._cost_sums[model].n,
    )
    assert done
    assert n == 0
    assert reranker_mod._cost_sums[model].period.carried.samples == 2


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_conc7_able_to_fail_without_the_series_lock(
    monkeypatch: pytest.MonkeyPatch, series: PeriodCostSeries
) -> None:
    monkeypatch.setattr(series, "_lock", _NoLock())
    series.record("m", 100, 1.0)
    n, _ = _reset_during_record(
        monkeypatch,
        hook_target=(cp, "_update_hook"),
        record=lambda: series.record("m", 300, 2.0),
        read_new_n=lambda: series.state("m").sums.n,
    )
    assert n == 2, "unguarded: the old period's sums survive the reset (the defect CONC-7 guards)"


# ---------------------------------------------------------------------------
# ADV-13 (advisory, owner FYI): a new period's first two-size fit from two
# samples at nearly equal sizes.
# ---------------------------------------------------------------------------


def test_adv13_first_two_size_fit_from_near_equal_sizes_is_noise_driven() -> None:
    """Demonstration, not a requirement. True cost 0.2 + 0.0004 x; the new
    period's first samples at 1000 and 1003 tokens carry +-2% noise. The fit
    that takes over at two sizes extrapolates badly; S101 hands over anyway
    and any spread guard would be a new constant (I3)."""
    carried = LinearFit(0.2, 0.0004, 40)
    state = PeriodSums(carried=carried).plus(1000, 0.6 * 1.02).plus(1003, 0.6012 * 0.98)
    fit = state.fit()
    truth_at_7000 = 0.2 + 0.0004 * 7000
    predicted = fit.predict(7000)
    # The two points give a negative fitted rate, so the ratio-of-sums fallback
    # (no overhead) takes over: about 4.2 s predicted for a 3.0 s batch, where
    # the carried fit predicted it exactly.
    assert fit.overhead == 0.0
    assert (predicted - truth_at_7000) / truth_at_7000 > 0.3
    assert carried.predict(7000) == pytest.approx(truth_at_7000)
