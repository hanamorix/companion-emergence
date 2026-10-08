"""Name-recall fix R1-recovery: stuck-width recovery at a chat lull (spec §7;
S100, S113, S116; criteria C15a-C15h, C16d, CONC-8, CONC-9 in
`changes/name-recall-fix/R1-recovery/1.5-criteria.md`), on the cost-series
increment's saved 24-hour window (S127 by sample identity, CS5).

All offline: the scripted clock and cost-scripted fake reranker of
`test_rerank_cost_model.py` (a call costs overhead + rate x padded tokens; one
token per character), the runtime model id patched to the fake's, the full
latency budget patched through `tunables.get_tunable`. No model, no network.
Recovery counts are checked against independent computations with
`fractions.Fraction`.
"""

from __future__ import annotations

import ast
import sqlite3
import threading
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import brain.memory.reranker as reranker_mod
from brain import tunables
from brain.bridge import supervisor
from brain.bridge.central_cadence import JobOutcome
from brain.memory import cost_series
from brain.memory.store import MemoryStore
from tests.unit.brain.bridge.test_central_cadence import (
    _denying_slot,
    _granting_slot,
    _persona,
    _real_jobs,
)
from tests.unit.brain.memory.test_rerank_cost_model import (
    _MODEL,
    _QUERY,
    _anchor_pair_tokens,
    _Clock,
    _CostScriptedProvider,
    _docs,
    _GrowingProvider,
    _reference_width,
)

_B = 4.0
# Pair size of every test document: longer than every anchor pair, so the
# longest pair of a 5 + 2 batch is a candidate and x = 7 * _L.
_L = 600
_X5 = 7 * _L
_REPO = Path(__file__).resolve().parents[4]

# The store this test's recall calls and re-measure runs use (an autouse fixture
# opens one per test) and a stepping wall clock, so no two samples share a
# `taken_at` and an "after the verdict" sample is strictly after it.
_STORE: MemoryStore


@pytest.fixture(autouse=True)
def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    global _STORE
    _STORE = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    now = [1_000_000.0]

    def _wall() -> float:
        now[0] += 0.001
        return now[0]

    monkeypatch.setattr(cost_series, "_wall_clock", _wall)
    yield _STORE
    _STORE.close()


def rerank_cost_estimate(model_id):
    return reranker_mod.rerank_cost_estimate(model_id, _STORE)


def _due(marker) -> bool:
    return reranker_mod.stuck_width_due(marker, lambda: nullcontext(_STORE))


def _remeasure(marker) -> bool:
    return reranker_mod.remeasure_stuck_width(marker, _STORE)


def _verdict() -> bool:
    return reranker_mod._too_long_remembered(_MODEL, _STORE)


def _window() -> cost_series.LinearSums:
    state = reranker_mod._rerank_cost.state(_STORE, _MODEL)
    assert state is not None
    return state.sums


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
    return reranker_mod.rerank_for_recall(
        provider, _QUERY, docs or _docs(50, _L), budget_seconds=budget, store=_STORE
    )


def _seven_doc_calls(provider) -> list[list[str]]:
    return [c for c in provider.calls if len(c) == 7]


# ---------------------------------------------------------------------------
# C15a: the record.
# ---------------------------------------------------------------------------


def test_c15a_a_budget_miss_records_the_query_and_its_first_five(clock: _Clock) -> None:
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 1_000.0, None)  # inflated
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
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 1_000.0, None)
    provider = _true_cost_provider(clock, 1.2)
    assert _message(provider, docs=_docs(4, _L)).hand_off == "pool"
    assert (
        reranker_mod.rerank_for_recall(
            provider, _QUERY, _docs(50, _L), budget_seconds=_B, max_real=4, store=_STORE
        ).hand_off
        == "allocation"
    )

    class _NoSizes(reranker_mod.FakeRerankerProvider):
        def pair_token_lengths(self, query, documents):
            return None

        def model_id(self) -> str:
            return _MODEL

    no_sizes = _NoSizes(default=0.0)
    assert _message(no_sizes).hand_off == "sizes"
    assert _MODEL not in reranker_mod._budget_misses


# ---------------------------------------------------------------------------
# C15b: stuck is judged on the FULL budget, not the query's own share.
# ---------------------------------------------------------------------------


def _healthy_estimate(provider) -> None:
    """Two measured sizes of the true cost (overhead 0, the provider's rate)."""
    for x in (_X5, 2 * _X5):
        reranker_mod._record_rerank_cost(_MODEL, _STORE, x, provider.rate * x, None)


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
        assert _due(1.0) is False
        assert _MODEL not in reranker_mod._budget_misses


def test_c15b_an_inflated_estimate_is_stuck(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _true_cost_provider(clock, 1.2)
    _cached(monkeypatch, provider)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, _X5, 40 * 1.2, None)
    assert _message(provider).hand_off == "budget"
    assert _due(1.0) is True


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
    reranker_mod._record_rerank_cost(_MODEL, _STORE, _X5, 40 * 1.2, None)
    assert _message(provider).hand_off == "budget"
    return provider


def test_c15d_due_needs_a_warmed_cached_provider(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _stuck(clock, monkeypatch)
    assert _due(1.0)
    monkeypatch.delitem(reranker_mod._provider_cache, _MODEL)
    assert not _due(1.0), "no cached provider: never builds one"
    _cached(monkeypatch, provider)
    with reranker_mod._cost_lock:
        reranker_mod._warmed_model_ids.discard(_MODEL)
    assert not _due(1.0), "not warmed: the session has not run"


def test_c15d_conc9_once_per_lull_per_model_id(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    # The estimate stays stuck after the run (one 1.2 s sample against a 40x
    # inflated one), and the run is within the budget (no "too long" verdict,
    # S127), so only the marker limits the next run.
    provider = _stuck(clock, monkeypatch)
    marker = float("-inf")  # a process that has seen no message
    assert _due(marker)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    # A run at the -inf marker (the record is cleared: 5 fit) ...
    assert _remeasure(marker)
    assert not _verdict()
    # ... then a tool call records a NEW miss without moving the marker.
    assert _message(provider).hand_off == "budget"
    assert not _due(marker), "one re-measure per lull, per model id"
    assert _due(5.0), "the marker moved: due again"


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
    reranker_mod._record_rerank_cost(_MODEL, _STORE, _X5, 40 * 1.2, None)
    assert _message(provider).hand_off == "budget"
    assert _due(1.0)


# ---------------------------------------------------------------------------
# C15e: the run.
# ---------------------------------------------------------------------------


def test_c15e_one_rerank_of_five_real_plus_two_anchors_recorded_at_seven_times_longest(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _stuck(clock, monkeypatch)
    miss = reranker_mod._budget_misses[_MODEL]
    before = _window()
    provider.calls.clear()
    assert _remeasure(1.0)
    assert len(provider.calls) == 1
    assert provider.calls[0] == [*miss.documents, *reranker_mod.ANCHOR_POOL[:2]]
    after = _window()
    assert after.n == before.n + 1 and after.sum_x == before.sum_x + _X5
    assert after.sum_y - before.sum_y == pytest.approx(1.2)
    assert _MODEL not in reranker_mod._budget_misses, "1.2 s <= B: cleared (S116 a)"


def test_c15e_run_rejudges_first_and_skips_a_no_longer_stuck_record(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _stuck(clock, monkeypatch)
    assert _due(1.0)
    # Between the due check and the run, chat reranks change the estimate.
    _healthy_estimate(provider)
    for _ in range(60):
        reranker_mod._record_rerank_cost(_MODEL, _STORE, 2 * _X5, provider.rate * 2 * _X5, None)
    provider.calls.clear()
    assert _remeasure(1.0) is False
    assert provider.calls == []
    assert _MODEL not in reranker_mod._budget_misses


def test_c15e_stamp_before_the_ram_check(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _true_cost_provider(clock, 1.2)
    _cached(monkeypatch, provider)
    # One inflated sample at a SMALL size with an RSS delta: a RAM figure whose
    # high water (100 tokens) is far below the job's 5 + 2 batch.
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 40 * 1.2 * 100 / _X5, 1e6)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    assert _message(provider).hand_off == "budget"
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0)
    provider.calls.clear()
    assert _remeasure(7.0) is False
    assert provider.calls == [], "RAM-skipped: no rerank"
    assert _MODEL in reranker_mod._budget_misses, "RAM-skipped: record kept"
    assert not _due(7.0), "stamped: not due again at the same marker"
    assert _due(8.0)


def test_s178_a_too_high_ram_figure_skips_the_job_until_its_samples_age_out_then_the_job_runs(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1d / K2c: the RAM bound skips the 5-candidate re-measure while the (too-high) figure's sample
    is in the window; once it has aged out this running process KEEPS the figure (S180) but it is
    stale, so the bound no longer skips the job: it runs and records the RAM sample that re-seeds
    the figure."""
    provider = _true_cost_provider(clock, 1.2)
    _cached(monkeypatch, provider)
    # An inflated time sample and a RAM figure of 10,000 bytes a token at a small high water.
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 40 * 1.2 * 100 / _X5, 1e6)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    assert _message(provider).hand_off == "budget"
    assert cost_series.flush(_STORE) >= 1
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0)
    assert rerank_cost_estimate(_MODEL).ram_bytes_per_token == pytest.approx(1e4)
    provider.calls.clear()
    assert _remeasure(7.0) is False and provider.calls == [], "the too-high figure skips the job"
    assert _MODEL in reranker_mod._budget_misses

    real = cost_series._wall_clock
    shift = [0.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: real() + shift[0])
    shift[0] = cost_series.window_seconds() + 60.0  # every sample is past the window
    est = rerank_cost_estimate(_MODEL)
    assert est is not None and est.ram_figure_kept is True, "the running process keeps its last figure; the carried time fit still says stuck"
    assert est.ram_bytes_per_token == pytest.approx(1e4)
    rss = iter(float(1_000_000 * n) for n in range(10_000))
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: next(rss))
    assert _due(8.0) is True
    assert _remeasure(8.0) is True
    assert len(_seven_doc_calls(provider)) == 1, "the 5 + 2 measurement ran"
    state = reranker_mod._ram_cost.state(_STORE, _MODEL)
    assert state is not None and state.sums.n == 1, "its RAM sample re-seeded the figure"
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token is not None and est.ram_figure_kept is False


# ---------------------------------------------------------------------------
# S180 (K2): a running process keeps its last RAM figure when the window empties; the lull job
# re-seeds it when it blocks a batch.
# ---------------------------------------------------------------------------

_BIG = 2_000  # pair size of the "record-high" batch: 5 + 2 of them are past the process's high water


def _running_with_a_too_high_figure(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, *, per_token: float = 500.0
) -> tuple[_GrowingProvider, list[float], int]:
    """A running process (it has recorded measured reranks, high water 2 * _X5) whose saved RAM figure is
    10,000 bytes a token (too high for a 1 MB headroom), the figure seen by an estimate call while its
    sample is in the window, and the time series saved. Returns the provider, the scripted wall clock and
    the high water."""
    wall, rss = [1_000_000.0], [1.0e9]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: rss[0])
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0e6)
    provider = _GrowingProvider(clock, rss, per_token=per_token)
    _cached(monkeypatch, provider)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, provider.rate * 100, 1.0e6)  # 10,000 B/token
    _healthy_estimate(provider)  # the time estimate (two sizes); the high water is now 2 * _X5
    assert rerank_cost_estimate(_MODEL).ram_bytes_per_token == pytest.approx(1.0e4)
    assert cost_series.flush(_STORE) >= 1
    return provider, wall, reranker_mod._ram_state[_MODEL].peak_x


def _age_out_the_window(wall: list[float]) -> None:
    wall[0] += cost_series.window_seconds() + 60.0


def test_s180_k2a_a_running_process_whose_kept_figure_blocks_a_batch_re_seeds_it_at_the_next_lull(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, wall, peak = _running_with_a_too_high_figure(clock, monkeypatch)
    assert peak == 2 * _X5
    _age_out_the_window(wall)
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_figure_kept is True and est.ram_bytes_per_token == pytest.approx(1.0e4)

    blocked = _message(provider, docs=_docs(50, _BIG))
    assert blocked.reranked is False and blocked.hand_off == "budget", "the kept figure blocks the batch (not vacuous)"
    assert [c for c in provider.calls if len(c) == 7] == [], "no measured rerank ran: the bound held"
    miss = reranker_mod._budget_misses[_MODEL]
    assert _due(1.0) is True, "the kept figure blocks the miss: the lull job is due"

    provider.calls.clear()
    assert _remeasure(1.0) is True
    assert len(provider.calls) == 1 and len(provider.calls[0]) == 7, "one 5 + 2 measurement, past the RAM-headroom skip"
    padded = 7 * _BIG
    assert padded > peak, "the blocked batch is past the high water, so it records a sample"
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_figure_kept is False and est.ram_bytes_per_token == pytest.approx(500.0 * padded / (padded - peak))
    assert _MODEL not in reranker_mod._budget_misses, "5 fit within the budget: cleared (S116 a)"
    assert reranker_mod._remeasured_marker[_MODEL] == 1.0, "stamped: once per lull"
    with reranker_mod._cost_lock:
        reranker_mod._budget_misses[_MODEL] = miss
    assert _due(1.0) is False, "the same lull: not due again"

    again = _message(provider, docs=_docs(50, _BIG))
    est = rerank_cost_estimate(_MODEL)
    assert again.reranked and again.width == _reference_width(
        [_BIG] * 50, _anchor_pair_tokens(), est.overhead_seconds, est.seconds_per_token, _B,
        ram_per_token=est.ram_bytes_per_token, headroom=1.0e6, peak=padded,
    ), "the next recall is sized from the new figure"


def test_s180_k2b_a_figure_still_in_the_window_is_a_legitimate_bound_and_does_not_trigger_the_job(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _wall, _peak = _running_with_a_too_high_figure(clock, monkeypatch)
    blocked = _message(provider, docs=_docs(50, _BIG))
    assert blocked.hand_off == "budget" and _MODEL in reranker_mod._budget_misses
    assert rerank_cost_estimate(_MODEL).ram_figure_kept is False
    provider.calls.clear()
    assert _due(1.0) is False and _MODEL not in reranker_mod._budget_misses, "dropped as not stuck, as at base"
    assert _remeasure(1.0) is False and provider.calls == []


def test_s180_k2b_a_kept_figure_that_blocks_nothing_does_not_trigger_the_job(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, wall, _peak = _running_with_a_too_high_figure(clock, monkeypatch)
    _age_out_the_window(wall)
    assert rerank_cost_estimate(_MODEL).ram_figure_kept is True
    small = _message(provider, budget=0.2)  # a message remainder too small for 5, but the batch is inside the high water
    assert small.hand_off == "budget" and _MODEL in reranker_mod._budget_misses
    provider.calls.clear()
    assert _due(1.0) is False and _MODEL not in reranker_mod._budget_misses
    assert _remeasure(2.0) is False and provider.calls == []


def test_s180_k2d_a_re_seed_that_measures_no_growth_leaves_no_figure_and_is_not_repeated(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, wall, _peak = _running_with_a_too_high_figure(clock, monkeypatch, per_token=0.0)
    _age_out_the_window(wall)
    assert _message(provider, docs=_docs(50, _BIG)).hand_off == "budget"
    assert _due(1.0) and _remeasure(1.0)
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token == 0.0 and est.ram_figure_kept is False
    assert reranker_mod._ram_kept == {}, "a window rate of 0 forgets the stale figure"
    provider.calls.clear()
    assert _message(provider, docs=_docs(50, _BIG)).reranked, "no figure to block it any more"
    assert _due(2.0) is False and _MODEL not in reranker_mod._budget_misses


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
        assert _due(float(lull))
        assert _remeasure(float(lull))
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
    reranker_mod._record_rerank_cost(_MODEL, _STORE, _X5, float(40 * t), None)

    def predict(k: int) -> Fraction:  # ratio of sums at one size
        return (40 * t + k * t) / (k + 1)

    expected = _first_fit(predict)
    assert expected == 16
    assert predict(15) / _B > Fraction(101, 100) and predict(16) / _B < Fraction(99, 100)
    assert _drive(provider) == expected
    assert not _due(99.0)


def test_c15f_2_first_period_two_sizes(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    """The inflated sample at x1 = 2 * _X5, the re-measures at _X5: least
    squares (or its fallback) over the actual points, independently."""
    o, t5 = 0.2, 1.0
    provider = _true_cost_provider(clock, t5, overhead=o)
    _cached(monkeypatch, provider)
    x1 = 2 * _X5
    y1 = 40 * (o + provider.rate * x1)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, x1, y1, None)

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


def test_c15f_3_after_the_window_empties_the_rescaled_carry_recovers_in_one(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _true_cost_provider(clock, 1.2)
    _cached(monkeypatch, provider)
    # An inflated fit at two sizes, saved; then a downtime longer than the
    # window leaves only the carried estimate (S146, S114).
    reranker_mod._record_rerank_cost(_MODEL, _STORE, _X5, 40 * 1.2, None)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 2 * _X5, 80 * 1.2, None)
    assert cost_series.flush(_STORE) == 2
    real = cost_series._wall_clock
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: real() + cost_series.window_seconds() + 60.0)
    assert _window().n == 0
    assert _drive(provider) == 1
    assert rerank_cost_estimate(_MODEL).seconds_per_token * _X5 == pytest.approx(1.2)


def test_c15f_4_an_accurate_too_long_record_is_dropped_after_one_run(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _true_cost_provider(clock, 5.0)  # 5 real + 2 anchors truly take 5 s > B
    _cached(monkeypatch, provider)
    _healthy_estimate(provider)
    assert _message(provider).hand_off == "budget"
    assert _due(1.0)
    assert _remeasure(1.0)
    assert _MODEL not in reranker_mod._budget_misses, "S116 (b): too long, dropped"
    assert not _due(2.0)
    # S127: the verdict is remembered, so the next budget miss (a new record)
    # does not bring the job back; see the S127 tests below.
    assert _message(provider).hand_off == "budget"
    assert not _due(2.0)


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
    assert _due(1.0)
    assert _remeasure(1.0)
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
        assert _due(float(lull)) is False
        assert _remeasure(float(lull)) is False
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
    assert _remeasure(9.0) is False
    assert len(provider.calls) == calls
    assert reranker_mod._remeasured_marker[_MODEL] == stamp
    assert _MODEL in reranker_mod._budget_misses, "the new record is kept"


def test_s127_the_verdict_ends_when_a_new_sample_arrives(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    assert _message(provider).hand_off == "budget"
    assert not _due(2.0)
    before = rerank_cost_estimate(_MODEL)
    # Any measurement (here a slower batch, so the model still says 5 do not
    # fit and the record is still stuck).
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 4 * _X5, 4 * 6.0, None)
    after = rerank_cost_estimate(_MODEL)
    assert (after.overhead_seconds, after.seconds_per_token) != (
        before.overhead_seconds,
        before.seconds_per_token,
    )
    assert not _verdict()
    assert _due(2.0) is True, "the estimate changed: the job may run again"
    calls = len(_seven_doc_calls(provider))
    assert _remeasure(2.0) is True
    assert len(_seven_doc_calls(provider)) == calls + 1


def test_s127_a_sample_flushed_by_another_connection_ends_the_verdict(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _too_long(clock, monkeypatch)
    assert _verdict()
    other = MemoryStore(_STORE.db_path, integrity_check=False)
    try:
        # What another process's flush writes (its buffer is not this one's: the
        # rows appear in the database only when it flushes).
        now = cost_series._wall_clock()
        other.record_cost_samples(
            "rerank", "test-machine", _MODEL, [(3 * _X5, 9.0, now)], now=now, window_seconds=cost_series.window_seconds()
        )
    finally:
        other.close()
    assert not _verdict()


def test_s127_a_sample_that_ages_out_ends_the_verdict(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _too_long(clock, monkeypatch)
    assert _verdict()
    real = cost_series._wall_clock
    shift = [0.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: real() + shift[0])
    shift[0] = 60.0
    assert _verdict(), "a time step that changes nothing keeps the verdict"
    shift[0] = cost_series.window_seconds() + 60.0
    assert not _verdict(), "the samples it counted left the window"


def test_s127_the_verdict_survives_its_own_flush_with_non_dyadic_seconds(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4-1: the buffer-to-database move re-sums the floats differently (the
    database `SUM` and the Python sum disagree in the last bits), so a verdict
    that compared the fitted (overhead, rate) pair would end at its own flush
    and bring the over-budget re-measure back at the next lull (`b3a60a8b`).
    The verdict compares which samples the window holds, so it survives."""
    provider = _true_cost_provider(clock, 5.0)
    _cached(monkeypatch, provider)
    points = [(_X5 + 17 * i, 5.0 + 0.13 * i + 0.0071 * (i % 3)) for i in range(1, 12)]
    for x, y in points:
        reranker_mod._record_rerank_cost(_MODEL, _STORE, x, y, None)
    assert _message(provider).hand_off == "budget"
    assert _due(1.0)
    assert _remeasure(1.0)  # one over-budget measurement; the verdict is remembered
    assert _verdict()
    # The premise (so the case is not vacuous): the same samples, summed in
    # Python (the buffer) and by the database (after the flush), give different
    # fitted floats.
    buffered = reranker_mod._rerank_cost.estimate(_STORE, _MODEL)
    assert cost_series.flush(_STORE) >= len(points)
    flushed = reranker_mod._rerank_cost.estimate(_STORE, _MODEL)
    assert buffered is not None and flushed is not None
    assert (buffered.fit.overhead, buffered.fit.rate) != (flushed.fit.overhead, flushed.fit.rate), (
        "the fit moved when the samples were flushed: with floats compared the verdict would end"
    )
    assert _verdict(), "but no sample arrived or aged out: the verdict holds across its own flush"
    assert _due(2.0) is False
    calls = len(_seven_doc_calls(provider))
    assert _remeasure(2.0) is False
    assert len(_seven_doc_calls(provider)) == calls


def test_s127_a_failed_read_keeps_the_verdict(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    _too_long(clock, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(_STORE, "cost_series_window_counts", boom)
    assert _verdict(), "no over-budget re-run on a database hiccup"


def test_s127_the_verdict_is_not_ended_by_the_jobs_own_sample(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimate was inflated (40 x 1.2 s at one size) and the job's own
    measurement (5 s > B) moves the fit; the verdict is still remembered,
    because that sample was taken before the verdict was reached."""
    provider = _stuck(clock, monkeypatch, seconds_for_five=5.0)
    before = rerank_cost_estimate(_MODEL)
    assert _remeasure(1.0)
    after = rerank_cost_estimate(_MODEL)
    assert after.seconds_per_token != before.seconds_per_token, "the job's sample moved the fit"
    assert _verdict()
    assert _message(provider).hand_off == "budget", "still stuck under the new fit"
    assert _due(2.0) is False


def test_s127_a_within_budget_run_remembers_nothing(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stuck(clock, monkeypatch)  # the run measures 1.2 s <= B
    assert _remeasure(1.0)
    assert not _verdict()


def test_s127_a_run_measuring_exactly_the_budget_is_not_too_long(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equality at B fits (S116 a clears on `<=`); only over B is the verdict."""
    provider = _CostScriptedProvider(clock, overhead=_B, rate=0.0)  # 5 + 2 take exactly B
    _cached(monkeypatch, provider)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, _X5, 40 * _B, None)  # inflated: stuck
    assert _message(provider).hand_off == "budget"
    began = clock.t
    assert _remeasure(1.0)
    assert clock.t - began == pytest.approx(_B)
    assert not _verdict()
    assert _MODEL not in reranker_mod._budget_misses


def test_s127_a_model_id_with_no_verdict_is_unaffected(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verdict is per model id: another id's verdict does not block this one."""
    with reranker_mod._cost_lock:
        reranker_mod._too_long["some-other-reranker"] = reranker_mod._Verdict(0.0, 0)
    provider = _stuck(clock, monkeypatch)
    assert _due(1.0) is True
    assert _remeasure(1.0) is True
    assert provider is not None


def test_s127_the_test_reset_forgets_verdicts() -> None:
    with reranker_mod._cost_lock:
        reranker_mod._too_long[_MODEL] = reranker_mod._Verdict(0.0, 0)
    reranker_mod._reset_rerank_cost_model()
    assert reranker_mod._too_long == {}


# ---------------------------------------------------------------------------
# S179: the "too long" verdict is saved with the host-keyed series and survives a restart.
# ---------------------------------------------------------------------------


def _restart(*, flush: bool = True) -> None:
    """What a restart looks like to the code: the buffer is flushed (a clean stop) or lost
    (a crash), every module-level cost state is forgotten and the file is opened by a new
    connection."""
    global _STORE
    if flush:
        cost_series.flush(_STORE)
    path = _STORE.db_path
    _STORE.close()
    reranker_mod._reset_rerank_cost_model()
    cost_series._reset_pending()
    _STORE = MemoryStore(path, integrity_check=False)


def _saved_verdict() -> tuple[float, int] | None:
    return _STORE.cost_verdict("rerank", cost_series.machine_id(), _MODEL)


def _after_restart(provider, monkeypatch: pytest.MonkeyPatch) -> None:
    """The new process: the provider is loaded again and the next message, still unable to
    fit 5 candidates, records a stuck budget miss (which also warms the model up)."""
    _cached(monkeypatch, provider)
    assert _message(provider).hand_off == "budget"
    assert _MODEL in reranker_mod._budget_misses


def test_s179_the_verdict_survives_a_restart_and_the_job_does_not_rerun_the_over_budget_rerank(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    assert len(_seven_doc_calls(provider)) == 1
    saved = _saved_verdict()
    assert saved is not None and saved[1] >= 1, "the verdict is saved under this machine's id and this model id"
    assert _STORE.cost_verdict("rerank", "another-machine", _MODEL) is None
    _restart()
    assert reranker_mod._too_long == {}, "the new process remembers nothing in memory"
    _after_restart(provider, monkeypatch)
    calls = len(provider.calls)
    held = clock.t
    assert _due(1.0) is False
    assert _remeasure(1.0) is False
    assert len(provider.calls) == calls and len(_seven_doc_calls(provider)) == 1, "no further 5 + 2 rerank"
    assert clock.t == held, "no rerank time spent at the lull after the restart"
    assert _saved_verdict() == saved, "the verdict still stands in the database"


def test_s179_the_saved_row_is_read_once_per_database_and_model_id_per_process(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    _restart()
    _after_restart(provider, monkeypatch)
    reads: list[str] = []
    real = _STORE.cost_verdict

    def spy(*a, **k):
        reads.append("read")
        return real(*a, **k)

    monkeypatch.setattr(_STORE, "cost_verdict", spy)
    for lull in (1.0, 2.0, 3.0):
        assert _due(lull) is False and _remeasure(lull) is False
    assert reads == ["read"], "one SELECT on the verdict table, however many due checks and runs"
    reads.clear()
    other = MemoryStore(_STORE.db_path.parent / "another.db", integrity_check=False)
    try:
        monkeypatch.setattr(other, "cost_verdict", spy)
        with reranker_mod._cost_lock:
            reranker_mod._too_long.clear()
        assert reranker_mod._too_long_remembered(_MODEL, other) is False
        assert reads == ["read"], "a second database has its own marker: it is read, and it holds no verdict"
        assert reranker_mod._too_long_remembered(_MODEL, _STORE) is False
        assert reads == ["read"], "the first database's marker is set: its row is not read again"
    finally:
        other.close()


def test_s179_a_sample_arriving_after_a_restart_ends_the_verdict_and_deletes_its_row(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    _restart()
    _after_restart(provider, monkeypatch)
    assert _due(1.0) is False
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 4 * _X5, 4 * 6.0, None)  # any new measurement
    assert _due(2.0) is True, "a sample arrived: the estimate changed (S175)"
    assert _saved_verdict() is None, "the row is gone with the verdict"
    _restart()  # a second restart finds no verdict, whatever the job does next
    _after_restart(provider, monkeypatch)
    assert reranker_mod._too_long_remembered(_MODEL, _STORE) is False
    calls = len(_seven_doc_calls(provider))
    assert _remeasure(2.0) is True
    assert len(_seven_doc_calls(provider)) == calls + 1


def test_s179_a_sample_flushed_by_another_connection_after_a_restart_ends_the_verdict(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    _restart()
    _after_restart(provider, monkeypatch)
    other = MemoryStore(_STORE.db_path, integrity_check=False)
    try:
        now = cost_series._wall_clock()
        other.record_cost_samples(
            "rerank", cost_series.machine_id(), _MODEL, [(3 * _X5, 9.0, now)], now=now,
            window_seconds=cost_series.window_seconds(),
        )
    finally:
        other.close()
    assert reranker_mod._too_long_remembered(_MODEL, _STORE) is False
    assert _saved_verdict() is None


def test_s179_a_sample_that_ages_out_ends_a_saved_verdict_after_a_restart(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    _restart()
    _after_restart(provider, monkeypatch)
    real = cost_series._wall_clock
    shift = [0.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: real() + shift[0])
    shift[0] = 60.0
    assert reranker_mod._too_long_remembered(_MODEL, _STORE) is True, "a step that changes nothing keeps it"
    shift[0] = cost_series.window_seconds() + 60.0
    assert reranker_mod._too_long_remembered(_MODEL, _STORE) is False, "the samples it counted left the window"
    assert _saved_verdict() is None


def test_s179_another_machines_verdict_does_not_count(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _too_long(clock, monkeypatch)
    saved = _saved_verdict()
    _restart()
    monkeypatch.setattr(cost_series, "_machine_override", "another-machine")
    assert reranker_mod._too_long_remembered(_MODEL, _STORE) is False, "S159: a copied database starts without it"
    assert _STORE.cost_verdict("rerank", "test-machine", _MODEL) == saved, "the other machine's row is untouched"
    _restart()
    monkeypatch.setattr(cost_series, "_machine_override", "test-machine")
    assert reranker_mod._too_long_remembered(_MODEL, _STORE) is True
    assert provider is not None


def test_s179_a_restart_after_a_crash_ends_the_verdict_and_costs_one_more_rerank(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consequence C2, pinned: the job's own sample (and any other buffered one) is lost in a
    crash, so the saved count is higher than the database holds and the verdict ends."""
    provider = _true_cost_provider(clock, 5.0)
    _cached(monkeypatch, provider)
    _healthy_estimate(provider)
    assert cost_series.flush(_STORE) == 2  # the estimate is in the database
    assert _message(provider).hand_off == "budget"
    assert _due(1.0) and _remeasure(1.0)  # the job's sample is buffered only
    assert _saved_verdict() is not None and reranker_mod._too_long_remembered(_MODEL, _STORE)
    _restart(flush=False)
    _after_restart(provider, monkeypatch)
    assert _due(2.0) is True, "the verdict does not survive a crash: one more over-budget rerank"
    assert _saved_verdict() is None
    calls = len(_seven_doc_calls(provider))
    assert _remeasure(2.0) is True
    assert len(_seven_doc_calls(provider)) == calls + 1


def test_s179_a_failed_write_keeps_the_in_process_verdict(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(_STORE, "save_cost_verdict", locked)
    provider = _too_long(clock, monkeypatch)  # the job runs; the failed write does not raise
    assert _verdict() is True, "the in-process verdict stands"
    assert _saved_verdict() is None
    assert provider is not None


def test_s179_a_failed_read_of_the_saved_row_means_no_verdict_for_now_and_is_tried_again(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    _restart()
    _after_restart(provider, monkeypatch)

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    real = _STORE.cost_verdict
    monkeypatch.setattr(_STORE, "cost_verdict", locked)
    assert reranker_mod._too_long_remembered(_MODEL, _STORE) is False, "no verdict for now, and no raise"
    monkeypatch.setattr(_STORE, "cost_verdict", real)
    assert reranker_mod._too_long_remembered(_MODEL, _STORE) is True, "the read is tried again at the next consult"


def test_s179_a_failed_delete_leaves_a_row_that_ends_again_at_the_next_load(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _too_long(clock, monkeypatch)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 4 * _X5, 4 * 6.0, None)

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    real = _STORE.clear_cost_verdict
    monkeypatch.setattr(_STORE, "clear_cost_verdict", locked)
    assert _verdict() is False and _saved_verdict() is not None, "ended in process; the row could not be deleted"
    monkeypatch.setattr(_STORE, "clear_cost_verdict", real)
    _restart()
    _after_restart(provider, monkeypatch)
    assert reranker_mod._too_long_remembered(_MODEL, _STORE) is False, "the stale row no longer holds: it ends again"
    assert _saved_verdict() is None


def test_s179_schema_rows_and_pruning(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "v.db", integrity_check=False)
    try:
        window = cost_series.window_seconds()
        store.save_cost_verdict("rerank", "m1", "a", 100.0, 3, now=100.0, window_seconds=window)
        store.save_cost_verdict("rerank", "m2", "b", 200.0, 4, now=200.0, window_seconds=window)
        assert store.cost_verdict("rerank", "m1", "a") == (100.0, 3)
        assert store.cost_verdict("rerank", "m1", "b") is None and store.cost_verdict("other", "m1", "a") is None
        store.save_cost_verdict("rerank", "m1", "a", 150.0, 5, now=150.0, window_seconds=window)
        assert store.cost_verdict("rerank", "m1", "a") == (150.0, 5), "a second verdict replaces the first"
        store.clear_cost_verdict("rerank", "m1", "a", 100.0)
        assert store.cost_verdict("rerank", "m1", "a") == (150.0, 5), "a verdict reached meanwhile survives"
        store.clear_cost_verdict("rerank", "m1", "a", 150.0)
        assert store.cost_verdict("rerank", "m1", "a") is None
        # every row of the series past the window goes on the next write, whatever its host or key
        later = 200.0 + window + 1.0
        store.save_cost_verdict("rerank", "m3", "c", later, 1, now=later, window_seconds=window)
        assert store.cost_verdict("rerank", "m2", "b") is None
        assert store.cost_verdict("rerank", "m3", "c") == (later, 1)
        # the boundary: a row exactly one window old goes (like a sample), one a moment younger stays;
        # another series' rows are not touched
        t = later + 10.0
        store.save_cost_verdict("other series", "m1", "z", 5.0, 1, now=5.0, window_seconds=window)
        store.save_cost_verdict("rerank", "m4", "d", t, 2, now=t, window_seconds=window)
        store.save_cost_verdict("rerank", "m5", "e", t + 0.5, 2, now=t + 0.5, window_seconds=window)
        store.save_cost_verdict("rerank", "m6", "f", t + window, 2, now=t + window, window_seconds=window)
        assert store.cost_verdict("rerank", "m4", "d") is None, "taken_at == now - window is pruned by the next write"
        assert store.cost_verdict("rerank", "m5", "e") == (t + 0.5, 2), "a moment younger stays"
        assert store.cost_verdict("other series", "m1", "z") == (5.0, 1), "another series is not pruned"
        store._conn.execute("BEGIN")  # noqa: SLF001
        with pytest.raises(RuntimeError):
            store.save_cost_verdict("rerank", "m3", "c", 1.0, 1, now=1.0, window_seconds=window)
        with pytest.raises(RuntimeError):
            store.clear_cost_verdict("rerank", "m6", "f", t + window)
        store._conn.rollback()  # noqa: SLF001
        assert store.cost_verdict("rerank", "m6", "f") == (t + window, 2)
    finally:
        store.close()


def test_s179_the_table_is_created_on_open_and_an_old_database_opens(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    MemoryStore(path, integrity_check=False).close()
    con = sqlite3.connect(path)
    con.execute("DROP TABLE cost_series_verdicts")  # a database from before the verdict was saved
    con.commit()
    con.close()
    store = MemoryStore(path, integrity_check=False)
    try:
        assert store.cost_verdict("rerank", "m", "k") is None
        store.save_cost_verdict("rerank", "m", "k", 1.0, 2, now=1.0, window_seconds=10.0)
        assert store.cost_verdict("rerank", "m", "k") == (1.0, 2)
    finally:
        store.close()


def test_s179_a_verdict_reached_while_the_saved_row_is_read_is_not_overwritten(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row is read outside `_cost_lock`; a verdict the job reaches meanwhile wins."""
    provider = _too_long(clock, monkeypatch)
    _restart()
    _after_restart(provider, monkeypatch)
    meanwhile = reranker_mod._Verdict(123.0, 4)
    real = _STORE.cost_verdict

    def reads_and_races(*a, **k):
        row = real(*a, **k)
        with reranker_mod._cost_lock:
            reranker_mod._too_long[_MODEL] = meanwhile
        return row

    monkeypatch.setattr(_STORE, "cost_verdict", reads_and_races)
    assert reranker_mod._load_saved_verdict(_MODEL, _STORE) is meanwhile
    assert reranker_mod._too_long[_MODEL] is meanwhile


def _verdict_statements(store: MemoryStore) -> list[str]:
    seen: list[str] = []
    store._conn.set_trace_callback(  # noqa: SLF001
        lambda sql: seen.append(sql) if "cost_series_verdicts" in sql else None
    )
    return seen


def test_s179_the_saved_row_is_read_once_per_process_also_when_there_is_none(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A2i by SQL count, for the common case (no verdict was ever reached) and for a saved row."""
    provider = _stuck(clock, monkeypatch)  # a stuck record, no verdict
    seen = _verdict_statements(_STORE)
    for lull in (1.0, 2.0, 3.0):
        assert _due(lull) is True
    assert len(seen) == 1 and seen[0].lstrip().upper().startswith("SELECT"), seen
    provider = _too_long(clock, monkeypatch)
    _restart()
    _after_restart(provider, monkeypatch)
    seen = _verdict_statements(_STORE)
    for lull in (1.0, 2.0, 3.0):
        assert _due(lull) is False and _remeasure(lull) is False
    assert len(seen) == 1, seen


def test_s179_a_reranking_turn_and_the_recording_of_its_cost_touch_no_verdict(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A2g at run time: with a verdict saved, messages (budget hand-offs and a real rerank on
    another model id) and the recording of their costs execute no verdict statement."""
    provider = _too_long(clock, monkeypatch)
    for name in ("save_cost_verdict", "clear_cost_verdict", "cost_verdict"):

        def spy(*a, _name=name, **k):
            raise AssertionError(f"{_name} ran on the reply path")

        monkeypatch.setattr(_STORE, name, spy)
    assert _message(provider).hand_off == "budget"
    other = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4, model="another-reranker")
    out = reranker_mod.rerank_for_recall(other, _QUERY, _docs(50, _L), budget_seconds=_B, store=_STORE)
    assert out.reranked and out.measured, "a real rerank ran and its cost was recorded"
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 3 * _X5, 3.0, 1e6)


def test_s179_only_the_lull_job_entry_points_touch_the_verdict() -> None:
    """A2g: nothing on the reply path (`rerank_for_recall` and the callers of the rerank) writes
    or deletes a verdict: the verdict helpers are called only from the due check and the job."""
    tree = ast.parse((_REPO / "brain/memory/reranker.py").read_text(encoding="utf-8"))
    names = {"_remember_too_long", "_too_long_remembered", "_end_verdict", "_load_saved_verdict"}
    callers: dict[str, set[str]] = {n: set() for n in names}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for call in [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]:
            if call.func.id in names and fn.name != call.func.id:
                callers[call.func.id].add(fn.name)
    assert callers == {
        "_remember_too_long": {"remeasure_stuck_width"},
        "_too_long_remembered": {"stuck_width_due", "remeasure_stuck_width"},
        "_end_verdict": {"_too_long_remembered"},
        "_load_saved_verdict": {"_too_long_remembered"},
    }
    storage = {
        "save_verdict": {"_remember_too_long"},
        "load_verdict": {"_load_saved_verdict"},
        "clear_verdict": {"_end_verdict"},
    }
    found: dict[str, set[str]] = {n: set() for n in storage}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for call in [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)]:
            if call.func.attr in storage:
                found[call.func.attr].add(fn.name)
    assert found == storage, "the series' verdict methods are called only by the verdict helpers"
    module_level = [
        c
        for stmt in tree.body
        if not isinstance(stmt, (ast.FunctionDef, ast.ClassDef))
        for c in ast.walk(stmt)
        if isinstance(c, ast.Call)
    ]
    assert not [c for c in module_level if getattr(c.func, "attr", getattr(c.func, "id", "")) in {*storage, *names}]
    definers = {"reranker.py", "cost_series.py", "store.py"}  # the helpers and the storage methods they use
    for path in (_REPO / "brain").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name != "reranker.py":
            assert not any(n in text for n in names), path
        if path.name not in definers:
            assert not any(n in text for n in ("save_verdict", "clear_verdict", "save_cost_verdict")), path


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
    assert _remeasure(1.0)
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
    seen: list[tuple[str, object, object]] = []
    monkeypatch.setattr(supervisor.cli_throttle, "chat_activity_marker", lambda: 42.5)
    monkeypatch.setattr(
        supervisor.reranker,
        "stuck_width_due",
        lambda m, factory: seen.append(("due", m, callable(factory))) or True,
    )
    monkeypatch.setattr(
        supervisor.reranker,
        "remeasure_stuck_width",
        lambda m, store: seen.append(("run", m, isinstance(store, MemoryStore))) or True,
    )
    job = _job(tmp_path, "rerank_width_remeasure")
    assert job.has_work() is True
    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    assert job.run() is JobOutcome.COMPLETED
    assert seen == [("due", 42.5, True), ("run", 42.5, True)]
    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _denying_slot)
    assert job.run() is JobOutcome.SKIPPED
    assert seen == [("due", 42.5, True), ("run", 42.5, True)], "slot lost: no rerank"


def test_c15g_the_due_check_opens_no_store_until_the_in_memory_checks_pass(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[int] = []

    def factory():
        opened.append(1)
        return nullcontext(_STORE)

    # No miss recorded, nothing warmed, no cached provider: every in-memory
    # check says no, so the store factory is never called.
    assert reranker_mod.stuck_width_due(1.0, factory) is False
    assert opened == []
    provider = _stuck(clock, monkeypatch)
    assert provider is not None
    assert reranker_mod.stuck_width_due(1.0, factory) is True
    assert opened == [1]


@pytest.mark.parametrize(
    ("tick", "outcome"),
    [(lambda *a, **k: False, JobOutcome.COMPLETED), (lambda *a, **k: True, JobOutcome.PAUSED)],
)
def test_c16d_the_compaction_job_maps_its_tick_and_touches_no_cost_series(
    tmp_path, monkeypatch: pytest.MonkeyPatch, tick, outcome
) -> None:
    """CS1d: the 24-hour compaction tick no longer starts a cost period (S146:
    a rolling window, no reset); every series' estimate is unchanged by it."""
    monkeypatch.setattr(supervisor, "_run_compaction_tick", tick)
    monkeypatch.setattr("brain.chat.compaction.build_compaction_provider", lambda _pd: None)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, _X5, 1.2, None)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 2 * _X5, 2.4, None)
    before = rerank_cost_estimate(_MODEL)
    assert _job(tmp_path, "compaction").run() is outcome
    assert rerank_cost_estimate(_MODEL) == before


def test_c16d_a_raising_compaction_job_propagates(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("compaction failed")

    monkeypatch.setattr("brain.chat.compaction.build_compaction_provider", lambda _pd: None)
    monkeypatch.setattr(supervisor, "_run_compaction_tick", boom)
    with pytest.raises(RuntimeError):
        _job(tmp_path, "compaction").run()


def test_c16d_nothing_starts_a_new_period_any_more() -> None:
    """S146 supersedes S101's reset at the compaction tick."""
    for path in sorted((_REPO / "brain").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "start_new_period", path


# ---------------------------------------------------------------------------
# C15h: the NellFace label (final: wording file section 8).
# ---------------------------------------------------------------------------


def test_c15h_job_descriptor_is_the_final_label_without_a_placeholder_marker() -> None:
    lines = (_REPO / "app" / "src" / "App.tsx").read_text(encoding="utf-8").splitlines()
    i = next(n for n, line in enumerate(lines) if "rerank_width_remeasure:" in line)
    assert lines[i].strip() == 'rerank_width_remeasure: "re-scoring cost check",'
    assert "—" not in lines[i]
    assert all("PLACEHOLDER" not in line for line in lines)
    assert all("cost_series_flush" not in line for line in lines), "no label entry for the flush job (O10)"
