"""Long-paste increment (S103, S131-S134): the keyword extra-words step's cost
series and sizing, in isolation (`brain.memory.keyword_budget`).

Criteria: L2f (`words_that_fit`), L4 (tier-2 samples, the first-sample discard),
L5 (period, tick, carry, the S133 lower-of-two estimate), L6 (the tier-1 seed,
the S134 discard, the residue), L10 (shared state). The recall-level criteria
(L1-L3, L6a, L9, L11, L13) are in `tests/unit/brain/chat/test_keyword_budget_recall.py`.

Every number is a dyadic fraction of the budget `B`, so the cost lines are exact
in floating point and "strictly inside" / "exactly at" fixtures mean what they
say. The store is a stand-in with a `db_path` (only the series key is read).
Synthetic data only.
"""

from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import tunables
from brain.memory import cost_period, keyword_budget
from brain.memory.cost_period import LinearFit

B = keyword_budget.recall_budget_seconds()
WORDS = [f"kel{i:05d}" for i in range(4096)]


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(keyword_budget, "_clock", c)
    return c


def _store(name: str = "a.db") -> SimpleNamespace:
    return SimpleNamespace(db_path=Path(name))


def _key(store: SimpleNamespace) -> str:
    return str(store.db_path)


def consume_discard() -> None:
    """Spend the process-wide first-sample discard through the production path."""
    keyword_budget.record_tier2("__discard__", 1, 0.0)


def seed_series(store, *samples: tuple[int, float]) -> None:
    """Write samples straight into the series (bypasses the discard by design)."""
    for x, y in samples:
        keyword_budget._record(_key(store), x, y)  # noqa: SLF001


def two_size_fit(store, overhead: float, rate: float) -> None:
    """Sizes 4 and 8: the OLS denominator is 16, so the fit is exactly (overhead, rate)."""
    seed_series(store, (4, overhead + 4 * rate), (8, overhead + 8 * rate))


def run_step(store, words, *, started=0.0, seed=None, cost=0.0, clock=None):
    """One `search_extra_words` call with a stub search; returns the word lists it searched."""
    calls: list[list[str]] = []

    def search(chunk):
        calls.append(list(chunk))
        if clock is not None:
            clock.advance(cost)
        return []

    keyword_budget.search_extra_words(store, words, started, search, seed)
    return calls


def sums(store):
    state = keyword_budget._keyword_cost.state(_key(store))  # noqa: SLF001
    return None if state is None else state.sums


def ols(samples: list[tuple[int, float]]) -> tuple[float, float]:
    """Independent least squares (overhead, rate) over (x, y) samples."""
    n = len(samples)
    sx = sum(x for x, _ in samples)
    sy = sum(y for _, y in samples)
    sxy = sum(x * y for x, y in samples)
    sxx = sum(x * x for x, _ in samples)
    rate = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    overhead = (sy - rate * sx) / n
    if overhead < 0.0 or rate <= 0.0:  # the documented degenerate fallback (cost_period.fit_linear)
        return 0.0, sy / sx
    return overhead, rate


# ---------------------------------------------------------------------------
# L2f: words_that_fit decides by the exact predicate.
# ---------------------------------------------------------------------------


def test_words_that_fit_table() -> None:
    fit = LinearFit(1.0, 0.5, 2)
    assert keyword_budget.words_that_fit(fit, 10, 0.5) == 0, "left below the overhead"
    assert keyword_budget.words_that_fit(fit, 10, 1.0) == 0, "overhead alone, no word fits"
    assert keyword_budget.words_that_fit(fit, 10, 1.25) == 0
    assert keyword_budget.words_that_fit(fit, 10, 1.5) == 1, "exactly predict(1): fits (<=)"
    assert keyword_budget.words_that_fit(fit, 10, 2.25) == 2
    assert keyword_budget.words_that_fit(fit, 10, 100.0) == 10, "clamped to the word count"
    assert keyword_budget.words_that_fit(fit, 0, 100.0) == 0
    assert keyword_budget.words_that_fit(LinearFit(0.25, 0.0, 1), 7, 0.25) == 7, "a zero rate: all fit"
    assert keyword_budget.words_that_fit(LinearFit(0.5, 0.0, 1), 7, 0.25) == 0


def test_words_that_fit_clamps_a_denormal_rate_and_an_infinite_quotient() -> None:
    assert keyword_budget.words_that_fit(LinearFit(0.0, 5e-324, 1), 9, 1.0) == 9
    assert keyword_budget.words_that_fit(LinearFit(0.0, 1e-320, 1), 9, float("inf")) == 9


def test_words_that_fit_decides_by_the_predicate_not_the_float_quotient() -> None:
    """A fixture found by search where int((left - o) / r) is off by one from the
    exact answer, both ways (the test fails loudly if the search finds none)."""
    o = 0.1
    below = above = None
    for ri in range(1, 200):
        r = ri / 100.0
        fit = LinearFit(o, r, 1)
        for n in range(1, 80):
            exact_left = fit.predict(n)
            if below is None and int(min(1000.0, (exact_left - o) / r)) < n:
                below = (fit, n, exact_left)
            just_under_next = math.nextafter(fit.predict(n + 1), 0.0)
            if above is None and int(min(1000.0, (just_under_next - o) / r)) > n:
                above = (fit, n, just_under_next)
    assert below is not None and above is not None, "no float-step fixture found"
    fit, n, left = below
    assert fit.predict(n) <= left and keyword_budget.words_that_fit(fit, n + 5, left) == n
    fit, n, left = above
    assert fit.predict(n + 1) > left >= fit.predict(n)
    assert keyword_budget.words_that_fit(fit, n + 5, left) == n


# ---------------------------------------------------------------------------
# L4: the measurement, tier-2 samples only; S134: the first is discarded.
# ---------------------------------------------------------------------------


def test_a_tier2_search_records_one_exact_sample(clock: Clock) -> None:
    store = _store()
    consume_discard()
    run_step(store, WORDS[:20], seed=(10, 10 * B / 1024), cost=B / 64, clock=clock)
    s = sums(store)
    assert (s.n, s.sum_x, s.sum_y, s.sum_xy, s.sum_xx) == (1, 20, B / 64, 20 * (B / 64), 400)


def test_the_sample_counts_the_words_sent_not_the_dropped_ones(clock: Clock) -> None:
    store = _store()
    consume_discard()
    calls = run_step(store, WORDS[:1000], started=-(B - B / 4), seed=(8, 8 * B / 1024), cost=B / 128, clock=clock)
    sent = len(calls[0])
    assert 0 < sent < 1000
    s = sums(store)
    assert s.n == 1 and s.sum_x == sent and s.sum_y == B / 128


def test_no_sample_from_tier_one_a_skip_or_a_raising_search(clock: Clock) -> None:
    store = _store()
    consume_discard()
    # timing tier 1 for the seed records nothing
    result, seed = keyword_budget.timed_tier1(store, ["alpha", "bravo", "charlie"], lambda: "r")
    assert result == "r" and seed is not None and seed[0] == 3
    assert sums(store) is None
    # an exhausted budget: no search, no sample
    assert run_step(store, WORDS[:10], started=-B, seed=(3, 0.0)) == []
    assert sums(store) is None

    def boom(_chunk):
        raise RuntimeError("store failure")

    with pytest.raises(RuntimeError):
        keyword_budget.search_extra_words(store, WORDS[:10], 0.0, boom, (3, 0.0))
    assert sums(store) is None, "a search that raised records nothing"


def test_a_step_where_one_word_does_not_fit_records_nothing(clock: Clock) -> None:
    store = _store()
    consume_discard()
    two_size_fit(store, B / 256, B / 1024)
    before = sums(store)
    # left = overhead + half a word: the step is decided as n = 0, not skipped by left <= 0
    started = -(B - (B / 256 + B / 2048))
    assert run_step(store, WORDS, started=started, seed=None, cost=B / 64, clock=clock) == []
    assert sums(store) == before


def test_the_budget_is_the_tunable_read_now(monkeypatch: pytest.MonkeyPatch) -> None:
    real = tunables.get_tunable

    def override(name, default, *a, **k):
        return 6.0 if name == "reranker.latency_budget_seconds" else real(name, default, *a, **k)

    monkeypatch.setattr(tunables, "get_tunable", override)
    assert keyword_budget.recall_budget_seconds() == 6.0
    store = _store()
    two_size_fit(store, 0.0, 6.0 / 1024)
    assert [len(c) for c in run_step(store, WORDS)] == [1024], "sized against the tunable's 6.0"


def test_the_estimate_follows_two_sizes_the_least_squares_fit(clock: Clock) -> None:
    store = _store()
    consume_discard()
    run_step(store, WORDS[:8], seed=(4, 0.0), cost=B / 64 + 8 * (B / 1024), clock=clock)
    run_step(store, WORDS[:24], seed=(4, 0.0), cost=B / 64 + 24 * (B / 1024), clock=clock)
    want = ols([(8, B / 64 + 8 * (B / 1024)), (24, B / 64 + 24 * (B / 1024))])
    got = keyword_budget._keyword_cost.fit(_key(store))  # noqa: SLF001
    assert got.overhead == pytest.approx(want[0], rel=1e-12)
    assert got.rate == pytest.approx(want[1], rel=1e-12)


def test_the_series_is_keyed_by_the_store(clock: Clock) -> None:
    a, b = _store("a.db"), _store("b.db")
    consume_discard()
    run_step(a, WORDS[:5], seed=(5, 0.0), cost=B / 32, clock=clock)
    assert sums(a).n == 1 and sums(b) is None


def test_the_first_tier2_sample_of_a_process_is_discarded_the_second_is_recorded(clock: Clock) -> None:
    store = _store()
    run_step(store, WORDS[:5], seed=(5, 0.0), cost=B / 32, clock=clock)
    assert sums(store) is None, "S134: the first sample is dropped"
    run_step(store, WORDS[:5], seed=(5, 0.0), cost=B / 32, clock=clock)
    assert sums(store).n == 1, "the second is recorded"


def test_the_discard_is_per_process_not_per_store(clock: Clock) -> None:
    a, b = _store("a.db"), _store("b.db")
    run_step(a, WORDS[:5], seed=(5, 0.0), cost=B / 32, clock=clock)
    run_step(b, WORDS[:5], seed=(5, 0.0), cost=B / 32, clock=clock)
    assert sums(a) is None and sums(b).n == 1, "a second store's first search is recorded"


def test_a_tick_does_not_rearm_the_discard(clock: Clock) -> None:
    store = _store()
    consume_discard()
    cost_period.start_new_period()
    run_step(store, WORDS[:5], seed=(5, 0.0), cost=B / 32, clock=clock)
    assert sums(store).n == 1


def test_a_raise_and_a_skip_do_not_spend_the_discard(clock: Clock) -> None:
    store = _store()

    def boom(_chunk):
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        keyword_budget.search_extra_words(store, WORDS[:5], 0.0, boom, (5, 0.0))
    assert run_step(store, WORDS[:5], started=-B, seed=(5, 0.0)) == []
    run_step(store, WORDS[:5], seed=(5, 0.0), cost=B / 32, clock=clock)
    assert sums(store) is None, "the first search that RETURNED is the discarded one"
    run_step(store, WORDS[:5], seed=(5, 0.0), cost=B / 32, clock=clock)
    assert sums(store).n == 1


# ---------------------------------------------------------------------------
# L5: per period, the tick, the carry, the S133 lower-of-two estimate.
# ---------------------------------------------------------------------------


def _carried(store, overhead: float, rate: float) -> None:
    two_size_fit(store, overhead, rate)
    cost_period.start_new_period()


def test_after_a_tick_the_sums_are_empty_and_the_fit_is_carried() -> None:
    store = _store()
    two_size_fit(store, 0.0, B / 256)
    cost_period.start_new_period()
    state = keyword_budget._keyword_cost.state(_key(store))  # noqa: SLF001
    assert state.sums.n == 0 and state.carried is not None
    assert state.carried.rate == B / 256


@pytest.mark.parametrize(
    ("seed_words", "seed_seconds", "seed_absent", "want"),
    [
        (8, 8 * B / 512, False, 512),  # seed cheaper than the carried fit: the seed wins
        (8, 8 * B / 128, False, 256),  # seed dearer: the carried fit wins
        (8, 0.0, True, 256),  # no seed (tier 1 raised): the carried fit alone
    ],
)
def test_the_estimate_after_a_tick_is_the_lower_of_the_carried_fit_and_the_seed(
    seed_words, seed_seconds, seed_absent, want
) -> None:
    store = _store()
    _carried(store, 0.0, B / 256)
    seed = None if seed_absent else (seed_words, seed_seconds)
    calls = run_step(store, WORDS, seed=seed)
    assert [len(c) for c in calls] == [want]


def test_crossing_fits_take_the_lower_predicted_cost_not_the_lower_rate() -> None:
    """Carried: overhead 3B/4, rate B/4096 (lower rate, high overhead); seed: rate
    B/2048. The lines cross at 3072 words (inside the list). At left = B the seed
    is cheaper at 2048 words and fits them; the carried fit fits only 1024. The
    plan's reading (the prefix that fits under either) gives 2048; a lower-rate
    pick would give 1024."""
    store = _store()
    _carried(store, 3 * B / 4, B / 4096)
    calls = run_step(store, WORDS, seed=(8, 8 * B / 2048))
    assert [len(c) for c in calls] == [2048]


def test_once_the_new_period_has_a_tier2_sample_the_carried_fit_is_rescaled_and_the_seed_is_ignored(
    clock: Clock,
) -> None:
    store = _store()
    _carried(store, 0.0, B / 256)
    consume_discard()
    run_step(store, WORDS[:64], seed=(8, 8 * B / 512), cost=64 * B / 128, clock=clock)  # measured at 2x the carried cost
    clock.t = 0.0
    calls = run_step(store, WORDS, seed=(8, 8 * B / 512))  # the seed would imply 512
    assert [len(c) for c in calls] == [128], "rescaled by one factor (2), seed not consulted"


def test_with_two_sizes_the_new_periods_own_fit_takes_over(clock: Clock) -> None:
    store = _store()
    _carried(store, 0.0, B / 256)
    consume_discard()
    run_step(store, WORDS[:32], seed=(8, 0.0), cost=32 * B / 64, clock=clock)
    clock.t = 0.0
    run_step(store, WORDS[:64], seed=(8, 0.0), cost=64 * B / 64, clock=clock)
    clock.t = 0.0
    calls = run_step(store, WORDS, seed=(8, 0.0))
    assert [len(c) for c in calls] == [64]


def test_the_hook_is_registered_by_importing_the_prompt_module_alone() -> None:
    code = (
        "import brain.chat.prompt\n"
        "from brain.memory import cost_period\n"
        "names = [getattr(getattr(h, '__self__', None), 'name', None) for h in cost_period._hooks]\n"
        "print('keyword extra words' in names)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "True"


def test_a_carried_seed_ten_times_too_high_is_rescaled_by_one_factor_at_the_first_new_sample(clock: Clock) -> None:
    store = _store()
    true_rate = B / 2048
    _carried(store, 0.0, 10 * true_rate)
    consume_discard()
    run_step(store, WORDS[:64], seed=(8, 8 * true_rate), cost=64 * true_rate, clock=clock)
    clock.t = 0.0
    calls = run_step(store, WORDS, seed=None)
    assert [len(c) for c in calls] == [2048], "the next sizing follows the one sample at once"


def test_an_outlier_sample_is_worked_out_by_later_samples_as_independent_least_squares_says() -> None:
    store = _store()
    r = B / 4096
    samples = [(4, 4 * r), (8, 8 * r), (12, 12 * r), (16, 16 * r), (512, 10 * 512 * r)]
    seed_series(store, *samples)
    costs = []
    for _ in range(6):
        costs.append(keyword_budget._keyword_cost.fit(_key(store)).predict(512))  # noqa: SLF001
        samples.append((64, 64 * r))
        seed_series(store, (64, 64 * r))
        want = ols(samples)
        got = keyword_budget._keyword_cost.fit(_key(store))  # noqa: SLF001
        assert got.rate == pytest.approx(want[1], rel=1e-9)
        assert got.overhead == pytest.approx(want[0], rel=1e-9, abs=1e-12)
    costs.append(keyword_budget._keyword_cost.fit(_key(store)).predict(512))  # noqa: SLF001
    assert all(a > b for a, b in zip(costs, costs[1:], strict=False)), "the outlier's pull falls monotonically"


def test_a_stuck_high_carried_fit_is_bounded_by_the_seed_after_a_tick(clock: Clock) -> None:
    store = _store()
    _carried(store, 0.0, B / 2)  # predict(1) = B/2
    consume_discard()
    calls = run_step(store, WORDS, started=-(3 * B / 4), seed=(8, 8 * B / 1024), cost=B / 1024, clock=clock)
    assert calls and len(calls[0]) >= 1, "left = B/4 < predict_carried(1), but the cheaper seed searches"
    assert sums(store).n == 1, "and its sample is recorded, so the next message is sized from it"


def test_pinned_within_a_period_a_stuck_high_own_fit_skips_until_the_tick() -> None:
    """NOT a recovery (owner residual): an own-period fit too high to fit one word
    skips the step, records nothing and stays; the next tick carries it and S133's
    seed bounds it from then on."""
    store = _store()
    seed_series(store, (4, 4 * B), (8, 8 * B))  # rate B: predict(1) = B
    assert run_step(store, WORDS, started=-(3 * B / 4), seed=(8, 8 * B / 1024)) == []
    assert sums(store).n == 2, "nothing recorded by the skip"
    cost_period.start_new_period()
    calls = run_step(store, WORDS, started=-(3 * B / 4), seed=(8, 8 * B / 1024))
    assert calls and len(calls[0]) >= 1


# ---------------------------------------------------------------------------
# L6: the seed, the residue.
# ---------------------------------------------------------------------------


def test_a_fresh_process_sizes_tier_two_from_its_own_tier_one_seed() -> None:
    store = _store()
    x1, y1 = 8, 8 * B / 512  # rate B/512
    left_words = 40
    started = -(B - (left_words + 0.5) * (B / 512) - 0.0)  # left = 40.5 words' cost
    started = -(B - (left_words + 0.5) * (B / 512))
    calls = run_step(store, WORDS, started=started, seed=(x1, y1))
    assert [len(c) for c in calls] == [left_words]


def test_the_seed_is_transient_and_the_discarded_first_sample_leaves_the_next_message_seeded_again(
    clock: Clock,
) -> None:
    store = _store()
    seed = (8, 8 * B / 512)
    clock.t = 0.0
    first = run_step(store, WORDS[:100], seed=seed, cost=B / 64, clock=clock)
    assert len(first[0]) == 100 and sums(store) is None, "S134: nothing recorded by the first search"
    clock.t = 0.0
    # the next message is sized by ITS OWN seed (wildly different), not the first's
    second = run_step(store, WORDS, started=-(B - 8.5 * (B / 64)), seed=(8, 8 * B / 64), cost=B / 64, clock=clock)
    assert [len(c) for c in second] == [8]
    assert sums(store).n == 1 and sums(store).sum_x == 8, "the second is recorded; no tier-1 sample anywhere"
    clock.t = 0.0
    third = run_step(store, WORDS, seed=(8, 8 * B / 4096))  # a very cheap seed is ignored now
    assert [len(c) for c in third] == [512], "sized from the series (rate B/512), not from the seed"


def test_no_fit_and_no_seed_means_no_extras() -> None:
    assert run_step(_store(), WORDS, seed=None) == []


def test_no_extras_when_nothing_is_left_even_with_a_seed() -> None:
    assert run_step(_store(), WORDS, started=-B, seed=(8, 0.0)) == []


def test_seed_fit_is_none_for_zero_words_and_through_the_origin_otherwise() -> None:
    assert keyword_budget.seed_fit(0, 1.0) is None
    fit = keyword_budget.seed_fit(4, 1.0)
    assert (fit.overhead, fit.rate) == (0.0, 0.25)


def test_timed_tier1_seed_counts_the_words_the_store_searches() -> None:
    """A 2-letter token inside the first 10 is dropped by the store's 3-character
    floor: x1 is the floor-filtered count, not the token count."""
    store = _store()
    tokens = ["alpha", "ab", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]
    _, seed = keyword_budget.timed_tier1(store, tokens, lambda: None)
    assert seed[0] == 9
    _, one_word_seed = keyword_budget.timed_tier1(store, ["ab", "cd"], lambda: None)
    assert one_word_seed is not None and one_word_seed[0] == 1, "S140: tier 1 searched no word: its time is a one-word cost"


# ---------------------------------------------------------------------------
# L10: the new series is one more accessor.
# ---------------------------------------------------------------------------


def test_a_concurrent_record_inside_another_records_window_loses_neither(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "race.db"
    started: list[threading.Thread] = []

    def hook() -> None:
        monkeypatch.setattr(cost_period, "_update_hook", None)
        t = threading.Thread(target=lambda: keyword_budget._keyword_cost.record(key, 6, 0.5))  # noqa: SLF001
        t.start()
        started.append(t)

    monkeypatch.setattr(cost_period, "_update_hook", hook)
    keyword_budget._keyword_cost.record(key, 4, 0.25)  # noqa: SLF001
    started[0].join()
    s = keyword_budget._keyword_cost.state(key).sums  # noqa: SLF001
    assert s.n == 2 and s.sum_x == 10 and s.sum_y == 0.75


def test_a_tick_between_the_fit_lookup_and_the_record_leaves_no_half_reset_sums(clock: Clock) -> None:
    store = _store()
    consume_discard()
    two_size_fit(store, 0.0, B / 1024)
    ticked: list[int] = []

    def search(chunk):
        cost_period.start_new_period()
        ticked.append(len(chunk))
        return []

    keyword_budget.search_extra_words(store, WORDS[:50], 0.0, search, None)
    s = sums(store)
    assert ticked == [50]
    assert s.n == 1 and s.sum_x == 50, "the record landed wholly in the new period"


def test_the_series_state_is_read_once_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store()
    consume_discard()
    two_size_fit(store, 0.0, B / 1024)
    reads: list[str] = []
    real = keyword_budget._keyword_cost.state  # noqa: SLF001

    def counting(key):
        reads.append(key)
        return real(key)

    monkeypatch.setattr(keyword_budget._keyword_cost, "state", counting)  # noqa: SLF001
    run_step(store, WORDS[:30], seed=None)
    assert len(reads) == 1


def test_two_threads_finishing_a_first_search_together_discard_exactly_one(monkeypatch: pytest.MonkeyPatch) -> None:
    barrier = threading.Barrier(2)
    monkeypatch.setattr(keyword_budget, "_record_hook", lambda: barrier.wait(timeout=5))
    # inside the critical section, between the check and the set: a window the lock must close
    monkeypatch.setattr(keyword_budget, "_flag_hook", lambda: time.sleep(0.05))
    threads = [
        threading.Thread(target=keyword_budget.record_tier2, args=("race2.db", 10, 0.25)) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s = keyword_budget._keyword_cost.state("race2.db").sums  # noqa: SLF001
    assert s.n == 1, "one discarded, one recorded"


# ---------------------------------------------------------------------------
# S139: the linear fit is never extrapolated; S140: tier 1 searched nothing.
# ---------------------------------------------------------------------------


def _words(n: int) -> list[str]:
    return WORDS[:n]


def test_s139_an_uncovered_series_is_sized_by_the_larger_cost_of_the_fit_and_the_seed() -> None:
    store = _store()
    two_size_fit(store, 0.0, B / 2048)  # sizes 4 and 8 only: 9+ extras are uncovered
    # seed dearer than the fit (rate B/1024): the seed bounds it: 1024 words fit, not the fit's 2048
    assert [len(c) for c in run_step(store, _words(4096), seed=(8, 8 * B / 1024))] == [1024]
    # seed cheaper than the fit: the fit is the larger cost: its 2048
    assert [len(c) for c in run_step(store, _words(4096), seed=(8, 8 * B / 4096))] == [2048]


def test_s139_a_series_that_has_measured_the_extras_size_is_sized_by_the_fit_alone() -> None:
    store = _store()
    two_size_fit(store, 0.0, B / 2048)  # largest sample: 8 words
    dear_seed = (8, 16 * B)  # a word costs 2B: nothing fits under it
    assert [len(c) for c in run_step(store, _words(8), seed=dear_seed)] == [8], "8 extras are covered: the dear seed is ignored"
    assert run_step(store, _words(9), seed=dear_seed) == [], "9 extras are not covered: the dear seed fits none"


def test_s139_without_a_seed_an_uncovered_series_has_only_its_fit() -> None:
    store = _store()
    two_size_fit(store, 0.0, B / 2048)
    assert [len(c) for c in run_step(store, _words(4096), seed=None)] == [2048]


def test_s139_coverage_is_the_largest_recorded_sample_not_the_discarded_first_one(clock: Clock) -> None:
    store = _store()
    run_step(store, _words(500), seed=(8, 0.0), cost=B / 64, clock=clock)  # the discarded first sample (500 words)
    assert sums(store) is None
    clock.t = 0.0
    run_step(store, _words(20), seed=(8, 0.0), cost=B / 64, clock=clock)  # recorded: 20 words
    est = keyword_budget.choose_estimate(store, 100, (8, 8 * B / 4096))
    assert est.source == "series+seed", "100 extras exceed the 20-word sample (the discarded 500 does not count)"
    assert keyword_budget.choose_estimate(store, 20, (8, 8 * B / 4096)).source == "series"


def test_s139_coverage_resets_with_the_period_and_the_tick_window_keeps_s133() -> None:
    store = _store()
    two_size_fit(store, 0.0, B / 256)
    keyword_budget._record(_key(store), 4096, 4096 * B / 256)  # noqa: SLF001  (a big sample this period)
    assert keyword_budget.choose_estimate(store, 4000, None).source == "series"
    keyword_budget._record(_key(store), 8, 8 * B / 256)  # noqa: SLF001  (a later, smaller sample must not shrink the coverage)
    assert keyword_budget.choose_estimate(store, 4000, (8, 8 * B / 512)).source == "series", "coverage is the LARGEST sample"
    cost_period.start_new_period()
    est = keyword_budget.choose_estimate(store, 4000, (8, 8 * B / 512))
    assert est.source == "carried/seed" and not est.larger, "S133 window after a tick: the lower cost"
    keyword_budget._record(_key(store), 8, 8 * B / 256)  # noqa: SLF001
    est = keyword_budget.choose_estimate(store, 4000, (8, 8 * B / 512))
    assert est.source == "series+seed" and est.larger, "the big sample belonged to the old period: uncovered again"


def test_estimate_prediction_and_fit_agree() -> None:
    a, b = LinearFit(0.0, B / 1024, 2), LinearFit(B / 4, B / 4096, 2)
    larger = keyword_budget.Estimate((a, b), True, "t")
    lower = keyword_budget.Estimate((a, b), False, "t")
    for n in (10, 1000, 3000):
        assert larger.predict(n) == max(a.predict(n), b.predict(n))
        assert lower.predict(n) == min(a.predict(n), b.predict(n))
    left = B / 2
    assert larger.words_that_fit(4096, left) == min(keyword_budget.words_that_fit(f, 4096, left) for f in (a, b))
    assert lower.words_that_fit(4096, left) == max(keyword_budget.words_that_fit(f, 4096, left) for f in (a, b))


def test_s140_tier_one_that_searched_nothing_seeds_with_its_own_time_as_a_one_word_cost(clock: Clock) -> None:
    store = _store()

    def tier1():
        clock.advance(B / 4)

    _, seed = keyword_budget.timed_tier1(store, ["ab", "cd"], tier1)
    assert seed == (1, B / 4)
    # sized as one word per B/4: left = 3B/4 fits 3 extras
    calls = run_step(store, _words(40), started=0.0, seed=seed)
    assert [len(c) for c in calls] == [3]


def test_a_combined_estimate_fits_exactly_the_predicate_on_its_combined_cost() -> None:
    """Independent of the implementation: the largest n whose combined predicted cost fits `left`,
    found by brute force, for both combination rules, including a zero rate and left == a predicted cost."""
    fits = [
        (LinearFit(0.0, B / 1024, 2), LinearFit(B / 4, B / 4096, 2)),
        (LinearFit(B / 8, 0.0, 1), LinearFit(0.0, B / 512, 1)),
        (LinearFit(B / 16, B / 2048, 2), LinearFit(B / 16, B / 2048, 2)),
    ]
    for pair in fits:
        for larger in (True, False):
            est = keyword_budget.Estimate(pair, larger, "t")
            for left in (0.0, B / 64, B / 8, est.predict(37), est.predict(200), B / 2, B):
                brute = max((n for n in range(0, 301) if n == 0 or est.predict(n) <= left), default=0)
                assert est.words_that_fit(300, left) == brute, (pair, larger, left)


def test_the_period_keeps_its_largest_sample_size_with_the_sums_and_resets_with_them() -> None:
    sums = cost_period.LinearSums().plus(5, 0.1).plus(3, 0.1).plus(9, 0.2).plus(4, 0.1)
    assert sums.max_x == 9, "the largest, not the last"
    store = _store()
    seed_series(store, (4, 0.25), (80, 1.0), (8, 0.5))
    assert keyword_budget._keyword_cost.state(_key(store)).sums.max_x == 80  # noqa: SLF001
    cost_period.start_new_period()
    state = keyword_budget._keyword_cost.state(_key(store))  # noqa: SLF001
    assert state.sums.max_x == 0 and state.sums.n == 0 and state.carried is not None
