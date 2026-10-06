"""The keyword extra-words step, the name walk and the per-message recall time
budget (name-recall fix S103, S130-S134, S139-S148, S155-S157, S160; spec
section 7, "Keyword step 3" and "Cost-series clarifications").

Passive recall's keyword search runs tier 1 (today's capped 10-word query), the
name query, and tier 2, the "extra words" (every selector token tier 1 does not
reach), in that order (S157). Tier 2 searches as many of those words as fit what
is left of the per-message recall time budget, in the selector's own salience
order, and drops the rest for that turn. Tier 1 and the name query's first page
always run (S103, S155).

"What is left" (S132) is literal: the budget (the `reranker.latency_budget_seconds`
tunable, the one the paragraph search and the rerank share) minus the wall time
this message's recall has already spent, measured on one monotonic clock from
the top of `_build_recall_block` (or of `search_memories`).

The cost of a search is predicted as `overhead + rate * words` (S101, S67), a
least-squares fit over the last 24 hours' samples (S146: a rolling window saved
in memories.db, `cost_series`; no daily reset), fed by tier-2 searches ONLY
(S131). The first tier-2 sample of a process is discarded (S134), as the rerank
and embed series discard their first call. Samples are buffered on the reply
path and saved at the next lull (S164).

The estimate is sized from the saved series; this message's own tier-1 search is
the SEED (S131, S140, S147): `(x1, seconds)` for the `x1 >= 1` words tier 1
searched, or `(1, seconds)` when tier 1 searched none AND no extra word reaches
the store's 3-character floor ("AI ML"); otherwise there is no seed. With no
estimate and no seed no extra word is searched.

The linear fit is never extrapolated (S139): the real cost is superlinear in the
word count, so until the series' largest measured size covers the current
extras, the budget is sized from the LARGER predicted cost of the fit and the
seed (the seed over-cuts, never overruns); with no seed the fit is used at most
up to its measured range. An estimate that fits no extra word while time is left
still searches ONE word (S156), which re-samples the series, so a stuck-high
estimate recovers.

The name walk (S145, S155, S157, S160) is the same routine for the keyword name
query's continuation past its first page: `NameWalkBudget` below.

The search tool's extras are not affected (S99): its keyword side never comes
here; its name query is budgeted the same way as passive recall's (S157).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from brain import tunables
from brain.memory import cost_series
from brain.memory import reranker as reranker_mod
from brain.memory.store import MemoryStore, split_by_raw_query_floor

log = logging.getLogger(__name__)

# Clock seam for the per-message budget and the cost samples. Production:
# `time.perf_counter` (monotonic, and fine-grained on every platform: a coarse
# clock would read a short search as 0.0 s and turn the estimate into "every word
# fits"). Tests script it.
_clock: Callable[[], float] = time.perf_counter

# The per-word keyword cost (S103, S131, S146): one key, one sample per tier-2
# search, x = words sent to the store, y = that call's seconds. The store the
# sample is read from and buffered against is the caller's.
_keyword_cost = cost_series.CostSeries("keyword extra words")
# The name-matching cost (S145, S146): the keyword name query's continuation
# walk, x = characters judged, y = the walk's seconds.
_name_cost = cost_series.CostSeries("name matching")
_NAME_KEY = "keyword"
_KEYWORD_KEY = ""

# S134 / S160: the first sample of this process in each series is discarded. One
# process-wide flag per series under its own lock.
_first_sample_lock = threading.Lock()
_first_sample_spent = False
_name_first_sample_lock = threading.Lock()
_name_first_sample_spent = False

# Test-only seams: `_record_hook` is called at the top of `record_tier2`, before
# the flag check; `_flag_hook` inside the lock, between the check and the set.
# `_name_record_hook` / `_name_flag_hook` are the same for the name walk.
_record_hook: Callable[[], None] | None = None
_flag_hook: Callable[[], None] | None = None
_name_record_hook: Callable[[], None] | None = None
_name_flag_hook: Callable[[], None] | None = None


def _reset_keyword_cost() -> None:
    """Test-only: re-arm both first-sample discards. The series live in the
    database and the process buffer (`cost_series._reset_pending`)."""
    global _first_sample_spent, _name_first_sample_spent
    with _first_sample_lock:
        _first_sample_spent = False
    with _name_first_sample_lock:
        _name_first_sample_spent = False


def now() -> float:
    """The budget clock's current reading."""
    return _clock()


def recall_budget_seconds() -> float:
    """The per-message recall time budget, read now (the same tunable and
    default `search_paragraphs` reads)."""
    return float(
        tunables.get_tunable("reranker.latency_budget_seconds", reranker_mod.LATENCY_BUDGET_SECONDS)
    )


def words_that_fit(fit: cost_series.LinearFit, count: int, left: float) -> int:
    """The longest prefix length `n <= count` with `fit.predict(n) <= left`.

    The closed-form quotient is only a candidate: the exact predicate decides
    (one correction up or down), so float rounding at `left == predict(n)`
    cannot cost a word. The quotient is clamped to `count` BEFORE `int()`, so a
    denormal rate or an infinite quotient cannot overflow."""
    if count <= 0 or left < fit.overhead:
        return 0
    if fit.rate <= 0.0:
        return count
    n = max(0, int(min(float(count), (left - fit.overhead) / fit.rate)))
    if n < count and fit.predict(n + 1) <= left:
        return n + 1
    if n > 0 and fit.predict(n) > left:
        return n - 1
    return n


def timed[T](call: Callable[[], T]) -> tuple[T, float]:
    """`call()` and the seconds it took on the budget clock. An exception from
    `call` propagates."""
    began = _clock()
    result = call()
    return result, _clock() - began


def timed_tier1[T](
    store: MemoryStore, legacy_tokens: Sequence[str], call: Callable[[], T]
) -> tuple[T, tuple[int, float]]:
    """Run tier 1's own search and return `(its result, (x1, seconds))`: `x1` =
    the words the store actually searches in tier 1 (its 3-character floor
    applied; 0 when it searched none), `seconds` = the whole call (S131: this
    message's own tier-1 search). `seed_for` turns it into the seed. An exception
    from `call` propagates (no seed); nothing is recorded, tier 1 is unchanged."""
    x1 = len(split_by_raw_query_floor(list(legacy_tokens))[0])
    result, seconds = timed(call)
    return result, (x1, seconds)


def seed_for(
    tier1: tuple[int, float] | None, extras: Sequence[str]
) -> tuple[int, float] | None:
    """The seed for a message's extra words from its tier-1 timing (S140, S147):
    `(x1, seconds)` when tier 1 searched `x1 >= 1` words; `(1, seconds)` (tier
    1's measured time taken as a one-word cost) when it searched none AND no
    extra word reaches the store's 3-character floor; otherwise `None` (the
    saved series sizes the extras, S146)."""
    if tier1 is None:
        return None
    x1, seconds = tier1
    if x1 >= 1:
        return (x1, seconds)
    if not split_by_raw_query_floor(list(extras))[0]:
        return (1, seconds)
    return None


def seed_fit(words: int, seconds: float) -> cost_series.LinearFit | None:
    """The per-message estimate from tier 1's own timing: seconds per word
    through the origin (the one-size form `cost_series.fit_linear` falls back
    to). `None` for zero words. Never stored."""
    if words <= 0:
        return None
    return cost_series.LinearFit(0.0, seconds / words, 1)


def record_tier2(store: MemoryStore, words: int, seconds: float) -> None:
    """Buffer one tier-2 sample, except the process's first (S134)."""
    global _first_sample_spent
    if _record_hook is not None:
        _record_hook()
    with _first_sample_lock:
        if not _first_sample_spent:
            if _flag_hook is not None:
                _flag_hook()
            _first_sample_spent = True
            return
    _keyword_cost.record(store, _KEYWORD_KEY, words, seconds)


def record_name_walk(store: MemoryStore, chars: int, seconds: float) -> None:
    """Buffer one name-walk sample (x = characters judged), except the process's
    first (S160, as S134)."""
    global _name_first_sample_spent
    if _name_record_hook is not None:
        _name_record_hook()
    with _name_first_sample_lock:
        if not _name_first_sample_spent:
            if _name_flag_hook is not None:
                _name_flag_hook()
            _name_first_sample_spent = True
            return
    _name_cost.record(store, _NAME_KEY, chars, seconds)


@dataclass(frozen=True)
class Estimate:
    """The cost estimate a step is sized from: one or two linear fits and an
    optional word cap. A prefix fits only if it fits under EVERY fit (the larger
    predicted cost, S139), and never exceeds `cap` words (the fit's measured
    range when there is no seed to over-cut with, S160)."""

    fits: tuple[cost_series.LinearFit, ...]
    cap: int | None
    source: str

    def predict(self, words: int) -> float:
        return max(f.predict(words) for f in self.fits)

    def words_that_fit(self, count: int, left: float) -> int:
        fitting = min(words_that_fit(f, count, left) for f in self.fits)
        return fitting if self.cap is None else min(fitting, self.cap)


def choose_estimate(
    store: MemoryStore, extras: int, seed: tuple[int, float] | None
) -> Estimate | None:
    """The estimate for a message with `extras` extra words (None: nothing to
    size from: no extras were searched). One read of the series.

    - The series has an estimate whose measured range covers `extras`: the
      estimate alone (S146, S114).
    - It does not cover them (S139: the linear fit is not extrapolated) and there
      is a seed: the LARGER cost of the estimate and the seed (S148: a carried
      estimate is sized the same way).
    - It does not cover them and there is no seed: the estimate, at most up to
      its measured range (S160).
    - No estimate: the seed alone; neither: None."""
    ranged = _keyword_cost.estimate(store, _KEYWORD_KEY)
    from_seed = seed_fit(*seed) if seed is not None else None
    if ranged is None:
        return None if from_seed is None else Estimate((from_seed,), None, "seed")
    if ranged.max_x >= extras:
        return Estimate((ranged.fit,), None, "series")
    if from_seed is not None:
        return Estimate((ranged.fit, from_seed), None, "series+seed")
    return Estimate((ranged.fit,), ranged.max_x, "series, capped at its measured range")


def search_extra_words(
    store: MemoryStore,
    words: list[str],
    started: float,
    search: Callable[[list[str]], list],
    tier1: tuple[int, float] | None,
) -> list:
    """Tier 2: search the longest prefix of `words` (the selector's salience
    order) that fits what is left of the recall budget; `[]` when none does and
    no probe is due.

    `started` is `now()` at the top of the message's recall; `search` is the
    caller's own ranker call for a word list; `tier1` is this message's tier-1
    `(x1, seconds)` from `timed_tier1`, or `None`. An estimate that fits no word
    while `left > 0` searches ONE word (S156), so a too-high estimate re-samples
    itself and recovers."""
    if not words:
        return []
    left = recall_budget_seconds() - (now() - started)
    if left <= 0:
        return []
    estimate = choose_estimate(store, len(words), seed_for(tier1, words))
    if estimate is None:
        return []
    n = estimate.words_that_fit(len(words), left)
    if n < len(words):
        log.debug(
            "keyword budget: searching %d of %d extra words (left %.3fs, estimate from %s)",
            n,
            len(words),
            left,
            estimate.source,
        )
    if n <= 0:
        n = 1  # S156: left > 0 and the estimate says nothing fits: one probe word
    found, seconds = timed(lambda: search(words[:n]))
    record_tier2(store, n, seconds)
    return found


class NameWalkBudget:
    """The budget of the keyword name query's continuation walk (S145, S155,
    S157, S160), the `KeepBudget` the store's `search_fts_scored` calls around the
    walk it makes past the first page.

    `begin()` reads the saved estimate FIRST (that read is not part of the walk's
    time), then starts the walk's clock, and returns `left` = the recall budget
    minus the time already spent since `started`, or `None` when none is left (the
    continuation query is then not run). `allows(rows_judged, chars)` is asked
    before each row: no once the walk's own elapsed time has reached `left`; from
    the second judged row on, no when the predicted cost of `chars` characters (the
    rows judged including this one) exceeds `left`; with no estimate the clock
    alone bounds the walk (S160). The first row is judged whenever the clock allows
    (S155). `finish(chars)` records one sample for a walk that judged at least one
    row (x = characters judged, y = the walk's seconds), none otherwise."""

    def __init__(self, store: MemoryStore, started: float) -> None:
        self._store = store
        self._started = started
        self._fit: cost_series.LinearFit | None = None
        self._left = 0.0
        self._began = 0.0

    def begin(self) -> float | None:
        ranged = _name_cost.estimate(self._store, _NAME_KEY)
        self._fit = None if ranged is None else ranged.fit
        self._began = now()
        self._left = recall_budget_seconds() - (self._began - self._started)
        return self._left if self._left > 0 else None

    def allows(self, rows_judged: int, chars: int) -> bool:
        if now() - self._began >= self._left:
            return False
        if rows_judged >= 1 and self._fit is not None:
            return self._fit.predict(chars) <= self._left
        return True

    def finish(self, chars: int) -> None:
        if chars > 0:
            record_name_walk(self._store, chars, now() - self._began)
