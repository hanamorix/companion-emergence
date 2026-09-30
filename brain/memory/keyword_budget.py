"""The keyword extra-words step and the per-message recall time budget
(name-recall fix S103, S130-S134; spec section 7, "Keyword step 3").

Passive recall's keyword search runs the name query, tier 1 (today's capped
10-word query) and tier 2, the "extra words" (every selector token tier 1 does
not reach). Tier 2 searches as many of those words as fit what is left of the
per-message recall time budget, in the selector's own salience order, and
drops the rest for that turn. Names and tier 1 always run.

"What is left" (S132) is literal: the budget (the `reranker.latency_budget_seconds`
tunable, the one the paragraph search and the rerank share) minus the wall time
this message's recall has already spent, measured on one monotonic clock from
the top of `_build_recall_block`.

The cost of a search is predicted as `overhead + rate * words` (S101, S67), a
least-squares fit over this period's running sums (`cost_period`), fed by
tier-2 searches ONLY (S131), reset at the 24-hour compaction tick with the
previous fit carried (S101, S114). The first tier-2 sample of a process is
discarded (S134), as the rerank and embed series discard their first call.

Until the period has a tier-2 sample the estimate comes from this message's own
tier-1 search (the "seed", S131: tier 1 always runs first, so tier 2 never runs
unbounded): alone in a fresh process, the lower of it and the carried fit after
a tick (S133). The seed is per message and is never stored. With no estimate at
all (tier 1 raised or searched no word) no extra word is searched.

The search tool is not affected (S99): its keyword side never comes here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence

from brain import tunables
from brain.memory import cost_period
from brain.memory import reranker as reranker_mod
from brain.memory.store import MemoryStore, split_by_raw_query_floor

log = logging.getLogger(__name__)

# Clock seam for the per-message budget and the cost samples. Production:
# `time.perf_counter` (monotonic, and fine-grained on every platform: a coarse
# clock would read a short search as 0.0 s and turn the estimate into "every word
# fits"). Tests script it.
_clock: Callable[[], float] = time.perf_counter

# The per-word keyword cost (S103, S131): per store database path, one sample
# per tier-2 search, x = words sent to the store, y = that call's seconds.
_keyword_cost = cost_period.PeriodCostSeries("keyword extra words")

# S134: the first tier-2 sample of this process is discarded. One process-wide
# flag under its own lock; the compaction tick does not re-arm it.
_first_sample_lock = threading.Lock()
_first_sample_spent = False

# Test-only seams: `_record_hook` is called at the top of `record_tier2`, before
# the flag check; `_flag_hook` inside the lock, between the check and the set.
_record_hook: Callable[[], None] | None = None
_flag_hook: Callable[[], None] | None = None


def _reset_keyword_cost() -> None:
    """Test-only: forget the series and re-arm the first-sample discard."""
    global _first_sample_spent
    _keyword_cost.clear()
    with _first_sample_lock:
        _first_sample_spent = False


def now() -> float:
    """The budget clock's current reading."""
    return _clock()


def recall_budget_seconds() -> float:
    """The per-message recall time budget, read now (the same tunable and
    default `search_paragraphs` reads)."""
    return float(
        tunables.get_tunable("reranker.latency_budget_seconds", reranker_mod.LATENCY_BUDGET_SECONDS)
    )


def words_that_fit(fit: cost_period.LinearFit, count: int, left: float) -> int:
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
) -> tuple[T, tuple[int, float] | None]:
    """Run tier 1's own search and return `(its result, the seed)`. The seed is
    `(x1, seconds)`: `x1` = the words the store actually searches in tier 1
    (its 3-character floor applied), `seconds` = the whole call (S131: this
    message's own tier-1 search). `None` when tier 1 searched no word. An
    exception from `call` propagates; nothing is recorded, tier 1 is unchanged."""
    x1 = len(split_by_raw_query_floor(list(legacy_tokens))[0])
    result, seconds = timed(call)
    return result, ((x1, seconds) if x1 > 0 else None)


def seed_fit(words: int, seconds: float) -> cost_period.LinearFit | None:
    """The per-message estimate from tier 1's own timing: seconds per word
    through the origin (the one-size form `cost_period.fit_linear` falls back
    to). `None` for zero words. Never stored."""
    if words <= 0:
        return None
    return cost_period.LinearFit(0.0, seconds / words, 1)


def record_tier2(key: str, words: int, seconds: float) -> None:
    """Record one tier-2 sample, except the process's first (S134)."""
    global _first_sample_spent
    if _record_hook is not None:
        _record_hook()
    with _first_sample_lock:
        if not _first_sample_spent:
            if _flag_hook is not None:
                _flag_hook()
            _first_sample_spent = True
            return
    _keyword_cost.record(key, words, seconds)


def search_extra_words(
    store: MemoryStore,
    words: list[str],
    started: float,
    search: Callable[[list[str]], list],
    seed: tuple[int, float] | None,
) -> list:
    """Tier 2: search the longest prefix of `words` (the selector's salience
    order) that fits what is left of the recall budget; `[]` when none does.

    `started` is `now()` at the top of the message's recall; `search` is the
    caller's own ranker call for a word list; `seed` is this message's tier-1
    `(x1, seconds)` or `None`."""
    if not words:
        return []
    left = recall_budget_seconds() - (now() - started)
    if left <= 0:
        return []
    key = str(store.db_path)
    state = _keyword_cost.state(key)  # read once: sizing and sample share it
    candidates: list[cost_period.LinearFit] = []
    if state is not None and state.sums.n > 0:
        own = state.fit()
        if own is not None:
            candidates.append(own)
        source = "series"
    else:
        # No tier-2 sample in this period yet (S131, S133): the carried fit
        # (after a tick) and this message's tier-1 seed; the lower cost wins.
        if state is not None and state.carried is not None:
            candidates.append(state.carried)
        from_seed = seed_fit(*seed) if seed is not None else None
        if from_seed is not None:
            candidates.append(from_seed)
        source = "carried/seed"
    if not candidates:
        return []
    n = max(words_that_fit(c, len(words), left) for c in candidates)
    if n < len(words):
        log.debug(
            "keyword budget: searching %d of %d extra words (left %.3fs, estimate from %s)",
            n,
            len(words),
            left,
            source,
        )
    if n <= 0:
        return []
    found, seconds = timed(lambda: search(words[:n]))
    record_tier2(key, n, seconds)
    return found
