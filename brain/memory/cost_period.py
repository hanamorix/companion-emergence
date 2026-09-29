"""Measured cost series averaged per 24-hour compaction period (name-recall
fix S101, S114, S118).

A cost series predicts the seconds of one call from its size x as
``overhead + rate * x``, fitted by ordinary least squares over running sums
(count, sum x, sum y, sum x*y, sum x*x). The sums cover only the time since
the last 24-hour compaction tick: at the tick (`start_new_period`, called by
the bridge's `compaction` gated job) every registered series empties its sums
and keeps the fit that was in effect as the CARRIED fit. Until the new period
has measurements at two different sizes (the minimum a two-parameter fit
needs), the estimate is the carried fit, rescaled by one factor so that its
prediction at the one size measured so far equals the mean of those
measurements (S114); from the second size on, the new period's own fit takes
over (S101). The first period of a process has no carried fit and behaves as
a plain running fit. A `nell chat --no-bridge` process runs no supervisor, so
it never calls `start_new_period` and keeps its process-lifetime sums.

Series registered here: the reranker's cost model (`reranker.py`, its own
lock and record type, hooked with `on_new_period`) and the paragraph embed
cost (`semantic_recall.py`, a `PeriodCostSeries`). A later series (S103's
per-word keyword cost) is one more `PeriodCostSeries`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearSums:
    """Running sums of (x, y) samples. `sum_x` / `sum_xx` are ints (x is an
    integer token count), so the least-squares denominator is exact."""

    n: int = 0
    sum_x: int = 0
    sum_y: float = 0.0
    sum_xy: float = 0.0
    sum_xx: int = 0

    def plus(self, x: int, y: float) -> LinearSums:
        return LinearSums(
            n=self.n + 1,
            sum_x=self.sum_x + x,
            sum_y=self.sum_y + y,
            sum_xy=self.sum_xy + x * y,
            sum_xx=self.sum_xx + x * x,
        )

    @property
    def has_two_sizes(self) -> bool:
        """True iff the samples include at least two distinct x."""
        return self.n * self.sum_xx - self.sum_x * self.sum_x > 0


@dataclass(frozen=True)
class LinearFit:
    """`overhead + rate * x` seconds; `samples` = the measurements behind it."""

    overhead: float
    rate: float
    samples: int

    def predict(self, x: int) -> float:
        return self.overhead + self.rate * x


def fit_linear(sums: LinearSums) -> LinearFit | None:
    """Ordinary least squares over the sums (spec §1, S67). Degenerate fits
    fall back to a zero overhead and the through-origin ratio of sums: a
    single size, a negative fitted overhead, or a non-positive fitted rate.
    `None` before any sample with a non-zero size."""
    if sums.n == 0 or sums.sum_x <= 0:
        return None
    denominator = sums.n * sums.sum_xx - sums.sum_x * sums.sum_x
    if denominator > 0:
        rate = (sums.n * sums.sum_xy - sums.sum_x * sums.sum_y) / denominator
        overhead = (sums.sum_y - rate * sums.sum_x) / sums.n
        if overhead >= 0.0 and rate > 0.0:
            return LinearFit(overhead, rate, sums.n)
    return LinearFit(0.0, sums.sum_y / sums.sum_x, sums.n)


@dataclass(frozen=True)
class PeriodSums:
    """One series key's state: this period's sums and the fit carried in
    from the previous period (`None` in a process's first period)."""

    sums: LinearSums = LinearSums()
    carried: LinearFit | None = None

    def plus(self, x: int, y: float) -> PeriodSums:
        return PeriodSums(self.sums.plus(x, y), self.carried)

    def fit(self) -> LinearFit | None:
        """The estimate in effect (S101, S114); see the module docstring."""
        sums, carried = self.sums, self.carried
        if carried is None or sums.has_two_sizes:
            return fit_linear(sums)
        if sums.n == 0:
            return carried
        # One size so far: every sample's x equals sum_x / n.
        predicted = carried.predict(sums.sum_x // sums.n)
        if predicted <= 0.0:
            return fit_linear(sums)
        factor = (sums.sum_y / sums.n) / predicted
        return LinearFit(carried.overhead * factor, carried.rate * factor, sums.n)

    def next_period(self) -> PeriodSums:
        """The state after a compaction tick: empty sums, the fit in effect
        carried (itself a rescaled carry when this period never reached two
        sizes)."""
        return PeriodSums(LinearSums(), self.fit())


# ---------------------------------------------------------------------------
# The period hook list.
# ---------------------------------------------------------------------------

_hooks: list[Callable[[], None]] = []
_hooks_lock = threading.Lock()


def on_new_period(hook: Callable[[], None]) -> Callable[[], None]:
    """Register `hook` to run at every `start_new_period()`; returns a
    callable that unregisters it."""
    with _hooks_lock:
        _hooks.append(hook)

    def unregister() -> None:
        with _hooks_lock:
            if hook in _hooks:
                _hooks.remove(hook)

    return unregister


def start_new_period() -> None:
    """Start a new cost period in every registered series (the 24-hour
    compaction tick, S101). Hooks run outside the list's lock, in order; one
    failing hook is logged and the others still run."""
    with _hooks_lock:
        hooks = list(_hooks)
    for hook in hooks:
        try:
            hook()
        except Exception:  # noqa: BLE001 — one series must not block the others
            log.exception("cost period: a series failed to start its new period")
    log.info("cost period: new period started for %d series", len(hooks))


# Test-only seam (CONC-7): called inside `PeriodCostSeries.record`, between
# reading the key's state and writing the new one. `None` in production.
_update_hook: Callable[[], None] | None = None


class PeriodCostSeries:
    """A keyed cost series (e.g. per model id) with period sums. One lock
    guards every read and read-modify-write, and `new_period` swaps every
    key's state under the same lock, so a reset never lands inside a record
    and a reader never sees half a sample. Registers itself with the period
    hook list; `unregister()` removes it (tests)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._states: dict[str, PeriodSums] = {}
        self.unregister = on_new_period(self.new_period)

    def record(self, key: str, x: int, y: float) -> None:
        with self._lock:
            current = self._states.get(key, PeriodSums())
            if _update_hook is not None:
                _update_hook()
            self._states[key] = current.plus(x, max(0.0, y))

    def fit(self, key: str) -> LinearFit | None:
        with self._lock:
            state = self._states.get(key)
        return None if state is None else state.fit()

    def state(self, key: str) -> PeriodSums | None:
        with self._lock:
            return self._states.get(key)

    def new_period(self) -> None:
        with self._lock:
            self._states = {key: state.next_period() for key, state in self._states.items()}

    def clear(self) -> None:
        """Test-only: forget every key."""
        with self._lock:
            self._states = {}
