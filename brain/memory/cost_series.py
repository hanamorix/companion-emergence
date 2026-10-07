"""Measured cost series over a rolling 24-hour window, saved in memories.db
(name-recall fix S146, S114, S127, S159, S163, S164).

A cost series predicts the seconds of one call from its size x as
``overhead + rate * x``, fitted by ordinary least squares over the samples
taken in the last 24 hours (`dev_constants.COST_SERIES_WINDOW_S`): a sample
counts from the moment it is taken until 24 hours later, then drops out. There
is no daily reset and nothing here ties to the compaction tick. A series whose
every size is 1 (the per-message setup `T_m`, the per-query scan `T_s`) is its
mean.

Six series use it: the rerank (`reranker.py`), the paragraph embed, the recall
setup and the paragraph scan (`semantic_recall.py`), the keyword extra words
and the name matching (`keyword_budget.py`). Each is keyed by what it measures
(a model id, a store) and, in storage, by the machine too (S159): a persona
database copied to another machine starts there as a first startup, because
this machine's random id (`machine_id`, S163) differs.

Storage (S146, S41, I9): two tables in memories.db. `cost_series_samples` holds
the samples; `cost_series_carried` holds, per key, the estimate in effect after
the last flush and the largest size it was measured at. The carried estimate is
what a window with no sample (a long downtime) still answers with, and what a
window with one size is rescaled from (S114: one factor so the carried fit's
prediction at that size equals the window's mean).

Writes (S164): a sample is only appended to an in-process buffer
(`CostSeries.record`); nothing on the reply path writes the database. The
buffer is saved by `flush(store)`: the bridge's lull-cadence job, `nell chat
--no-bridge`'s exit and the bridge's clean stop call it. A crash loses what the
buffer held (accepted). A read (`CostSeries.state`) counts the database's
window first and this process's unflushed samples second, so a flush that
completes between the two reads cannot make one sample count twice; samples
leave the buffer before they are written, so a read in that gap can miss them
for a moment but never counts one twice.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from brain import dev_constants, paths

if TYPE_CHECKING:
    from brain.memory.store import MemoryStore

log = logging.getLogger(__name__)

# S171: the series name of the rerank's RAM growth (x = padded tokens past the
# process high water, y = RSS bytes it grew by), and the series whose estimate is
# the weighted ratio of the window's sums (bytes per token) rather than a fit.
RAM_SERIES = "rerank ram"
THROUGH_ORIGIN_SERIES = frozenset({RAM_SERIES})

# Wall clock for `taken_at` and the window cut. Samples are shared with other
# processes, so the clock must be one they agree on; a jump of the system clock
# moves the window (accepted). Tests script it.
_wall_clock: Callable[[], float] = time.time


def window_seconds() -> float:
    """The window length, read now (`dev_constants.COST_SERIES_WINDOW_S`)."""
    return float(dev_constants.COST_SERIES_WINDOW_S)


# ---------------------------------------------------------------------------
# The fit.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearSums:
    """Running sums of (x, y) samples. `sum_x` / `sum_xx` are ints (x is an
    integer size), so the least-squares denominator is exact."""

    n: int = 0
    sum_x: int = 0
    sum_y: float = 0.0
    sum_xy: float = 0.0
    sum_xx: int = 0
    # The largest x among the samples (name-recall fix S139: the range the fit
    # has measured), atomic with the sums.
    max_x: int = 0

    def plus(self, x: int, y: float) -> LinearSums:
        return LinearSums(
            n=self.n + 1,
            sum_x=self.sum_x + x,
            sum_y=self.sum_y + y,
            sum_xy=self.sum_xy + x * y,
            sum_xx=self.sum_xx + x * x,
            max_x=max(self.max_x, x),
        )

    def plus_sums(self, other: LinearSums) -> LinearSums:
        """The sums of both sample sets."""
        return LinearSums(
            n=self.n + other.n,
            sum_x=self.sum_x + other.sum_x,
            sum_y=self.sum_y + other.sum_y,
            sum_xy=self.sum_xy + other.sum_xy,
            sum_xx=self.sum_xx + other.sum_xx,
            max_x=max(self.max_x, other.max_x),
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
    """Ordinary least squares over the sums (spec section 1, S67). Degenerate fits
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
class RangedFit:
    """A fit and the largest size it has measured (S139: a fit is never
    extrapolated by the keyword step, and a carried fit keeps its range)."""

    fit: LinearFit
    max_x: int


@dataclass(frozen=True)
class SeriesState:
    """One series key's state: the window's sums and the carried estimate
    (`None` before this machine's first flush for the key)."""

    sums: LinearSums = LinearSums()
    carried: RangedFit | None = None

    def estimate(self, through_origin: bool = False) -> RangedFit | None:
        """The estimate in effect (S146, S114): an empty window answers with
        the carried estimate as it is; a window with two sizes with its own
        fit; a window with one size with the carried fit rescaled by one
        factor so its prediction at that size equals the window's mean (the
        measured range is then the larger of the two); with nothing carried,
        the window's own fit. `through_origin` (a ratio series, S171): the
        window's weighted ratio of sums as the rate, no overhead, whatever
        sizes it holds; an empty window still answers with the carried ratio."""
        sums, carried = self.sums, self.carried
        if sums.n == 0:
            return carried
        if through_origin:
            if sums.sum_x <= 0:
                return carried
            return RangedFit(LinearFit(0.0, sums.sum_y / sums.sum_x, sums.n), sums.max_x)
        if carried is None or sums.has_two_sizes:
            return _own(sums)
        # One size: every sample's x equals sum_x / n.
        predicted = carried.fit.predict(sums.sum_x // sums.n)
        if predicted <= 0.0:
            return _own(sums)
        factor = (sums.sum_y / sums.n) / predicted
        return RangedFit(
            LinearFit(carried.fit.overhead * factor, carried.fit.rate * factor, sums.n),
            max(sums.max_x, carried.max_x),
        )


def _own(sums: LinearSums) -> RangedFit | None:
    fit = fit_linear(sums)
    return None if fit is None else RangedFit(fit, sums.max_x)


# ---------------------------------------------------------------------------
# Failure reporting: warn once per kind, then debug (a broken database would
# otherwise log on every message).
# ---------------------------------------------------------------------------

_warned: set[str] = set()
_warned_lock = threading.Lock()


def _note_failure(what: str, exc: BaseException) -> None:
    with _warned_lock:
        first = what not in _warned
        _warned.add(what)
    (log.warning if first else log.debug)("cost series: %s failed (%s)", what, exc, exc_info=first)


# ---------------------------------------------------------------------------
# The machine id (S159, S163).
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"[0-9a-f]{32}")
_machine_lock = threading.Lock()
_machine_cache: str | None = None
# Test seam: a fixed id so no test writes a real home directory.
_machine_override: str | None = None


def _read_machine_id(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, UnicodeDecodeError):
        return None
    return text if _ID_RE.fullmatch(text) else None


def _create_machine_id(path: Path) -> str:
    """The id kept in `path`, created on first use. The file is written whole
    to a temp name and linked into place, which is atomic and exclusive: of two
    processes racing here exactly one link wins and both read the winner's
    complete file. Filesystems without hard links fall back to an exclusive
    create."""
    existing = _read_machine_id(path)
    if existing is not None:
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = uuid.uuid4().hex
    temp = path.with_name(f"{path.name}.{fresh}.tmp")
    temp.write_text(fresh, encoding="utf-8")
    try:
        if path.exists():
            winner = _read_machine_id(path)  # another process may have just won
            if winner is not None:
                return winner
            # A file is there but holds no valid id (empty, damaged): replace it
            # whole, so persistence is not lost for good on this machine.
            os.replace(temp, path)
            return fresh
        try:
            os.link(temp, path)
        except FileExistsError:
            pass
        except OSError:  # no hard links on this filesystem
            try:
                with open(path, "x", encoding="utf-8") as handle:
                    handle.write(fresh)
            except FileExistsError:
                pass
    finally:
        temp.unlink(missing_ok=True)
    winner = _read_machine_id(path)
    if winner is None:
        raise OSError(f"machine id file {path} is not readable")
    return winner


def machine_id() -> str:
    """This machine's random id (S163): kept in `paths.get_home() /
    "machine_id"`, the app's per-user directory (it also holds `app_config.json`
    and `tunables.json`; it is outside every persona folder; `KINDLED_HOME` is
    honoured), created once, read once per process. Unreadable or unwritable:
    a process-lifetime random id and one warning (this process's samples then
    read as another machine's)."""
    global _machine_cache
    if _machine_override is not None:
        return _machine_override
    with _machine_lock:
        if _machine_cache is None:
            try:
                _machine_cache = _create_machine_id(paths.get_home() / "machine_id")
            except OSError as exc:
                _note_failure("machine id file", exc)
                _machine_cache = uuid.uuid4().hex
        return _machine_cache


# ---------------------------------------------------------------------------
# The buffer (S164).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sample:
    db: str
    series: str
    host: str
    key: str
    x: int
    seconds: float
    taken_at: float


_pending: list[_Sample] = []
_pending_lock = threading.Lock()

# Test seams, `None` in production: `_pending_hook` runs inside `flush` between
# taking the samples out of the buffer and writing them; `_between_reads_hook`
# inside `CostSeries.state` / `window_counts` between the database read and the
# buffer read.
_pending_hook: Callable[[], None] | None = None
_between_reads_hook: Callable[[], None] | None = None


def _db_key(db_path: str | os.PathLike[str]) -> str:
    return str(Path(db_path).resolve())


def _store_key(store: MemoryStore) -> str:
    """The buffer's key for a store's database: its resolved path, or, for an
    in-memory database (tests; each connection is its own database), the
    connection's identity."""
    if str(store.db_path) == ":memory:":
        return f":memory:{id(store)}"
    return _db_key(store.db_path)


def has_pending(db_path: str | os.PathLike[str]) -> bool:
    """Whether the buffer holds samples for the database at `db_path` (the
    flush job's due test; no database is opened)."""
    db = _db_key(db_path)
    with _pending_lock:
        return any(s.db == db for s in _pending)


def flush(store: MemoryStore) -> int:
    """Save this database's buffered samples (S164) and return how many were
    written. The samples leave the buffer first (under its lock) and are
    written one transaction per (series, host, key) with the store's
    `record_cost_samples`, which also prunes the key's expired rows and saves
    the carried estimate. Samples older than the window are dropped, not
    written. A group whose write raises (a locked database, a connection
    already inside a transaction) goes back into the buffer; nothing is lost
    and nothing is written twice. Never raises for a failed write."""
    db = _store_key(store)
    now = _wall_clock()
    window = window_seconds()
    with _pending_lock:
        mine = [s for s in _pending if s.db == db]
        if mine:
            _pending[:] = [s for s in _pending if s.db != db]
    if _pending_hook is not None:
        _pending_hook()
    groups: dict[tuple[str, str, str], list[_Sample]] = {}
    for sample in mine:
        if sample.taken_at > now - window:
            groups.setdefault((sample.series, sample.host, sample.key), []).append(sample)
    written = 0
    pending = list(groups.items())
    for index, ((series, host, key), group) in enumerate(pending):
        try:
            store.record_cost_samples(
                series,
                host,
                key,
                [(s.x, s.seconds, s.taken_at) for s in group],
                now=now,
                window_seconds=window,
            )
            written += len(group)
        except (sqlite3.OperationalError, RuntimeError) as exc:
            _note_failure("flush", exc)
            # A locked or busy database (or a connection inside someone else's
            # transaction, `record_cost_samples`' RuntimeError): put this group and every group not yet
            # tried back, and stop (it is locked for the rest too, and a stop must
            # not wait out the busy timeout once per series).
            with _pending_lock:
                for _unwritten_key, unwritten in pending[index:]:
                    _pending.extend(unwritten)
            break
        except Exception as exc:  # noqa: BLE001 - a failure of this group alone
            # Not a lock: retrying would fail the same way at the head of every
            # flush and starve the other series, so this group is dropped.
            _note_failure("flush (group dropped)", exc)
    return written


def _reset_pending() -> None:
    """Test-only: empty the buffer and re-arm the warn-once flags."""
    with _pending_lock:
        _pending.clear()
    with _warned_lock:
        _warned.clear()


def _reset_machine_id() -> None:
    """Test-only: forget the cached machine id."""
    global _machine_cache
    with _machine_lock:
        _machine_cache = None


# ---------------------------------------------------------------------------
# The series.
# ---------------------------------------------------------------------------


class CostSeries:
    """One named series, keyed by what it measures (a model id, a store path;
    `""` for a series with one key). Holds no state: the samples live in the
    process buffer and in memories.db."""

    def __init__(self, name: str) -> None:
        self.name = name

    def record(self, store: MemoryStore, key: str, x: int, seconds: float) -> None:
        """Buffer one sample (no database access: S164)."""
        self.record_many(store, key, [(x, seconds)])

    def record_many(
        self, store: MemoryStore, key: str, samples: Iterable[tuple[int, float]]
    ) -> None:
        """Buffer several samples taken now."""
        now = _wall_clock()
        host = machine_id()
        db = _store_key(store)
        rows = [_Sample(db, self.name, host, key, x, max(0.0, y), now) for x, y in samples]
        if rows:
            with _pending_lock:
                _pending.extend(rows)

    def _buffered(self, db: str, host: str, key: str, cut: float) -> list[_Sample]:
        with _pending_lock:
            return [
                s
                for s in _pending
                if s.db == db
                and s.series == self.name
                and s.host == host
                and s.key == key
                and s.taken_at > cut
            ]

    def state(self, store: MemoryStore, key: str) -> SeriesState | None:
        """The window's sums (database first, then this process's buffer) and
        the carried estimate. `None` when the database read fails (a failed
        read is "no estimate"; warned once, then debug)."""
        now = _wall_clock()
        window = window_seconds()
        host = machine_id()
        try:
            sums, carried = store.cost_series_state(
                self.name, host, key, now=now, window_seconds=window
            )
        except Exception as exc:  # noqa: BLE001 - fail soft: no estimate
            _note_failure("read", exc)
            return None
        if _between_reads_hook is not None:
            _between_reads_hook()
        for sample in self._buffered(_store_key(store), host, key, now - window):
            sums = sums.plus(sample.x, sample.seconds)
        return SeriesState(sums, carried)

    def estimate(self, store: MemoryStore, key: str) -> RangedFit | None:
        """The estimate in effect (see `SeriesState.estimate`)."""
        state = self.state(store, key)
        return None if state is None else state.estimate(self.name in THROUGH_ORIGIN_SERIES)

    def fit(self, store: MemoryStore, key: str) -> LinearFit | None:
        found = self.estimate(store, key)
        return None if found is None else found.fit

    def mean(self, store: MemoryStore, key: str) -> float | None:
        """The window's mean seconds for an every-size-1 series (`T_m`,
        `T_s`), or `None` before any measurement."""
        fit = self.fit(store, key)
        return None if fit is None else fit.predict(1)

    def window_counts(self, store: MemoryStore, key: str, since: float) -> tuple[int, int]:
        """`(before, after)`: how many window samples (database first, then this
        process's buffer) were taken at or before `since`, and after it. S127
        asks whether the estimate has changed since a verdict by asking
        whether this pair has. Raises when the database read fails."""
        now = _wall_clock()
        window = window_seconds()
        host = machine_id()
        before, after = store.cost_series_window_counts(
            self.name, host, key, since=since, now=now, window_seconds=window
        )
        if _between_reads_hook is not None:
            _between_reads_hook()
        for sample in self._buffered(_store_key(store), host, key, now - window):
            if sample.taken_at > since:
                after += 1
            else:
                before += 1
        return before, after
