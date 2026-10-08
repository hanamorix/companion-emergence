"""Cost-series increment (name-recall fix S146, S114, S159, S163, S164): the
rolling 24-hour window, the carried estimate, the two storage tables, the
process buffer and its flush, the machine id.

Oracles are independent: numpy least squares (`np.linalg.lstsq`) or values
derived by hand in the test, never the implementation's own formula. Tests run
against REAL SQLite files (`tmp_path`); "scripted wall clock" is
`cost_series._wall_clock` set by the test.

Criteria: CS1, CS3c/d, CS4, CS6a/b/c/e, CS7 (unit half), CS17b (the flush's
own half) in `changes/name-recall-fix/cost-series/1.5-criteria.md`.
"""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from brain import dev_constants
from brain.memory import cost_series
from brain.memory.cost_series import (
    CostSeries,
    LinearFit,
    LinearSums,
    RangedFit,
    SeriesState,
    fit_linear,
)
from brain.memory.store import MemoryStore

W = dev_constants.COST_SERIES_WINDOW_S
T0 = 1_000_000.0


def _ols(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Independent (overhead, rate) by numpy least squares."""
    a = np.array([[1.0, float(x)] for x, _ in points])
    b = np.array([y for _, y in points])
    (overhead, rate), *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(overhead), float(rate)


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(cost_series, "_wall_clock", c)
    return c


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    yield s
    s.close()


@pytest.fixture
def series() -> CostSeries:
    return CostSeries("test series")


def _state(points: list[tuple[int, float]], carried: RangedFit | None = None) -> SeriesState:
    sums = LinearSums()
    for x, y in points:
        sums = sums.plus(x, y)
    return SeriesState(sums, carried)


def _record(series: CostSeries, store: MemoryStore, points: list[tuple[int, float]], key: str = "m") -> None:
    for x, y in points:
        series.record(store, key, x, y)


# ---------------------------------------------------------------------------
# The estimate (fit maths moved from the period module; CS4a/b, S114).
# ---------------------------------------------------------------------------


def test_a_window_with_no_carry_is_the_plain_least_squares_fit() -> None:
    points = [(1000, 0.7), (3000, 1.5), (2000, 1.1)]
    est = _state(points).estimate()
    overhead, rate = _ols(points)
    assert est is not None and est.fit.samples == 3 and est.max_x == 3000
    assert est.fit.overhead == pytest.approx(overhead) and est.fit.rate == pytest.approx(rate)
    one = _state([(1000, 0.5), (1000, 0.7)]).estimate()
    assert one is not None and one.fit == LinearFit(0.0, 1.2 / 2000, 2), "one size, no carry: ratio of sums"
    assert SeriesState().estimate() is None


def test_cs4_an_empty_window_answers_with_the_carried_estimate_as_it_is() -> None:
    carried = RangedFit(LinearFit(0.1, 0.0002, 9), 5000)
    assert SeriesState(LinearSums(), carried).estimate() == carried


def test_cs4a_one_size_rescales_the_carried_fit_by_one_factor() -> None:
    carried = RangedFit(LinearFit(0.1, 0.0002, 9), 3000)  # 0.5 s at x = 2000
    est = _state([(2000, 1.0)], carried).estimate()
    assert est is not None
    assert est.fit.predict(2000) == pytest.approx(1.0)
    assert est.fit.overhead == pytest.approx(0.2) and est.fit.rate == pytest.approx(0.0004), "factor 1.0 / 0.5 = 2"
    assert est.max_x == 3000, "the larger of the window's and the carried range"
    three = _state([(2000, 1.0), (2000, 1.2), (2000, 1.4)], carried).estimate()
    assert three is not None and three.fit.predict(2000) == pytest.approx(1.2)
    # Two samples of ONE size still rescale (the carried shape), not a fit.
    two = _state([(2000, 1.0), (2000, 1.0)], RangedFit(LinearFit(0.1, 0.0002, 9), 3000)).estimate()
    assert two is not None and two.fit.predict(4000) == pytest.approx(1.0 * (0.1 + 0.8) / 0.5)


def test_cs4b_two_sizes_take_the_windows_own_fit() -> None:
    carried = RangedFit(LinearFit(0.1, 0.0002, 9), 3000)
    points = [(2000, 1.0), (2000, 1.2), (2000, 1.4), (4000, 2.0)]
    est = _state(points, carried).estimate()
    overhead, rate = _ols(points)
    assert est is not None and est.fit.overhead == pytest.approx(overhead) and est.fit.rate == pytest.approx(rate)
    assert est.max_x == 4000


def test_a_carried_fit_with_no_usable_factor_falls_back_to_the_new_ratio() -> None:
    est = _state([(1000, 0.5)], RangedFit(LinearFit(0.0, 0.0, 1), 1000)).estimate()
    assert est is not None and est.fit == LinearFit(0.0, 0.5 / 1000, 1)


def test_fit_linear_degenerate_cases() -> None:
    assert fit_linear(LinearSums()) is None
    sums = LinearSums().plus(1000, 0.1).plus(2000, 1.0)  # a negative fitted overhead
    assert fit_linear(sums) == LinearFit(0.0, 1.1 / 3000, 2)


# ---------------------------------------------------------------------------
# CS1: the rolling window; buffered and flushed samples alike.
# ---------------------------------------------------------------------------


def _assert_fit(fit: LinearFit | None, points: list[tuple[int, float]]) -> None:
    overhead, rate = _ols(points)
    assert fit is not None
    assert fit.overhead == pytest.approx(overhead, abs=1e-9) and fit.rate == pytest.approx(rate, abs=1e-12)


@pytest.mark.parametrize("flushed", [True, False])
def test_cs1a_a_sample_counts_for_exactly_the_window(
    store: MemoryStore, series: CostSeries, clock: Clock, flushed: bool
) -> None:
    points = [(1000, 0.3), (3000, 0.7), (2000, 0.5001)]
    _record(series, store, points)
    if flushed:
        assert cost_series.flush(store) == 3
    clock.now = T0 + W - 1
    _assert_fit(series.fit(store, "m"), points)
    clock.now = T0 + W
    state = series.state(store, "m")
    assert state is not None and state.sums.n == 0, "at exactly t + W the samples no longer count"
    if flushed:
        carried = series.estimate(store, "m")
        assert carried is not None and carried.max_x == 3000, "the carried estimate applies (CS4c)"
        _assert_fit(carried.fit, points)
    else:
        assert series.estimate(store, "m") is None, "never flushed: nothing carried either"


@pytest.mark.parametrize("flushed", [True, False])
def test_cs1b_only_the_later_samples_count_when_the_earlier_age_out(
    store: MemoryStore, series: CostSeries, clock: Clock, flushed: bool
) -> None:
    early = [(1000, 0.3), (3000, 0.7)]
    late = [(1000, 0.35), (5000, 1.25), (2000, 0.55)]
    _record(series, store, early)
    clock.now = T0 + W / 2
    _record(series, store, late)
    if flushed:
        cost_series.flush(store)
    clock.now = T0 + W
    _assert_fit(series.fit(store, "m"), late)
    clock.now = T0 + W - 1
    _assert_fit(series.fit(store, "m"), early + late)


def test_cs1_mean_series_is_the_mean_of_the_window(store: MemoryStore, series: CostSeries, clock: Clock) -> None:
    for seconds in (0.2, 0.4, 0.9):
        series.record(store, "", 1, seconds)
    cost_series.flush(store)
    clock.now = T0 + W - 1
    assert series.mean(store, "") == pytest.approx(0.5)


def test_cs1_a_negative_sample_is_clamped_to_zero(store: MemoryStore, series: CostSeries) -> None:
    series.record(store, "m", 1, -0.5)
    series.record(store, "m", 1, 0.5)
    assert series.mean(store, "m") == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# CS2: a restart continues from the saved series; a second connection counts.
# ---------------------------------------------------------------------------


def test_cs2_restart_equal_estimate_and_a_second_connections_sample_counts(
    tmp_path: Path, series: CostSeries, clock: Clock
) -> None:
    path = tmp_path / "memories.db"
    first = MemoryStore(path, integrity_check=False)
    _record(series, first, [(1000, 0.3), (3000, 0.7)])
    cost_series.flush(first)
    before = series.estimate(first, "m")
    first.close()
    cost_series._reset_pending()  # a new process: no in-process state
    second = MemoryStore(path, integrity_check=False)
    try:
        assert series.estimate(second, "m") == before
        other = MemoryStore(path, integrity_check=False)
        try:
            _record(series, other, [(2000, 0.6)])
            cost_series.flush(other)
        finally:
            other.close()
        _assert_fit(series.fit(second, "m"), [(1000, 0.3), (3000, 0.7), (2000, 0.6)])
    finally:
        second.close()


# ---------------------------------------------------------------------------
# CS3c: another machine; CS3d: the machine id.
# ---------------------------------------------------------------------------


def test_cs3c_another_machine_starts_as_a_first_startup(
    store: MemoryStore, series: CostSeries, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(series, store, [(1000, 0.3), (3000, 0.7)])
    cost_series.flush(store)
    mine = series.estimate(store, "m")
    assert mine is not None
    monkeypatch.setattr(cost_series, "_machine_override", "another-machine")
    assert series.estimate(store, "m") is None, "no window and no carried row for this host"
    _record(series, store, [(500, 9.0)])
    cost_series.flush(store)
    monkeypatch.setattr(cost_series, "_machine_override", "test-machine")
    assert series.estimate(store, "m") == mine, "the other machine's samples never change this one's"


@pytest.fixture
def real_machine_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("KINDLED_HOME", str(home))
    monkeypatch.setattr(cost_series, "_machine_override", None)
    cost_series._reset_machine_id()
    return home


def test_cs3d_machine_id_is_created_once_in_the_app_home_outside_personas(real_machine_id: Path) -> None:
    first = cost_series.machine_id()
    assert len(first) == 32 and (real_machine_id / "machine_id").read_text() == first
    assert not (real_machine_id / "personas").exists(), "never under the persona folders"
    assert not list(real_machine_id.glob("*.tmp")), "the temp file is gone"
    assert cost_series.machine_id() == first
    cost_series._reset_machine_id()
    assert cost_series.machine_id() == first, "stable across a fresh read of the file"


def test_cs3d_machine_id_is_stable_across_a_fresh_process(real_machine_id: Path) -> None:
    first = cost_series.machine_id()
    code = "from brain.memory import cost_series; print(cost_series.machine_id())"
    env_home = str(real_machine_id)
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={"KINDLED_HOME": env_home, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert out.stdout.strip() == first


def test_cs3d_two_processes_racing_the_first_creation_end_with_one_id(real_machine_id: Path) -> None:
    code = (
        "import sys, time\n"
        "from brain.memory import cost_series\n"
        "while time.time() < float(sys.argv[1]): pass\n"
        "print(cost_series.machine_id())\n"
    )
    import os
    import time

    go = str(time.time() + 6.0)
    env = {"KINDLED_HOME": str(real_machine_id), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    procs = [
        subprocess.Popen([sys.executable, "-c", code, go], env=env, stdout=subprocess.PIPE, text=True)
        for _ in range(3)
    ]
    ids = [p.communicate(timeout=120)[0].strip() for p in procs]
    assert len(set(ids)) == 1 and len(ids[0]) == 32
    assert (real_machine_id / "machine_id").read_text() == ids[0]
    assert not list(real_machine_id.glob("*.tmp"))


def test_cs3d_a_losing_racer_reads_the_winners_complete_file(
    real_machine_id: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The create is atomic: when the link finds the file already there (another
    process won), the loser returns the winner's id, never its own."""
    winner = "a" * 32
    real_link = cost_series.os.link

    def link_after_a_winner(src, dst):
        Path(dst).write_text(winner)  # the other process finished first
        return real_link(src, dst)  # raises FileExistsError

    monkeypatch.setattr(cost_series.os, "link", link_after_a_winner)
    assert cost_series.machine_id() == winner
    assert not list(real_machine_id.glob("*.tmp"))


def test_cs3d_an_unwritable_home_gives_a_process_lifetime_id_and_one_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the home directory should be")
    monkeypatch.setenv("KINDLED_HOME", str(blocked))
    monkeypatch.setattr(cost_series, "_machine_override", None)
    cost_series._reset_machine_id()
    with caplog.at_level("DEBUG", logger="brain.memory.cost_series"):
        first = cost_series.machine_id()
        assert cost_series.machine_id() == first and len(first) == 32
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# CS4c, CS7 (unit half): the carried row; the tables on open.
# ---------------------------------------------------------------------------


def test_cs4c_each_flush_saves_the_estimate_in_effect_and_its_range(
    store: MemoryStore, series: CostSeries, clock: Clock
) -> None:
    _record(series, store, [(1000, 0.3), (3000, 0.7)])
    cost_series.flush(store)
    sums, carried = store.cost_series_state(series.name, "test-machine", "m", now=clock.now, window_seconds=W)
    assert sums.n == 2 and carried is not None and carried.max_x == 3000
    _assert_fit(carried.fit, [(1000, 0.3), (3000, 0.7)])
    clock.now = T0 + 100
    series.record(store, "m", 5000, 1.1)
    cost_series.flush(store)
    _, carried = store.cost_series_state(series.name, "test-machine", "m", now=clock.now, window_seconds=W)
    assert carried is not None and carried.max_x == 5000
    _assert_fit(carried.fit, [(1000, 0.3), (3000, 0.7), (5000, 1.1)])


def test_cs4c_the_mean_series_carried_row_equals_the_window_mean(
    store: MemoryStore, series: CostSeries, clock: Clock
) -> None:
    for seconds in (0.2, 0.4, 0.9):
        series.record(store, "", 1, seconds)
    cost_series.flush(store)
    _, carried = store.cost_series_state(series.name, "test-machine", "", now=clock.now, window_seconds=W)
    assert carried is not None and carried.fit.predict(1) == pytest.approx(0.5)


def test_cs4_a_carried_one_size_flush_rescales_once_not_compounding(
    store: MemoryStore, series: CostSeries, clock: Clock
) -> None:
    _record(series, store, [(1000, 0.3), (3000, 0.7)])
    cost_series.flush(store)
    clock.now = T0 + W + 10  # both samples aged out; the carried fit stands
    _record(series, store, [(2000, 1.0)])
    cost_series.flush(store)
    _, once = store.cost_series_state(series.name, "test-machine", "m", now=clock.now, window_seconds=W)
    assert once is not None and once.fit.predict(2000) == pytest.approx(1.0)
    series.record(store, "m", 2000, 1.0)
    cost_series.flush(store)
    _, twice = store.cost_series_state(series.name, "test-machine", "m", now=clock.now, window_seconds=W)
    assert twice is not None and twice.fit.predict(2000) == pytest.approx(1.0)
    assert twice.fit.rate == pytest.approx(once.fit.rate), "the rescale telescopes"


def _table_hashes(path: Path) -> dict[str, str]:
    con = sqlite3.connect(path)
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        out = {}
        for name in names:
            if name.startswith("cost_series") or name.endswith(("_content", "_data", "_idx", "_docsize", "_config")):
                continue
            rows = con.execute(f'SELECT * FROM "{name}" ORDER BY 1').fetchall()
            out[name] = hashlib.sha256(repr(rows).encode()).hexdigest()
        return out
    finally:
        con.close()


def test_cs7_a_legacy_database_gains_both_tables_and_the_index_and_loses_nothing(tmp_path: Path) -> None:
    path = tmp_path / "memories.db"
    seeded = MemoryStore(path, integrity_check=False)
    from brain.memory.store import Memory

    seeded.create(Memory.create_new(content="a legacy row", memory_type="conversation", domain="x"))
    seeded.close()
    con = sqlite3.connect(path)
    con.executescript(
        "DROP INDEX idx_cost_series_samples; DROP TABLE cost_series_samples; DROP TABLE cost_series_carried;"
    )
    con.commit()
    con.close()
    before = _table_hashes(path)
    reopened = MemoryStore(path, integrity_check=False)
    reopened.close()
    con = sqlite3.connect(path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    indexes = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    con.close()
    assert {"cost_series_samples", "cost_series_carried"} <= tables
    assert "idx_cost_series_samples" in indexes
    assert _table_hashes(path) == before, "every pre-existing table's rows are unchanged"
    MemoryStore(path, integrity_check=False).close()  # idempotent
    assert _table_hashes(path) == before


def test_cs7_a_flush_on_a_connection_inside_a_transaction_raises_and_keeps_the_samples(
    store: MemoryStore, series: CostSeries
) -> None:
    _record(series, store, [(1000, 0.3)])
    store._conn.execute("BEGIN")
    store._conn.execute("INSERT INTO cost_series_samples VALUES ('s', 'h', 'k', 1, 0.1, 1.0)")
    with pytest.raises(RuntimeError):
        store.record_cost_samples("s", "h", "k", [(1, 0.1, 1.0)], now=T0, window_seconds=W)
    assert cost_series.flush(store) == 0, "flush fails soft"
    assert cost_series.has_pending(store.db_path), "and the samples are back in the buffer"
    store._conn.rollback()
    assert cost_series.flush(store) == 1 and not cost_series.has_pending(store.db_path)


# ---------------------------------------------------------------------------
# CS17b (the flush's half): what is written, dropped, kept.
# ---------------------------------------------------------------------------


def test_cs17b_flush_writes_empties_and_drops_aged_samples(
    store: MemoryStore, series: CostSeries, clock: Clock
) -> None:
    _record(series, store, [(1000, 0.3)])
    clock.now = T0 + W + 5
    _record(series, store, [(2000, 0.5), (3000, 0.7)])
    assert cost_series.has_pending(store.db_path)
    assert cost_series.flush(store) == 2, "the aged sample is dropped, not written"
    assert not cost_series.has_pending(store.db_path)
    n = store._conn.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0]
    assert n == 2


def test_cs17b_flush_drops_a_sample_that_aged_while_nothing_was_appended(
    store: MemoryStore, series: CostSeries, clock: Clock
) -> None:
    """Without a later append the buffer's own prune never runs (S176), so the flush's own filter is
    what keeps an aged sample out of the database (the prune must not stand in for it)."""
    _record(series, store, [(1000, 0.3), (2000, 0.5)])
    clock.now = T0 + W + 5
    assert cost_series.flush(store) == 0
    assert store._conn.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0] == 0  # noqa: SLF001
    assert not cost_series.has_pending(store.db_path)


def test_cs17b_a_failing_write_puts_the_samples_back(
    store: MemoryStore, series: CostSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(series, store, [(1000, 0.3), (3000, 0.7)])

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "record_cost_samples", boom)
    assert cost_series.flush(store) == 0
    assert cost_series.has_pending(store.db_path)
    monkeypatch.undo()
    assert cost_series.flush(store) == 2


def test_samples_of_another_database_stay_in_the_buffer(tmp_path: Path, series: CostSeries) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = MemoryStore(tmp_path / "a" / "memories.db", integrity_check=False)
    b = MemoryStore(tmp_path / "b" / "memories.db", integrity_check=False)
    try:
        series.record(a, "m", 1000, 0.3)
        series.record(b, "m", 1000, 0.4)
        assert cost_series.flush(a) == 1
        assert cost_series.has_pending(b.db_path) and not cost_series.has_pending(a.db_path)
        fit_b = series.fit(b, "m")
        assert fit_b is not None and fit_b.predict(1000) == pytest.approx(0.4)
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------------------
# CS6: two writers, one buffer.
# ---------------------------------------------------------------------------


def test_cs6a_no_lost_update_when_a_second_flush_starts_inside_the_first(
    tmp_path: Path, series: CostSeries, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "memories.db"
    a = MemoryStore(path, integrity_check=False)
    first = [(1000, 0.3), (3000, 0.7)]
    second = [(2000, 0.9)]
    _record(series, a, first)
    b_returned = threading.Event()
    b_error: list[BaseException] = []

    def flush_b() -> None:
        try:
            b = MemoryStore(path, integrity_check=False)
            try:
                _record(series, b, second)
                cost_series.flush(b)
            finally:
                b.close()
        except BaseException as exc:  # noqa: BLE001
            b_error.append(exc)
        finally:
            b_returned.set()

    started: list[threading.Thread] = []

    def hook() -> None:
        # Inside A's write transaction, between reading the window and writing
        # the carried row: B tries to flush through its own connection. In the
        # correct code B blocks on A's write lock until A commits.
        monkeypatch.setattr(a, "_cost_record_hook", None)
        t = threading.Thread(target=flush_b)
        started.append(t)
        t.start()
        b_returned.wait(0.8)  # bounded; B returning here would be the split-transaction defect

    monkeypatch.setattr(a, "_cost_record_hook", hook)
    try:
        assert cost_series.flush(a) == 2
        started[0].join(60)
        assert not b_error, b_error
        _, carried = a.cost_series_state(series.name, "test-machine", "m", now=clock.now, window_seconds=W)
        n = a._conn.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0]
        assert n == 3, "both samples stored"
        assert carried is not None
        _assert_fit(carried.fit, first + second)
    finally:
        a.close()


def test_cs6b_a_real_second_process_and_this_one_both_flush(tmp_path: Path, series: CostSeries) -> None:
    path = tmp_path / "memories.db"
    mine = MemoryStore(path, integrity_check=False)
    code = (
        "import sys\n"
        "from brain.memory import cost_series\n"
        "from brain.memory.store import MemoryStore\n"
        "cost_series._machine_override = 'test-machine'\n"
        "store = MemoryStore(sys.argv[1], integrity_check=False)\n"
        "for name in ('test series', 'other series'):\n"
        "    s = cost_series.CostSeries(name)\n"
        "    for i in range(1, 8):\n"
        "        s.record(store, 'm', 1000 * i, 0.1 * i)\n"
        "print(cost_series.flush(store))\n"
        "store.close()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, text=True
    )
    try:
        other = CostSeries("other series")
        for s in (series, other):
            for i in range(1, 5):
                s.record(mine, "m", 500 * i, 0.05 * i)
        written = cost_series.flush(mine)
        assert written == 8
    finally:
        out, _ = proc.communicate(timeout=120)
    assert out.strip() == "14"
    try:
        for name in ("test series", "other series"):
            n = mine._conn.execute(
                "SELECT COUNT(*) FROM cost_series_samples WHERE series = ?", (name,)
            ).fetchone()[0]
            assert n == 7 + 4, name
        mine_pts = [(500 * i, 0.05 * i) for i in range(1, 5)]
        theirs = [(1000 * i, 0.1 * i) for i in range(1, 8)]
        _assert_fit(series.fit(mine, "m"), mine_pts + theirs)
    finally:
        mine.close()


def test_cs6c_a_record_between_take_out_and_write_is_neither_lost_nor_written_twice(
    store: MemoryStore, series: CostSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(series, store, [(1000, 0.3)])
    injected = threading.Event()

    def hook() -> None:
        monkeypatch.setattr(cost_series, "_pending_hook", None)
        series.record(store, "m", 2000, 0.6)
        injected.set()

    monkeypatch.setattr(cost_series, "_pending_hook", hook)
    assert cost_series.flush(store) == 1
    assert injected.is_set()
    assert cost_series.has_pending(store.db_path), "the injected sample waits for the next flush"
    assert store._conn.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0] == 1
    assert cost_series.flush(store) == 1
    assert store._conn.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0] == 2


def test_cs6c_a_reader_during_a_flush_counts_no_sample_twice(
    store: MemoryStore, series: CostSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(series, store, [(1000, 0.3), (3000, 0.7)])
    seen: list[int] = []

    def hook() -> None:
        monkeypatch.setattr(cost_series, "_pending_hook", None)
        state = series.state(store, "m")  # the samples are in neither place now
        seen.append(state.sums.n if state else -1)

    monkeypatch.setattr(cost_series, "_pending_hook", hook)
    cost_series.flush(store)
    assert seen == [0], "out of the buffer, not yet written: briefly missed, never doubled"
    state = series.state(store, "m")
    assert state is not None and state.sums.n == 2


def test_cs6e_a_flush_completing_between_the_two_reads_counts_each_sample_once(
    store: MemoryStore, series: CostSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The database is read first and the buffer second (spec item 3): a flush
    that completes between the two reads moves the samples out of the buffer
    after the database read missed them, so they count at most once."""
    _record(series, store, [(1000, 0.3), (3000, 0.7)])
    counted: list[int] = []

    def flush_between() -> None:
        monkeypatch.setattr(cost_series, "_between_reads_hook", None)
        cost_series.flush(store)

    monkeypatch.setattr(cost_series, "_between_reads_hook", flush_between)
    state = series.state(store, "m")
    assert state is not None
    counted.append(state.sums.n)
    assert counted[0] <= 2 and counted[0] != 4, "never twice"
    assert counted == [0], "database first, buffer second: the moved samples are missed this once"
    again = series.state(store, "m")
    assert again is not None and again.sums.n == 2


def test_cs6e_the_swapped_read_order_would_count_twice(
    store: MemoryStore, series: CostSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why the order matters: reading the buffer first and the database second,
    the same completed flush makes each sample count in both."""
    _record(series, store, [(1000, 0.3), (3000, 0.7)])
    db = cost_series._db_key(store.db_path)
    in_buffer = series._buffered(db, "test-machine", "m", T0 - W)
    cost_series.flush(store)
    sums, _ = store.cost_series_state(series.name, "test-machine", "m", now=cost_series._wall_clock(), window_seconds=W)
    assert len(in_buffer) + sums.n == 4


def test_s127_window_counts_split_at_the_verdict_time(
    store: MemoryStore, series: CostSeries, clock: Clock
) -> None:
    _record(series, store, [(1000, 0.3), (3000, 0.7)])
    clock.now = T0 + 10
    assert series.window_counts(store, "m", T0 + 5) == (2, 0)
    series.record(store, "m", 2000, 0.5)  # taken at T0 + 10, after the verdict time
    assert series.window_counts(store, "m", T0 + 5) == (2, 1)
    cost_series.flush(store)
    assert series.window_counts(store, "m", T0 + 5) == (2, 1), "moving a sample to the database changes no count"
    clock.now = T0 + W + 1  # the two oldest age out
    assert series.window_counts(store, "m", T0 + 5) == (0, 1)


def test_cs1d_the_period_module_is_gone_and_nothing_imports_it() -> None:
    """S146 supersedes S101: no per-period series, no tick hook."""
    import importlib.util

    name = "cost_" + "period"
    assert importlib.util.find_spec(f"brain.memory.{name}") is None
    repo = Path(__file__).resolve().parents[4]
    for path in sorted((repo / "brain").rglob("*.py")):
        assert name not in path.read_text(encoding="utf-8"), path


def test_cs1_a_flush_prunes_exactly_the_rows_that_have_left_the_window(
    store: MemoryStore, series: CostSeries, clock: Clock
) -> None:
    """The prune boundary is the read boundary: a row with taken_at == now - W is
    out of the window and deleted; one tick later than that stays."""
    series.record(store, "m", 1000, 0.1)  # taken at T0
    clock.now = T0 + 5
    series.record(store, "m", 2000, 0.2)  # taken at T0 + 5
    store.record_cost_samples("test series", "test-machine", "m", [], now=T0 + W, window_seconds=W)
    rows = [r[0] for r in store._conn.execute("SELECT taken_at FROM cost_series_samples")]  # noqa: SLF001
    assert rows == [], "nothing was flushed yet: only the explicit prune ran on an empty table"
    cost_series.flush(store)
    store.record_cost_samples("test series", "test-machine", "m", [], now=T0 + W, window_seconds=W)
    left = sorted(r[0] for r in store._conn.execute("SELECT taken_at FROM cost_series_samples"))  # noqa: SLF001
    assert left == [T0 + 5], "the sample taken at exactly now - W is pruned; the later one stays"


def test_cs3d_the_machine_id_is_created_through_an_atomic_exclusive_link(
    real_machine_id: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    real_link = cost_series.os.link

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        return real_link(src, dst)

    monkeypatch.setattr(cost_series.os, "link", spy)
    ident = cost_series.machine_id()
    assert len(calls) == 1 and calls[0][1].endswith("machine_id") and calls[0][0].endswith(".tmp")
    assert (real_machine_id / "machine_id").read_text() == ident


def test_cs6c_concurrent_records_and_flushes_lose_no_sample(store: MemoryStore, series: CostSeries) -> None:
    """Threads recording while another flushes: after a final flush every sample
    is in the database exactly once (the buffer's take-out and the appends are
    under one lock)."""
    import sys

    threads_n, per_thread = 4, 400
    stop = threading.Event()
    errors: list[BaseException] = []

    def record(offset: int) -> None:
        try:
            for i in range(per_thread):
                series.record(store, "m", offset * per_thread + i + 1, 0.001)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def flusher() -> None:
        # Its own connection: a sqlite3 connection is used by one thread.
        other = MemoryStore(store.db_path, integrity_check=False)
        try:
            while not stop.is_set():
                cost_series.flush(other)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            other.close()

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        workers = [threading.Thread(target=record, args=(k,)) for k in range(threads_n)]
        bg = threading.Thread(target=flusher)
        bg.start()
        for w in workers:
            w.start()
        for w in workers:
            w.join(60)
        stop.set()
        bg.join(60)
    finally:
        sys.setswitchinterval(previous)
    assert not errors, errors
    cost_series.flush(store)
    total = store._conn.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0]  # noqa: SLF001
    assert total == threads_n * per_thread


def test_cs6c_the_take_out_is_atomic_against_a_record_made_while_it_runs(
    store: MemoryStore, series: CostSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record that arrives while the flush is taking its samples out (between
    reading the buffer and emptying it) must wait for the take-out, not be wiped
    with it: the injected thread blocks on the buffer lock in the real code."""
    series.record(store, "m", 1000, 0.1)
    threads: list[threading.Thread] = []

    class Interleaving(list):
        armed = True

        def __iter__(self):
            yield from list.__iter__(self)
            if Interleaving.armed:
                Interleaving.armed = False
                t = threading.Thread(target=lambda: series.record(store, "m", 2000, 0.2))
                t.start()
                t.join(0.3)  # in the real code it is blocked on the lock; unguarded it finishes now
                threads.append(t)

    monkeypatch.setattr(cost_series, "_pending", Interleaving(cost_series._pending))
    assert cost_series.flush(store) == 1
    threads[0].join(10)
    assert cost_series.flush(store) == 1, "the sample made during the take-out was kept for the next flush"
    assert store._conn.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0] == 2  # noqa: SLF001


@pytest.mark.parametrize("content", [b"", b"\xff\xfe not utf-8", b"ABCDEF" * 6, b"short"])
def test_cs3d_a_damaged_machine_id_file_is_replaced_not_given_up_on(
    real_machine_id: Path, content: bytes
) -> None:
    (real_machine_id / "machine_id").write_bytes(content)
    ident = cost_series.machine_id()
    assert len(ident) == 32 and (real_machine_id / "machine_id").read_text() == ident
    cost_series._reset_machine_id()
    assert cost_series.machine_id() == ident, "and the repaired file is what later processes read"


def test_flush_stops_at_the_first_failed_write_and_keeps_every_untried_group(
    store: MemoryStore, series: CostSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = CostSeries("other series")
    _record(series, store, [(1000, 0.1)])
    _record(other, store, [(1000, 0.2)])
    tried: list[str] = []

    def boom(self, name, *a, **k):
        tried.append(name)
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patched:
        patched.setattr(MemoryStore, "record_cost_samples", boom)
        assert cost_series.flush(store) == 0
    assert len(tried) == 1, "one busy timeout at most, not one per series"
    assert cost_series.flush(store) == 2, "both groups were kept for the next flush"


def test_flush_drops_a_group_that_fails_for_a_reason_other_than_a_lock_and_saves_the_rest(
    store: MemoryStore, series: CostSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = CostSeries("other series")
    _record(series, store, [(1000, 0.1)])
    _record(other, store, [(1000, 0.2)])
    real = MemoryStore.record_cost_samples

    def picky(self, name, *a, **k):
        if name == "test series":
            raise ValueError("a deterministic per-group failure")
        return real(self, name, *a, **k)

    with monkeypatch.context() as patched:
        patched.setattr(MemoryStore, "record_cost_samples", picky)
        assert cost_series.flush(store) == 1, "the other series is saved"
    assert not cost_series.has_pending(store.db_path), "and the failing group does not block every later flush"


def test_a_flush_prunes_expired_rows_of_other_hosts_and_keys_of_the_same_series(
    store: MemoryStore, series: CostSeries, clock: Clock
) -> None:
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO cost_series_samples VALUES ('test series', 'old-host', 'retired-model', 5, 0.1, ?)", (T0 - W - 1,)
    )
    store._conn.commit()  # noqa: SLF001
    series.record(store, "m", 1000, 0.1)
    cost_series.flush(store)
    hosts = {r[0] for r in store._conn.execute("SELECT host FROM cost_series_samples")}  # noqa: SLF001
    assert hosts == {"test-machine"}


def _production_series():
    from brain.memory import keyword_budget, reranker, semantic_recall

    return [
        ("rerank", reranker._rerank_cost, "m"),  # noqa: SLF001
        ("paragraph embed", semantic_recall._embed_cost, "m"),  # noqa: SLF001
        ("recall setup", semantic_recall._message_overhead, ""),  # noqa: SLF001
        ("paragraph scan", semantic_recall._scan_time, ""),  # noqa: SLF001
        ("keyword extra words", keyword_budget._keyword_cost, ""),  # noqa: SLF001
        ("name matching", keyword_budget._name_cost, "keyword"),  # noqa: SLF001
    ]


@pytest.mark.parametrize("which", range(6))
@pytest.mark.parametrize("flushed", [True, False])
def test_cs1_each_production_series_counts_a_sample_for_exactly_the_window(
    store: MemoryStore, clock: Clock, which: int, flushed: bool
) -> None:
    """CS1a on EACH of the six series' own objects (not the generic class): the
    estimate at t0 + W - 1 is the independent least-squares fit, at exactly t0 + W
    the samples no longer count."""
    name, series, key = _production_series()[which]
    assert series.name == name
    points = [(1000, 0.3), (3000, 0.7), (2000, 0.5001)]
    for x, y in points:
        series.record(store, key, x, y)
    if flushed:
        cost_series.flush(store)
    clock.now = T0 + W - 1
    _assert_fit(series.fit(store, key), points)
    clock.now = T0 + W
    state = series.state(store, key)
    assert state is not None and state.sums.n == 0


@pytest.mark.parametrize("which", range(6))
def test_cs2_each_production_series_survives_a_restart(tmp_path: Path, clock: Clock, which: int) -> None:
    _name, series, key = _production_series()[which]
    path = tmp_path / "memories.db"
    first = MemoryStore(path, integrity_check=False)
    for x, y in [(1000, 0.3), (3000, 0.7)]:
        series.record(first, key, x, y)
    cost_series.flush(first)
    before = series.estimate(first, key)
    first.close()
    cost_series._reset_pending()
    second = MemoryStore(path, integrity_check=False)
    try:
        assert before is not None and series.estimate(second, key) == before
    finally:
        second.close()


def test_cs6e_window_counts_also_reads_the_database_before_the_buffer(
    store: MemoryStore, series: CostSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(series, store, [(1000, 0.3), (3000, 0.7)])

    def flush_between() -> None:
        monkeypatch.setattr(cost_series, "_between_reads_hook", None)
        cost_series.flush(store)

    monkeypatch.setattr(cost_series, "_between_reads_hook", flush_between)
    before, after = series.window_counts(store, "m", T0 - 1)
    assert before + after <= 2, "a flush between the two reads never makes a sample count twice"
    assert sum(series.window_counts(store, "m", T0 - 1)) == 2


# ---------------------------------------------------------------------------
# Follow-up (S171, S176; R2-5): the ratio estimate, the bounded buffer, CS1b per series.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", range(6))
@pytest.mark.parametrize("flushed", [True, False])
def test_cs1b_each_production_series_counts_only_the_later_samples_when_the_earlier_age_out(
    store: MemoryStore, clock: Clock, which: int, flushed: bool
) -> None:
    """Parent CS1b on EACH series' own object: samples at t0 and at t0 + W/2; at t0 + W only the
    later ones count (independent least-squares fit of them alone)."""
    _name, series, key = _production_series()[which]
    early = [(1000, 0.3), (3000, 0.7)]
    later = [(2000, 0.5), (4000, 0.9)]
    for x, y in early:
        series.record(store, key, x, y)
    clock.now = T0 + W / 2
    for x, y in later:
        series.record(store, key, x, y)
    if flushed:
        cost_series.flush(store)
    clock.now = T0 + W
    _assert_fit(series.fit(store, key), later)


def test_the_ratio_estimate_is_the_weighted_ratio_of_the_sums_whatever_the_sizes() -> None:
    """S171: a through-origin series answers sum(y) / sum(x), no overhead, even with two sizes
    (an ordinary fit of these points would have an overhead and a different slope)."""
    state = _state([(1000, 3_000_000.0), (3000, 5_000_000.0)])
    ratio = state.estimate(True)
    assert ratio is not None and ratio.fit.overhead == 0.0 and ratio.max_x == 3000
    assert ratio.fit.rate == pytest.approx(8_000_000.0 / 4_000)
    ordinary = state.estimate()  # least squares: 2,000,000 B overhead and 1,000 B a token
    assert ordinary is not None and ordinary.fit.rate == pytest.approx(1_000.0)
    # S178: an empty window answers nothing, whatever row is carried (an ordinary series
    # still answers its carried fit: S146, CS3b)
    carried = RangedFit(LinearFit(0.0, 1234.5, 3), 2500)
    assert SeriesState(LinearSums(), carried).estimate(True) is None
    assert SeriesState(LinearSums(), RangedFit(LinearFit(0.0, 9.9e12, 7), 99)).estimate(True) is None
    assert SeriesState(LinearSums(), carried).estimate() == carried


def test_a_ratio_window_whose_sizes_sum_to_zero_has_no_figure() -> None:
    """S178: samples of size 0 give no ratio (and no division), carried row or not."""
    zero = LinearSums().plus(0, 5.0)
    carried = RangedFit(LinearFit(0.0, 1234.5, 3), 2500)
    assert SeriesState(zero, carried).estimate(True) is None
    assert SeriesState(zero, None).estimate(True) is None


def test_a_ratio_window_with_one_sample_answers_that_samples_ratio() -> None:
    """S178 / A1b: the figure ages out sample by sample, not all at once."""
    one = LinearSums().plus(2000, 5_000_000.0)
    carried = RangedFit(LinearFit(0.0, 1.0e12, 9), 9000)
    got = SeriesState(one, carried).estimate(True)
    assert got is not None and got.fit.rate == pytest.approx(2500.0) and got.max_x == 2000


def test_the_ram_series_saves_its_ratio_as_the_carried_row_but_never_answers_from_it(store: MemoryStore, clock: Clock) -> None:
    ram = CostSeries(cost_series.RAM_SERIES)
    ram.record(store, "m", 1000, 1_000_000.0)
    ram.record(store, "m", 3000, 5_000_000.0)
    cost_series.flush(store)
    row = store._conn.execute(  # noqa: SLF001
        "SELECT overhead, rate FROM cost_series_carried WHERE series = ?", (cost_series.RAM_SERIES,)
    ).fetchone()
    assert row[0] == 0.0 and row[1] == pytest.approx(6_000_000.0 / 4000)
    clock.now = T0 + W + 1  # every sample aged out: the ratio series has no figure (S178), the row is not read
    assert ram.fit(store, "m") is None
    other = CostSeries("a time series")
    other.record(store, "m", 1000, 0.1)
    other.record(store, "m", 2000, 0.2)
    cost_series.flush(store)
    clock.now = T0 + 2 * W + 2  # an ordinary series still answers its carried fit after downtime (S146, CS3b)
    assert other.fit(store, "m") is not None


def _buffered() -> list[cost_series._Sample]:  # noqa: SLF001
    with cost_series._pending_lock:  # noqa: SLF001
        return list(cost_series._pending)  # noqa: SLF001


def test_cs17f_appending_a_sample_drops_every_buffered_sample_past_the_window(
    store: MemoryStore, series: CostSeries, clock: Clock, tmp_path: Path
) -> None:
    other_store = MemoryStore(tmp_path / "other.db", integrity_check=False)
    try:
        other = CostSeries("another series")
        series.record(store, "m", 1000, 0.1)  # at T0
        other.record(other_store, "k", 1000, 0.1)  # at T0, another database and series
        clock.now = T0 + 5
        series.record(store, "m", 2000, 0.2)  # at T0 + 5
        assert len(_buffered()) == 3, "nothing is past the window yet"
        clock.now = T0 + W  # the T0 samples are exactly one window old: gone; T0 + 5 stays
        series.record(store, "m", 3000, 0.3)
        assert sorted(s.x for s in _buffered()) == [2000, 3000]
        clock.now = T0 + W + 5 + 1  # now the T0 + 5 sample is past the window too
        other.record(other_store, "k", 4000, 0.4)
        assert sorted(s.x for s in _buffered()) == [3000, 4000]
    finally:
        other_store.close()


def test_cs17f_a_flush_that_keeps_failing_holds_at_most_one_window_and_loses_none_in_it(
    store: MemoryStore, series: CostSeries, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(`flush` itself already drops expired samples before it writes, so this scenario is bounded
    with or without the append-time prune; the prune is what bounds a buffer that is never flushed,
    pinned by the test above. This one pins that no in-window sample is lost on the way.)"""
    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patched:
        patched.setattr(store, "record_cost_samples", locked)
        for step in range(30):
            clock.now = T0 + step * (W / 10)  # three days of samples, ten per window
            series.record(store, "m", 1000 + step, 0.1)
            assert cost_series.flush(store) == 0
        held = [s for s in _buffered() if s.db == cost_series._store_key(store)]  # noqa: SLF001
        assert len(held) == 10 and all(s.taken_at > clock.now - W for s in held)
    assert cost_series.flush(store) == 10, "a flush that finally succeeds writes the in-window ones once"
