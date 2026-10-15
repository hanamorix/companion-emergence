"""S220: the reranker's memory growth by call shape, as a rolling 24-hour series saved with the
others (S146 window, S159 host key, S164 buffer). Real SQLite files; scripted wall clock."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import dev_constants
from brain.memory import cost_series, ram_fit
from brain.memory.cost_series import RAM_SHAPE_SERIES, CostSeries
from brain.memory.store import MemoryStore

W = dev_constants.COST_SERIES_WINDOW_S
T0 = 1_000_000.0
MODEL = "reranker-under-test"


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
def shapes() -> CostSeries:
    return CostSeries(RAM_SHAPE_SERIES)


def _x(pairs: int, length: int, *, full: bool = False) -> int:
    x = ram_fit.encode_shape(pairs, length, full=full)
    assert x is not None
    return x


def _by_shape(groups):
    return {ram_fit.decode_shape(x)[:2]: (n, total) for x, n, total in groups}


def test_the_window_comes_back_as_one_group_per_packed_shape(store, shapes, clock) -> None:
    shapes.record(store, MODEL, _x(1, 1024, full=True), 190.0)
    shapes.record(store, MODEL, _x(1, 1024, full=True), 192.0)
    shapes.record(store, MODEL, _x(2, 512), 68.0)
    cost_series.flush(store)
    shapes.record(store, MODEL, _x(2, 512), 70.0)  # still buffered
    groups = shapes.shape_groups(store, MODEL)
    assert groups is not None
    assert _by_shape(groups) == {(1, 1024): (2, 382.0), (2, 512): (2, 138.0)}


def test_a_hundred_thousand_samples_of_one_shape_are_one_group(store, shapes, clock) -> None:
    x = _x(7, 300)
    shapes.record_many(store, MODEL, [(x, 5.0)] * 100_000)
    cost_series.flush(store)
    groups = shapes.shape_groups(store, MODEL)
    assert groups == [(x, 100_000, 500_000.0)]


def test_a_sample_counts_for_exactly_the_window(store, shapes, clock) -> None:
    shapes.record(store, MODEL, _x(1, 1024, full=True), 190.0)
    cost_series.flush(store)
    clock.now = T0 + W - 1
    assert len(shapes.shape_groups(store, MODEL)) == 1
    clock.now = T0 + W
    assert shapes.shape_groups(store, MODEL) == []


def test_another_machines_rows_are_not_this_machines_window(
    store, shapes, clock, monkeypatch
) -> None:
    monkeypatch.setattr(cost_series, "_machine_override", "a" * 32)
    shapes.record(store, MODEL, _x(1, 1024, full=True), 190.0)
    cost_series.flush(store)
    monkeypatch.setattr(cost_series, "_machine_override", "b" * 32)
    assert shapes.shape_groups(store, MODEL) == []
    shapes.record(store, MODEL, _x(3, 100), 4.0)
    assert [ram_fit.decode_shape(x)[:2] for x, _n, _t in shapes.shape_groups(store, MODEL)] == [
        (3, 100)
    ]


def test_another_models_rows_are_not_this_models_window(store, shapes, clock) -> None:
    shapes.record(store, "other-model", _x(1, 1024, full=True), 190.0)
    cost_series.flush(store)
    assert shapes.shape_groups(store, MODEL) == []


def test_recording_writes_nothing_to_the_database(store, shapes, clock) -> None:
    shapes.record(store, MODEL, _x(1, 1024, full=True), 190.0)
    rows = store._conn.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0]  # noqa: SLF001
    assert rows == 0, "S164: the reply path only buffers"
    assert cost_series.has_pending(store.db_path)


def test_a_flush_saves_the_samples_and_writes_no_carried_row(store, shapes, clock) -> None:
    for pairs, length, grown in [(1, 1024, 190.0), (1, 512, 8.0), (2, 512, 68.0)]:
        shapes.record(store, MODEL, _x(pairs, length), grown)
    assert cost_series.flush(store) == 3
    conn = store._conn  # noqa: SLF001
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM cost_series_samples WHERE series = ?", (RAM_SHAPE_SERIES,)
        ).fetchone()[0]
        == 3
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM cost_series_carried WHERE series = ?", (RAM_SHAPE_SERIES,)
        ).fetchone()[0]
        == 0
    )


def test_the_shape_series_survives_a_restart(tmp_path, shapes, clock) -> None:
    path = tmp_path / "memories.db"
    first = MemoryStore(path, integrity_check=False)
    shapes.record(first, MODEL, _x(1, 1024, full=True), 190.0)
    shapes.record(first, MODEL, _x(1, 512), 8.0)
    cost_series.flush(first)
    before = shapes.shape_groups(first, MODEL)
    first.close()
    cost_series._reset_pending()  # noqa: SLF001
    second = MemoryStore(path, integrity_check=False)
    try:
        assert shapes.shape_groups(second, MODEL) == before and len(before) == 2
    finally:
        second.close()


def test_the_retired_per_token_series_is_deleted_by_the_first_shape_flush(
    store, shapes, clock
) -> None:
    conn = store._conn  # noqa: SLF001
    old = cost_series.RETIRED_SERIES[0]
    conn.execute(
        "INSERT INTO cost_series_samples VALUES (?, 'h', ?, 1000, 5e6, ?)", (old, MODEL, T0)
    )
    conn.execute(
        "INSERT INTO cost_series_carried VALUES (?, 'h', ?, 0.0, 5000.0, 1, 1000, ?)",
        (old, MODEL, T0),
    )
    conn.commit()
    other = CostSeries("a time series")
    other.record(store, MODEL, 1000, 0.1)
    cost_series.flush(store)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM cost_series_samples WHERE series = ?", (old,)
        ).fetchone()[0]
        == 1
    ), "an ordinary flush leaves it"
    shapes.record(store, MODEL, _x(1, 1024, full=True), 190.0)
    cost_series.flush(store)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM cost_series_samples WHERE series = ?", (old,)
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM cost_series_carried WHERE series = ?", (old,)
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM cost_series_samples WHERE series = 'a time series'"
        ).fetchone()[0]
        == 1
    )


def test_the_flush_of_many_large_packed_shapes_does_not_overflow_sqlite(
    store, shapes, clock
) -> None:
    """Sums of x squared over packed shapes overflow SQLite integers at ~170,000 rows; the
    shape series never sums them."""
    x = _x(58, 1000)
    shapes.record_many(store, MODEL, [(x, 1.0)] * 200_000)
    assert cost_series.flush(store) == 200_000
    assert shapes.shape_groups(store, MODEL) == [(x, 200_000, 200_000.0)]


def test_a_failed_database_read_is_no_fit(store, shapes, clock, monkeypatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("database is gone")

    monkeypatch.setattr(store, "cost_series_shape_groups", _boom)
    assert shapes.shape_groups(store, MODEL) is None


def test_a_flush_between_the_two_reads_never_counts_a_sample_twice(
    store, shapes, clock, monkeypatch
) -> None:
    shapes.record(store, MODEL, _x(1, 1024, full=True), 190.0)
    monkeypatch.setattr(cost_series, "_between_reads_hook", lambda: cost_series.flush(store))
    groups = shapes.shape_groups(store, MODEL)
    assert sum(n for _x_, n, _t in groups) <= 1
    monkeypatch.setattr(cost_series, "_between_reads_hook", None)
    assert sum(n for _x_, n, _t in shapes.shape_groups(store, MODEL)) == 1
