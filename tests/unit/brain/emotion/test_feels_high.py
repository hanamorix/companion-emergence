"""`engine.feels_high(name)`: the one predicate every reader that used to fire on a fixed level calls (S52, S53).

A feeling is high when its shift reads stronger (a rising run), or when its short average today is at or above
`lowest + FEELS_HIGH_RANGE_FRACTION * (highest - lowest)` of its own daily short averages in the 30-day snapshot. With
fewer than two snapshot days, or a flat range, only the shift part applies.

The rule is `engine_utils.feels_high_from`; `EmotionEngine.feels_high` gathers its inputs. `read()` is the stub in this
branch (Engine 4 builds the real one), so the engine tests patch it with the readings they need and keep the snapshot in
a real state file read through the engine's own state cache.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files
from brain.emotion.engine import EmotionEngine, Reading
from brain.emotion.engine_files import LoadedState
from brain.emotion.engine_utils import (
    Average,
    EmotionState,
    EngineState,
    SnapshotDay,
    feels_high_from,
)

# Snapshot shorts 2.0 and 6.0: with the fraction 0.5 the threshold is exactly 4.0 (all exact in binary).
RANGE = (2.0, 4.0, 6.0, 3.0)


def _days(*shorts: float) -> list[SnapshotDay]:
    return [SnapshotDay(f"2026-10-{i + 1:02d}", s, 0.0, True, None) for i, s in enumerate(shorts)]


# --- the rule itself ----------------------------------------------------------------------------------------------


def test_shift_stronger_alone_is_high_even_when_the_range_says_low():
    assert feels_high_from(1.0, +1, _days(*RANGE)) is True


def test_a_falling_run_alone_is_not_high():
    assert feels_high_from(1.0, -1, _days(*RANGE)) is False


def test_the_range_part_is_high_in_the_upper_half_and_not_below_it():
    assert feels_high_from(5.0, 0, _days(*RANGE)) is True
    assert feels_high_from(3.5, 0, _days(*RANGE)) is False


def test_the_range_part_counts_a_value_above_the_highest_day():
    assert feels_high_from(9.0, 0, _days(*RANGE)) is True


def test_at_the_threshold_counts_as_high_and_just_under_does_not():
    assert feels_high_from(4.0, 0, _days(*RANGE)) is True
    assert feels_high_from(3.999, 0, _days(*RANGE)) is False


def test_the_fraction_is_read_from_the_dev_constants_at_call_time(monkeypatch):
    days = _days(*RANGE)
    assert feels_high_from(3.0, 0, days) is False
    monkeypatch.setattr(dev_constants, "FEELS_HIGH_RANGE_FRACTION", 0.25)  # threshold 3.0
    assert feels_high_from(3.0, 0, days) is True
    assert feels_high_from(2.9, 0, days) is False


@pytest.mark.parametrize("snapshot", [[], _days(6.0)], ids=["no days", "one day"])
def test_fewer_than_two_snapshot_days_leaves_only_the_shift_part(snapshot):
    assert feels_high_from(9.0, 0, snapshot) is False
    assert feels_high_from(0.0, +1, snapshot) is True


def test_a_flat_range_leaves_only_the_shift_part():
    flat = _days(4.0, 4.0, 4.0)
    assert feels_high_from(9.0, 0, flat) is False
    assert feels_high_from(4.0, 0, flat) is False
    assert feels_high_from(0.0, +1, flat) is True


def test_the_rule_is_read_from_the_daily_short_averages_not_the_long_ones():
    days = [
        SnapshotDay(f"2026-10-{i + 1:02d}", s, 9.0 - s, True, None) for i, s in enumerate(RANGE)
    ]
    assert feels_high_from(5.0, 0, days) is True
    assert feels_high_from(3.5, 0, days) is False


# --- the engine method --------------------------------------------------------------------------------------------


def _state(snapshots: dict[str, list[SnapshotDay]]) -> EngineState:
    state = EngineState(saved_at=datetime(2026, 11, 4, 12, 0, tzinfo=UTC))
    for name, days in snapshots.items():
        state.emotions[name] = EmotionState(
            short=Average(5.0, 1.0), long=Average(3.0, 1.0), snapshot=days
        )
    return state


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An engine over a real state file; `read()` (the stub in this branch) answers from `engine.readings`."""
    eng = EmotionEngine(tmp_path)
    eng.readings = {}  # type: ignore[attr-defined]
    monkeypatch.setattr(EmotionEngine, "read", lambda self: dict(self.readings))  # type: ignore[attr-defined]
    return eng


def _write(tmp_path: Path, snapshots: dict[str, list[SnapshotDay]]) -> None:
    engine_files.save_state(tmp_path, _state(snapshots))


def test_engine_shift_stronger_is_high(engine, tmp_path):
    _write(tmp_path, {"love": _days(*RANGE)})
    engine.readings = {"love": Reading(1.0, 0.5, 0.5, +1)}
    assert engine.feels_high("love") is True


def test_engine_a_falling_run_alone_is_not_high(engine, tmp_path):
    _write(tmp_path, {"love": _days(*RANGE)})
    engine.readings = {"love": Reading(1.0, 3.0, -2.0, -1)}
    assert engine.feels_high("love") is False


def test_engine_range_part_reads_the_snapshot_of_that_feeling(engine, tmp_path):
    _write(tmp_path, {"love": _days(*RANGE), "grief": _days(0.0, 1.0, 2.0)})
    engine.readings = {"love": Reading(5.0, 3.0, 2.0, 0), "grief": Reading(5.0, 3.0, 2.0, 0)}
    assert engine.feels_high("love") is True  # 5.0 is in the upper half of 2.0..6.0
    engine.readings = {"love": Reading(3.5, 3.0, 0.5, 0), "grief": Reading(1.9, 3.0, -1.1, 0)}
    assert engine.feels_high("love") is False
    assert (
        engine.feels_high("grief") is True
    )  # 1.9 is in the upper half of 0.0..2.0, and love's range is not used


def test_engine_boundary_at_the_threshold_is_high(engine, tmp_path):
    _write(tmp_path, {"love": _days(*RANGE)})
    engine.readings = {"love": Reading(4.0, 3.0, 1.0, 0)}
    assert engine.feels_high("love") is True
    engine.readings = {"love": Reading(3.999, 3.0, 0.9, 0)}
    assert engine.feels_high("love") is False


@pytest.mark.parametrize(
    "snapshot", [[], _days(6.0), _days(4.0, 4.0)], ids=["none", "one day", "flat"]
)
def test_engine_too_few_days_or_flat_range_is_shift_only(engine, tmp_path, snapshot):
    _write(tmp_path, {"love": snapshot})
    engine.readings = {"love": Reading(9.0, 3.0, 6.0, 0)}
    assert engine.feels_high("love") is False
    engine.readings = {"love": Reading(0.0, 3.0, -3.0, +1)}
    assert engine.feels_high("love") is True


def test_a_name_read_does_not_return_is_not_high(engine, tmp_path):
    _write(tmp_path, {"love": _days(*RANGE)})
    engine.readings = {"grief": Reading(9.0, 3.0, 6.0, +1)}
    assert engine.feels_high("love") is False


def test_an_empty_read_is_not_high(engine, tmp_path):
    _write(tmp_path, {"love": _days(*RANGE)})
    engine.readings = {}
    assert engine.feels_high("love") is False


def test_the_stub_read_of_this_branch_makes_every_name_not_high(tmp_path):
    _write(tmp_path, {"love": _days(*RANGE)})
    assert EmotionEngine(tmp_path).feels_high("love") is False  # read() is {} until Engine 4


def test_nothing_is_written_by_asking(engine, tmp_path):
    _write(tmp_path, {"love": _days(*RANGE)})
    engine.readings = {"love": Reading(5.0, 3.0, 2.0, 0)}
    before = engine_files.state_path(tmp_path).read_bytes()
    engine.feels_high("love")
    assert engine_files.state_path(tmp_path).read_bytes() == before


# --- failure paths: the predicate never raises --------------------------------------------------------------------


def test_a_state_cache_that_raises_gives_the_shift_answer_only(engine, monkeypatch, caplog):
    def boom():
        raise OSError("disk went away")

    monkeypatch.setattr(engine._state_cache, "get", boom)
    engine.readings = {"love": Reading(9.0, 3.0, 6.0, 0)}
    with caplog.at_level(logging.WARNING, logger="brain.emotion.engine"):
        assert engine.feels_high("love") is False
    engine.readings = {"love": Reading(0.0, 3.0, -3.0, +1)}
    assert engine.feels_high("love") is True
    assert any("feels_high" in r.getMessage() for r in caplog.records)


def test_a_state_that_is_unreadable_between_the_read_and_the_snapshot_is_shift_only(
    engine, monkeypatch
):
    monkeypatch.setattr(engine._state_cache, "get", lambda: LoadedState(None, None, None, "gone"))
    engine.readings = {"love": Reading(9.0, 3.0, 6.0, 0)}
    assert engine.feels_high("love") is False
    engine.readings = {"love": Reading(0.0, 3.0, -3.0, +1)}
    assert engine.feels_high("love") is True


def test_a_name_in_the_read_but_not_in_the_state_is_shift_only(engine, tmp_path):
    _write(tmp_path, {"grief": _days(*RANGE)})
    engine.readings = {"love": Reading(9.0, 3.0, 6.0, 0)}
    assert engine.feels_high("love") is False
    engine.readings = {"love": Reading(0.0, 3.0, -3.0, +1)}
    assert engine.feels_high("love") is True
