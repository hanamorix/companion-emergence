"""#660: a clock set forward and then corrected. The rewind brings the open day and `saved_at` back to the real clock.

Design: `docs/source-spec/2026-11-27-660-clock-forward-unfreeze-design.md`. Times are local noon of a civil day (`local_noon`),
so no test depends on the host's time zone.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import pytest

from brain.emotion import engine_files, engine_rebuild
from brain.emotion.engine_utils import (
    Average,
    EmotionState,
    EngineState,
    SnapshotDay,
    rewind_due,
    rewind_future_day,
)
from tests.unit.brain.emotion._engine_support import (
    install,
    local_day,
    local_noon,
    make_persona,
    row,
    save,
    showing,
    simple_state,
    state_of,
)


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    return make_persona(tmp_path / "persona")


def _day(n: int) -> str:
    return local_day(n).isoformat()


def _snap(n: int) -> SnapshotDay:
    return SnapshotDay(_day(n), 4.0, 4.0, True, None)


def _state(active: int, snapshot_days: list[int], last_entry: int | None) -> EngineState:
    return EngineState(
        active_day=_day(active),
        emotions={
            "joy": EmotionState(
                short=Average(4.0, 1.0),
                long=Average(4.0, 1.0),
                last_entry_day=None if last_entry is None else _day(last_entry),
                snapshot=[_snap(n) for n in snapshot_days],
            )
        },
    )


# --- the pure rewind ------------------------------------------------------------------------------------------------


def test_a_corrected_clock_drops_the_future_days_and_moves_the_open_day_back() -> None:
    # Real days 1-10 closed; the clock jumped to day 20 (closing day 10 early), ran past two wrong midnights (days 20, 21
    # closed), then was corrected to day 11.
    state = _state(active=22, snapshot_days=[8, 9, 10, 20, 21], last_entry=22)
    assert rewind_future_day(state, local_noon(local_day(11))) is True
    joy = state.emotions["joy"]
    assert [d.day for d in joy.snapshot] == [_day(8), _day(9), _day(10)]
    assert state.active_day == _day(11)
    assert joy.last_entry_day == _day(11)


def test_a_same_day_correction_leaves_the_open_day_at_tomorrow_and_counts_as_no_change() -> None:
    # The jump closed day 10 early and set the open day to day 20; corrected on day 10 itself. Day 10 has a snapshot, so the
    # open day can come back only to day 11, and once it is there a second pass that day changes nothing.
    state = _state(active=20, snapshot_days=[9, 10], last_entry=20)
    clock = local_noon(local_day(10))
    assert rewind_future_day(state, clock) is True
    assert state.active_day == _day(11) and state.emotions["joy"].last_entry_day == _day(11)
    assert rewind_due(state, clock) is False
    assert rewind_future_day(state, clock) is False
    assert state.active_day == _day(11) and [d.day for d in state.emotions["joy"].snapshot] == [_day(9), _day(10)]


def test_with_no_snapshot_left_the_open_day_comes_back_to_the_clocks_day() -> None:
    state = _state(active=30, snapshot_days=[25], last_entry=30)  # every snapshot day is a wrong one
    assert rewind_future_day(state, local_noon(local_day(12))) is True
    assert state.active_day == _day(12) and state.emotions["joy"].snapshot == []
    assert state.emotions["joy"].last_entry_day == _day(12)


def test_a_clock_on_or_after_the_open_day_is_no_rewind() -> None:
    for clock_day in (10, 11):
        state = _state(active=10, snapshot_days=[8, 9], last_entry=10)
        assert rewind_due(state, local_noon(local_day(clock_day))) is False
        assert rewind_future_day(state, local_noon(local_day(clock_day))) is False
        assert state.active_day == _day(10) and len(state.emotions["joy"].snapshot) == 2
    assert rewind_future_day(EngineState(), local_noon(local_day(3))) is False  # no open day yet


def test_an_earlier_last_entry_day_is_kept() -> None:
    state = _state(active=20, snapshot_days=[9], last_entry=8)
    assert rewind_future_day(state, local_noon(local_day(10))) is True
    assert state.active_day == _day(10) and state.emotions["joy"].last_entry_day == _day(8)


# --- through the drain ----------------------------------------------------------------------------------------------


def _forward_state(*, real: int, wrong_until: int) -> EngineState:
    """A state a clock set forward left behind: real days up to `real` closed, the clock jumped to day 20 (closing `real`
    early) and kept running, closing days 20 to `wrong_until - 1`; saved and open on day `wrong_until`."""
    saved_at = local_noon(local_day(wrong_until))
    state = simple_state(saved_at=saved_at, day=_day(wrong_until))
    state.emotions["joy"].snapshot = [_snap(n) for n in (*range(real - 2, real + 1), *range(20, wrong_until))]
    return state


def test_the_drain_rewinds_a_corrected_clock_and_day_closes_resume(persona: Path) -> None:
    save(persona, _forward_state(real=10, wrong_until=22))
    clock = local_noon(local_day(11))
    install(persona, now=clock).note_active(clock)
    state = state_of(persona)  # it reloads: the rewound state passes validation
    assert state.active_day == _day(11)
    assert state.saved_at == clock
    assert [d.day for d in state.emotions["joy"].snapshot] == [_day(8), _day(9), _day(10)]

    later = local_noon(local_day(12))
    install(persona, now=later).note_active(later)
    assert [d.day for d in state_of(persona).emotions["joy"].snapshot] == [_day(8), _day(9), _day(10), _day(11)]


def test_the_quick_skip_does_not_hide_a_due_rewind(persona: Path) -> None:
    save(persona, _forward_state(real=10, wrong_until=22))
    wrong = local_noon(local_day(22))
    engine = install(persona, now=wrong)
    engine.note_active(wrong)  # a clean drain at the wrong clock: nothing to do, the queue key is remembered
    assert engine._can_skip(wrong)  # the control: an unchanged idle persona is skipped
    clock = local_noon(local_day(11))
    engine._now = lambda: clock  # the clock is corrected
    engine.note_active(clock)
    assert state_of(persona).active_day == _day(11)


def test_a_rewinding_drain_deletes_no_change_log_file(persona: Path) -> None:
    # The wrong clock ran to day 50: a prune anchored on day 50 would delete every file before day 20, all of these.
    save(persona, _forward_state(real=10, wrong_until=50))
    folder = engine_files.changes_dir(persona)
    folder.mkdir(parents=True)
    names = [f"{_day(n)}.jsonl" for n in (2, 6, 10)]
    for name in names:
        (folder / name).write_text("", encoding="utf-8")
    clock = local_noon(local_day(11))
    install(persona, now=clock).note_active(clock)
    assert state_of(persona).active_day == _day(11)
    assert sorted(p.name for p in folder.iterdir()) == names


def _local_midnight(n: int) -> datetime:
    return datetime.combine(local_day(n), time.min).astimezone(UTC)


@pytest.mark.no_base_failure(
    reason="#660 pins the lock-wait race the rewind must leave alone; the base code has no rewind, so it passes there"
)
@pytest.mark.parametrize("present", [True, False], ids=["a present emotion", "no present emotion"])
def test_a_drain_that_read_its_clock_before_another_pass_closed_the_day_does_not_rewind(persona: Path, present: bool) -> None:
    # Another pass closed day 10 one second after midnight: open day 11, saved at 00:00:01. This drain read its clock at
    # 23:59:59 on day 10, before waiting for the lock; under the lock the clock reads 00:00:02.
    saved = _local_midnight(11) + timedelta(seconds=1)
    state = simple_state(saved_at=saved, day=_day(11))
    if present:
        state.emotions["joy"].snapshot = [_snap(10)]
    else:
        state.emotions = {}  # no present emotion: the close appended no snapshot
    save(persona, state)
    install(persona, now=saved + timedelta(seconds=1)).note_active(_local_midnight(11) - timedelta(seconds=1))
    after = state_of(persona)
    assert after.active_day == _day(11)
    assert after.saved_at >= saved


def test_the_direct_apply_rewinds_too(persona: Path) -> None:
    save(persona, _forward_state(real=10, wrong_until=22))
    clock = local_noon(local_day(11))
    install(persona, now=clock)._apply_direct([showing("m1", clock, {"joy": 5.0})])
    state = state_of(persona)
    assert state.active_day == _day(11) and state.saved_at == clock
    assert state.emotions["joy"].last_entry_day == _day(11)


@pytest.mark.no_base_failure(
    reason="#660 pins that a rebuild after a corrected clock needs no rewind: it starts from a fresh state, so it passes on base"
)
def test_a_recovery_rebuild_after_a_corrected_clock_has_no_future_day(tmp_path: Path) -> None:
    persona = make_persona(tmp_path / "persona", [row(local_noon(local_day(9)), emotions={"joy": 6.0}, mid="a")])
    forward = _forward_state(real=10, wrong_until=22)
    forward.switch_over_at = local_noon(local_day(0))  # a recovery carries the switch-over time from the state it replaces
    save(persona, forward)
    clock = local_noon(local_day(11))
    engine_rebuild.rebuild(persona, "recovery", clock, faded_emotions_of=engine_rebuild._stored_emotions, modifier_of=lambda _m: 1.0)
    state = state_of(persona)
    assert state.active_day == _day(11)
    assert all(d.day <= _day(11) for e in state.emotions.values() for d in e.snapshot)
