"""The drain (`note_active`) of the emotion engine (Engine 4, #579): position consumption, the day close, the early-out, failures.

Direct apply, recovery first and batch are in `test_engine_direct_apply.py`. Expected numbers are hand-worked in the
comment beside them. Times are local noon of a civil day (`local_noon`); the engine of each test is installed with a fixed clock.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files
from brain.emotion.engine_utils import EngineState, RepeatEntry
from tests.unit.brain.emotion._engine_support import (
    LockSpy,
    change,
    engine_errors,
    install,
    local_day,
    local_noon,
    make_persona,
    queue,
    queue_events,
    queue_text_lines,
    ruin_the_state,
    save,
    showing,
    simple_state,
    state_of,
    tree,
)

D = [local_noon(local_day(n)) for n in range(60)]  # D[n] = local noon of day n


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    return make_persona(tmp_path / "persona")


def _day(n: int) -> str:
    return local_day(n).isoformat()


def _errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def _count_saves(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Wrap `save_state`: each call records (the position it saves, the queue file's size at that moment)."""
    saves: list[tuple[int, int]] = []
    real = engine_files.save_state

    def spy(persona_dir: Path, state) -> None:  # noqa: ANN001
        path = engine_files.queue_path(persona_dir)
        saves.append((state.queue_line, path.stat().st_size if path.exists() else 0))
        real(persona_dir, state)

    monkeypatch.setattr(engine_files, "save_state", spy)
    return saves


def _fail_emptying(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make `empty_queue` fail (its `write_jsonl` raises on every try); returns the list of attempts."""
    attempts: list[int] = []

    def refuse(*args: object, **kwargs: object) -> None:
        attempts.append(1)
        raise PermissionError("the queue file is held")

    monkeypatch.setattr(engine_files, "write_jsonl", refuse)
    return attempts


# --- the early-out (D1, D2) -------------------------------------------------------------------------------


def test_a_quiet_pass_takes_no_lock_and_writes_nothing_and_a_new_line_is_then_taken_once(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = install(persona)
    queue(persona, showing("m1", D[3], {"joy": 6.0}))
    engine.note_active(D[3])  # a draining pass: folds, saves, empties, and remembers the emptied file's key
    before = tree(persona)
    spy = LockSpy(monkeypatch)
    engine.note_active(D[3])
    engine.note_active(D[3])  # the second quiet call as well: the key was taken AFTER the emptying
    assert (spy.calls, spy.within, spy.acquisitions) == (0, [], [])
    assert tree(persona) == before
    # positive control: a stub that returns at once fails here
    queue(persona, showing("m2", D[3], {"joy": 2.0}))
    engine.note_active(D[3])
    assert len(spy.within) == 1 and spy.blocking == []
    assert state_of(persona).emotions["joy"].short.avg != 6.0  # the new showing was folded
    assert queue_events(persona) == []


def test_the_early_out_does_not_fire_when_the_state_came_from_a_backup(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = install(persona, D[10])
    queue(persona, showing("m1", D[3]))
    engine.note_active(D[10])
    engine_files.state_path(persona).write_bytes(b"\x00 not json")  # the backup copy is now the only usable one
    spy = LockSpy(monkeypatch)
    engine.note_active(D[10])
    assert len(spy.within) == 1


def test_the_early_out_does_not_fire_when_the_day_advanced_or_the_queue_changed(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = install(persona, D[10])
    queue(persona, showing("m1", D[3]))
    engine.note_active(D[3])
    spy = LockSpy(monkeypatch)
    engine.note_active(D[4])  # a later local day
    assert len(spy.within) == 1
    queue(persona, change(D[4]))
    engine.note_active(D[4])  # the queue file's key changed
    assert len(spy.within) == 2


def test_the_early_out_does_not_fire_with_a_missing_or_unreadable_state_and_never_raises(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = install(persona, D[10])
    spy = LockSpy(monkeypatch)
    engine.note_active(D[3])  # no state, no queue: a clean pass that writes nothing
    engine.note_active(D[3])
    assert len(spy.within) == 2
    assert not engine_files.state_path(persona).exists()
    save(persona, simple_state(saved_at=D[2], day=_day(2)))
    for suffix in ("", ".bak1", ".bak2", ".bak3"):
        target = engine_files.state_path(persona).with_name(engine_files.state_path(persona).name + suffix)
        if target.exists():
            target.write_bytes(b"\x00")
    engine.note_active(D[3])  # an unreadable state on a persona not seen switched over: starts empty and writes
    assert len(spy.within) == 3


def test_a_saved_state_with_no_open_day_is_never_skipped_past(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save(persona, EngineState(saved_at=D[2]))  # nothing folded yet: no open day, no last update
    engine = install(persona, D[10])
    spy = LockSpy(monkeypatch)
    engine.note_active(D[3])
    engine.note_active(D[3])
    assert len(spy.within) == 2  # a comparison with a missing day must not be made, and must not raise


# --- folding in time order, the day close (D3, D10) ------------------------------------------------------


def test_queued_events_fold_in_time_order_and_a_day_never_closes_over_a_queued_event_dated_in_it(persona: Path) -> None:
    engine = install(persona)
    # The file holds day 2 FIRST and day 1 second; the drain must fold day 1, close it, then fold day 2.
    queue(persona, showing("m2", D[2], {"joy": 5.0}), showing("m1", D[1], {"joy": 6.0, "sadness": 4.0}))
    engine.note_active(D[3])
    state = state_of(persona)
    joy, sadness = state.emotions["joy"], state.emotions["sadness"]
    # Day 1 closes when day 2's event arrives; day 2 closes at `now` (day 3). Joy had an entry on both days.
    assert [(d.day, d.felt) for d in joy.snapshot] == [(_day(1), True), (_day(2), True)]
    # Sadness had an entry on day 1 only: day 2 closes with a quiet-day zero, so it is not felt that day.
    assert [(d.day, d.felt) for d in sadness.snapshot] == [(_day(1), True), (_day(2), False)]
    assert sadness.short.avg < 4.0 and sadness.long.avg < 4.0  # the zero entry pulled both averages down
    assert state.active_day == _day(3)
    assert state.saved_at == D[3]


def test_the_close_at_now_happens_once_after_the_last_event_and_without_events_a_new_day_still_closes(persona: Path) -> None:
    save(persona, simple_state(saved_at=D[1], joy=4.0, day=_day(1)))
    engine = install(persona)
    engine.note_active(D[3])  # no queue file at all; day 1 is open and day 3 begins
    state = state_of(persona)
    assert state.active_day == _day(3)
    assert [d.day for d in state.emotions["joy"].snapshot] == [_day(1)]


def test_a_showing_after_the_first_halves_within_a_day_and_a_repeat_entry_is_pruned_after_its_window(persona: Path) -> None:
    engine = install(persona)
    at = D[3]
    queue(persona, showing("m1", at, {"joy": 6.0}), showing("m1", at, {"joy": 6.0}), showing("m1", at, {"joy": 6.0}))
    engine.note_active(at)
    state = state_of(persona)
    # weights 1, 1/2, 1/4 at the same instant: 1.75 in all
    assert state.emotions["joy"].short.weight == pytest.approx(1.75)
    assert state.repeats["m1"].count == 2
    # a new queue line, 25 hours later: the entry (first showing 25 h ago) is pruned by the apply step
    queue(persona, change(at + timedelta(hours=25)))
    engine.note_active(at + timedelta(hours=25))
    assert "m1" not in state_of(persona).repeats


def test_read_never_prunes_the_repeat_list(persona: Path) -> None:
    state = simple_state(saved_at=D[3], day=_day(3))
    state.repeats["old"] = RepeatEntry(first_at=D[1], count=1)
    save(persona, state)
    engine = install(persona)
    assert "joy" in engine.read()  # a real read of this state
    assert "old" in state_of(persona).repeats
    queue(persona, change(D[4], {"joy": 1.0}))  # the control: the apply step is what prunes it
    engine.note_active(D[4])
    assert "old" not in state_of(persona).repeats


def test_change_log_files_older_than_the_boundary_go_at_a_day_close_anchored_on_the_day_before_the_close(persona: Path) -> None:
    state = simple_state(saved_at=D[40], day=_day(40))
    save(persona, state)
    folder = engine_files.changes_dir(persona)
    folder.mkdir(parents=True)
    for n in (5, 20, 40):
        (folder / f"{_day(n)}.jsonl").write_text("", encoding="utf-8")
    engine = install(persona)
    engine.note_active(local_noon(local_day(400)))  # a clock jumped far forward: the close is anchored on day 40, not on day 400
    names = sorted(p.name for p in folder.iterdir())
    assert names == [f"{_day(20)}.jsonl", f"{_day(40)}.jsonl"]  # day 5 is 35 days before day 40; day 20 is 20


def test_no_change_log_line_is_written_in_the_shadow_phase_and_faded_values_are_the_stored_ones(persona: Path) -> None:
    engine = install(persona)
    engine.fold_change({"joy": 2.0}, D[3], "pass2")
    engine.note_active(D[3])
    assert not engine_files.changes_dir(persona).exists()
    assert state_of(persona).emotions["joy"].short.avg == 2.0


# --- consumption by position (D4, D5, D6) -----------------------------------------------------------------


def test_a_drain_saves_the_end_position_with_the_fold_then_empties_the_file_then_saves_position_zero(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = install(persona)
    queue(persona, showing("m1", D[3]), showing("m2", D[3]))
    saves = _count_saves(monkeypatch)
    engine.note_active(D[3])
    assert [position for position, _ in saves] == [2, 0]
    assert saves[0][1] > 0 and saves[1][1] == 0  # the first save came before the emptying, the second after it
    assert engine_files.queue_path(persona).exists() and queue_text_lines(persona) == []
    assert state_of(persona).queue_line == 0


def test_a_failed_emptying_keeps_the_position_folds_nothing_twice_saves_no_second_time_and_retries(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = install(persona)
    queue(persona, showing("m1", D[3], {"joy": 6.0}), showing("m2", D[3], {"joy": 2.0}))
    attempts = _fail_emptying(monkeypatch)
    engine.note_active(D[3])
    after_first = state_of(persona)
    assert (after_first.queue_line, len(queue_text_lines(persona)), len(attempts)) == (2, 2, 1)
    saves = _count_saves(monkeypatch)
    engine.note_active(D[3])
    again = state_of(persona)
    assert saves == []  # no second save
    assert again.emotions["joy"].short.weight == after_first.emotions["joy"].short.weight  # nothing folded twice
    assert (again.queue_line, len(attempts)) == (2, 2)  # the emptying was tried again, the position kept
    monkeypatch.undo()
    engine.note_active(D[3])  # the emptying works now: position 0 is saved, the queue is empty
    assert (state_of(persona).queue_line, queue_text_lines(persona)) == (0, [])


def test_an_emptying_that_succeeds_then_new_lines_then_a_drain_folds_only_those(persona: Path) -> None:
    engine = install(persona)
    queue(persona, showing("m1", D[3], {"joy": 6.0}))
    engine.note_active(D[3])
    weight = state_of(persona).emotions["joy"].short.weight
    queue(persona, showing("m2", D[3], {"joy": 6.0}))
    engine.note_active(D[3])
    assert state_of(persona).emotions["joy"].short.weight == pytest.approx(weight + 1.0)  # one more entry, not two


def test_a_drain_that_reads_nothing_and_closes_no_day_does_not_save(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save(persona, simple_state(saved_at=D[3], day=_day(3)))
    engine = install(persona)  # a new instance: nothing remembered, so the early-out cannot answer
    saves = _count_saves(monkeypatch)
    before = tree(persona)
    engine.note_active(D[3])
    assert saves == []
    assert tree(persona) == before
    queue(persona, showing("m1", D[3], {"grief": 2.0}))  # the control: with a line to fold the same fresh instance does save
    install(persona).note_active(D[3])
    assert [position for position, _ in saves] == [1, 0]


def test_a_drain_that_finds_an_empty_queue_under_a_stale_position_saves_the_position_as_zero(persona: Path) -> None:
    """The start-up drain clears a stale position (the maintainer's narrowing of S22): the stored position 5 is past a queue that has
    no lines (a crash between the emptying and the save of position 0, or a failed second save), so nothing is read, no day closes
    and nothing is caught up, and the drain still saves, with position 0. Only the position changes in the state file; the control is
    the early-out test above (a stored position 0 saves nothing).
    """
    before_state = simple_state(saved_at=D[3], day=_day(3), queue_line=5)
    save(persona, before_state)
    assert not engine_files.queue_path(persona).exists()
    install(persona).note_active(D[3])  # a new instance: the start-up drain
    after = state_of(persona)
    assert after.queue_line == 0
    before_state.queue_line = 0
    assert after == before_state  # nothing else moved, `saved_at` included


def test_a_stale_position_over_an_emptied_queue_file_is_cleared_the_same_way(persona: Path) -> None:
    save(persona, simple_state(saved_at=D[3], day=_day(3), queue_line=2))
    queue_file = engine_files.queue_path(persona)
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    queue_file.write_bytes(b"")
    install(persona).note_active(D[3])
    assert state_of(persona).queue_line == 0 and queue_file.read_bytes() == b""


def test_a_position_past_an_emptied_queue_reads_from_the_start(persona: Path) -> None:
    save(persona, simple_state(saved_at=D[3], day=_day(3), queue_line=5))
    queue(persona, showing("m1", D[3], {"grief": 3.0}), showing("m2", D[3], {"grief": 3.0}))
    install(persona).note_active(D[3])
    assert state_of(persona).emotions["grief"].short.weight == 2.0  # both lines folded: the file has 2 lines, the position was 5


def test_a_queue_of_only_bad_lines_still_advances_the_position_and_is_emptied(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save(persona, simple_state(saved_at=D[3], day=_day(3)))
    queue_file = engine_files.queue_path(persona)
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    queue_file.write_bytes(b"not json at all\n{broken\n")
    engine = install(persona)
    _fail_emptying(monkeypatch)
    engine.note_active(D[3])
    assert state_of(persona).queue_line == 2  # `lines read` is "end_line > from", not "events non-empty"
    monkeypatch.undo()
    engine.note_active(D[3])
    assert (state_of(persona).queue_line, queue_file.read_bytes()) == (0, b"")


def test_a_fresh_persona_that_is_not_live_drains_a_first_showing_into_a_new_state_file(persona: Path) -> None:
    engine = install(persona)
    assert not engine_files.state_path(persona).exists()
    queue(persona, showing("m1", D[3], {"joy": 6.0}))
    engine.note_active(D[3])
    assert state_of(persona).emotions["joy"].short.avg == 6.0
    assert engine.status().caught_up


def test_an_unreadable_state_on_a_persona_that_is_not_live_is_replaced_by_an_empty_one_once(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save(persona, simple_state(saved_at=D[3]))
    for suffix in ("", ".bak1", ".bak2", ".bak3"):
        target = engine_files.state_path(persona).with_name(engine_files.state_path(persona).name + suffix)
        if target.exists():
            target.write_bytes(b"\x00")
    engine = install(persona)
    engine.note_active(D[3])  # no lines and no day closed, yet the fallback to an empty state is saved
    assert engine_files.read_state(persona).problem is None
    saves = _count_saves(monkeypatch)
    install(persona).note_active(D[3])  # the failing load is not retried on every call
    assert saves == []


def test_a_drain_over_an_unreadable_state_of_a_persona_this_instance_saw_switched_over_flags_it_and_keeps_everything(
    persona: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A switched persona's state is never replaced by an empty one: the pass is marked failed, nothing is written, the queue is
    kept, and the recovery job repairs it. (An instance that never saw the switch-over cannot tell, and starts empty: the test
    above, accepted edge (a) of Planning B25 SW2.)"""
    save(persona, simple_state(saved_at=D[3], day=_day(3), switch_over_at=D[1]))
    queue(persona, showing("m1", D[4], {"joy": 6.0}))
    engine = install(persona, D[5])
    assert engine.is_live()
    ruin_the_state(persona)
    before = tree(persona / "state")
    with caplog.at_level(logging.DEBUG, logger="brain.emotion.engine"):
        engine.note_active(D[5])
    assert len(engine_errors(caplog)) == 1 and engine._catch_up_failed is True
    assert tree(persona / "state") == before  # the stored copies and the queue lines are exactly as they were
    assert len(queue_events(persona)) == 1


# --- the stamp (V1, B17) and the time given -------------------------------------------------------------------


def test_the_drain_never_stamps_a_saved_at_older_than_the_state_it_loaded_nor_rewinds_the_fold(persona: Path) -> None:
    save(persona, simple_state(saved_at=D[8], day=_day(8)))
    queue(persona, change(D[6], {"joy": 1.0}))
    engine = install(persona)
    engine.note_active(D[5])  # a clock read before a wait for the lock, older than a direct apply's save
    state = state_of(persona)
    assert state.saved_at == D[8]
    assert state.last_update == D[8]
    engine.note_active(D[9])
    assert state_of(persona).saved_at == D[9]


def test_a_naive_now_is_read_as_utc_and_a_now_that_is_not_a_time_is_logged_and_does_nothing(
    persona: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = install(persona)
    queue(persona, showing("m1", D[3]))
    engine.note_active(datetime(2026, 5, 4, 12, 0))
    assert state_of(persona).saved_at == datetime(2026, 5, 4, 12, 0, tzinfo=UTC)
    with caplog.at_level(logging.ERROR):
        engine.note_active("yesterday")  # type: ignore[arg-type]
    assert any("not a time" in m for m in _errors(caplog))


# --- failures: note_active never raises (D7, D8) ---------------------------------------------------------------


def test_a_save_that_raises_keeps_the_queue_and_the_position_and_the_next_drain_retries(
    persona: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    save(persona, simple_state(saved_at=D[3], day=_day(3)))
    engine = install(persona)
    queue(persona, showing("m1", D[3], {"joy": 6.0}))
    real = engine_files.save_state
    monkeypatch.setattr(engine_files, "save_state", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with caplog.at_level(logging.ERROR):
        engine.note_active(D[3])  # must not raise
    assert len(queue_text_lines(persona)) == 1 and state_of(persona).queue_line == 0
    assert any("the drain failed" in m for m in _errors(caplog))
    monkeypatch.setattr(engine_files, "save_state", real)
    engine.note_active(D[3])
    assert queue_text_lines(persona) == []
    assert state_of(persona).emotions["joy"].short.weight > 1.0


def test_a_failing_second_save_leaves_the_position_at_the_end_the_file_emptied_and_the_next_drain_folds_nothing_twice(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = install(persona)
    queue(persona, showing("m1", D[3], {"joy": 6.0}))
    real = engine_files.save_state
    calls: list[int] = []

    def second_fails(persona_dir: Path, state) -> None:  # noqa: ANN001
        calls.append(state.queue_line)
        if state.queue_line == 0:
            raise OSError("the second save failed")
        real(persona_dir, state)

    monkeypatch.setattr(engine_files, "save_state", second_fails)
    engine.note_active(D[3])
    assert calls == [1, 0]
    assert queue_text_lines(persona) == [] and state_of(persona).queue_line == 1  # the S22 edge: position past an emptied file
    weight = state_of(persona).emotions["joy"].short.weight
    monkeypatch.setattr(engine_files, "save_state", real)
    engine.note_active(D[3])  # nothing queued: reads nothing, folds nothing, the key was left unset so it takes the lock
    assert state_of(persona).emotions["joy"].short.weight == weight
    assert state_of(persona).queue_line == 0  # the same drain clears the position the failed save left past the emptied file


def test_a_queue_read_that_raises_oserror_keeps_everything_and_the_next_drain_retries(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = install(persona)
    queue(persona, showing("m1", D[3]))
    real = engine_files.read_queue
    monkeypatch.setattr(engine_files, "read_queue", lambda *a, **k: (_ for _ in ()).throw(PermissionError("held")))
    before = tree(persona)
    engine.note_active(D[3])
    assert tree(persona) == before
    monkeypatch.setattr(engine_files, "read_queue", real)
    engine.note_active(D[3])
    assert queue_text_lines(persona) == [] and "joy" in state_of(persona).emotions


def test_a_lock_wait_that_runs_out_keeps_the_queue_and_never_raises(
    persona: Path, monkeypatch: pytest.MonkeyPatch, held_lock, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.2)
    engine = install(persona)
    queue(persona, showing("m1", D[3]))
    with caplog.at_level(logging.ERROR), held_lock(engine_files.state_path(persona)):
        engine.note_active(D[3])
    assert len(queue_text_lines(persona)) == 1 and not engine_files.state_path(persona).exists()
    assert any("gave up on the engine lock" in m for m in _errors(caplog))
    engine.note_active(D[3])  # the holder is gone: the same call drains
    assert queue_text_lines(persona) == []


def test_an_unexpected_exception_is_logged_with_a_traceback_and_does_not_raise(
    persona: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    engine = install(persona)
    queue(persona, showing("m1", D[3]))
    monkeypatch.setattr(engine_files, "read_state", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("a bug")))
    with caplog.at_level(logging.WARNING):
        engine.note_active(D[3])
    assert any(r.exc_info and "unexpected RuntimeError" in r.getMessage() for r in caplog.records)
    assert len(queue_text_lines(persona)) == 1


def test_a_queue_with_a_bad_line_and_good_ones_folds_the_good_saves_the_end_position_and_logs_the_bad_once(
    persona: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    engine = install(persona)
    queue_file = engine_files.queue_path(persona)
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    good = json.dumps(engine_files.event_to_json(showing("m1", D[3], {"joy": 6.0})))
    queue_file.write_text(good + "\n{not json\n" + good.replace("m1", "m2") + "\n", encoding="utf-8")
    _fail_emptying(monkeypatch)
    with caplog.at_level(logging.ERROR, logger="brain.emotion.engine_files"):
        engine.note_active(D[3])
    state = state_of(persona)
    assert state.queue_line == 3 and state.emotions["joy"].short.weight == 2.0
    assert len([m for m in _errors(caplog) if "line 2 dropped" in m]) == 1


def test_a_queued_event_with_a_huge_finite_modifier_folds_and_the_drain_saves(persona: Path) -> None:
    queue_file = engine_files.queue_path(persona)
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    raw = engine_files.event_to_json(showing("m1", D[3], {"joy": 6.0}))
    raw["modifier"] = 1e308
    queue_file.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    install(persona).note_active(D[3])
    assert state_of(persona).emotions["joy"].short.avg == 6.0  # the modifier was held to its range at the read


def test_the_drain_takes_the_engine_lock_once_with_the_owners_wait_and_never_the_blocking_form(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 7.25)
    spy = LockSpy(monkeypatch)
    engine = install(persona)
    queue(persona, showing("m1", D[3]))
    engine.note_active(D[3])
    assert spy.within == [(engine_files.state_path(persona), 7.25)]
    assert spy.blocking == []


def test_the_engine_and_the_replay_make_no_raw_file_call_and_take_no_lock_of_their_own() -> None:
    """X5 (P18): every file operation of the drain, the direct apply and the replay goes through the shared accessors
    (`file_ops`, `json_io`, `engine_files`); the lock is `engine_files.state_lock` and nothing else."""
    import ast

    from brain.emotion import engine, engine_rebuild

    raw_attrs = {"stat", "exists", "is_file", "is_dir", "unlink", "mkdir", "rmdir", "rename", "replace", "read_text", "write_text", "read_bytes", "write_bytes", "touch", "iterdir", "glob"}
    for module in (engine, engine_rebuild):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "open":
                    offenders.append(f"open() at line {node.lineno}")
                if isinstance(func, ast.Attribute) and func.attr in raw_attrs:
                    offenders.append(f".{func.attr}() at line {node.lineno}")
                if isinstance(func, ast.Attribute) and func.attr in {"file_lock", "locked_within"}:
                    offenders.append(f"a lock call at line {node.lineno}")
            if isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names):
                offenders.append(f"import os at line {node.lineno}")
        assert offenders == [], f"{module.__name__}: {offenders}"
