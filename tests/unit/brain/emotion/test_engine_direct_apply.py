"""The direct apply, recovery first and the batch scope of the emotion engine (Engine 4, #579).

A memory's own emotions (a commit, an assigned update, the start-only date event) are applied to the state under the engine
lock when the memory is written; a showing and a chat change are queued. A state loaded from a backup copy is recovered
first (the catch-up), and the caller's own event is skipped when the replay already counted that memory. Expected numbers
are hand-worked in the comment beside them (a quiet off day does not age a weight: `close_days` moves `last_update` past it).
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine, engine_files, engine_rebuild, engine_utils
from brain.emotion.engine_utils import Average, EmotionState
from brain.memory.store import MemoryStore
from tests.unit.brain.emotion._engine_support import (
    NOW,
    LockSpy,
    commit_event,
    install,
    local_day,
    local_noon,
    lose_the_state,
    make_live,
    make_persona,
    queue,
    queue_events,
    queue_text_lines,
    row,
    ruin_the_state,
    save,
    save_as_backup,
    showing,
    simple_state,
    state_of,
    tree,
)

D = [local_noon(local_day(n)) for n in range(30)]


def _day(n: int) -> str:
    return local_day(n).isoformat()


@pytest.fixture
def world(tmp_path: Path) -> Iterator[tuple[Path, MemoryStore]]:
    persona = make_persona(tmp_path / "persona")
    store = MemoryStore(persona / "memories.db")
    yield persona, store
    store.close()


def _backup_world(tmp_path: Path, rows=(), *, saved_at=None, start=True, live=False):
    """A persona whose ONLY usable state is a backup saved at day 3 (one joy entry, average 4, weight 1), engine clock at day 8.

    With `live` the backup holds a switch-over time (day 1): the persona was switched over, and the copy says so.
    """
    persona = make_persona(tmp_path / "persona", list(rows))
    state = simple_state(saved_at=saved_at or D[3], joy=4.0, day=_day(3), switch_over_at=D[1] if live else None)
    if not start:
        state.persona_start_at = None
    save_as_backup(persona, state)
    return persona, install(persona, D[8])


def _count_saves(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    saves: list[int] = []
    real = engine_files.save_state
    monkeypatch.setattr(engine_files, "save_state", lambda persona_dir, state: (saves.append(state.queue_line), real(persona_dir, state))[1])
    return saves


# --- the split by kind (A1) -----------------------------------------------------------------------------------


def test_a_counted_create_changes_the_state_and_writes_no_queue_line(world) -> None:
    persona, store = world
    install(persona, NOW)
    memory = row(NOW - timedelta(hours=2), emotions={"joy": 6.0})
    store.create(memory)
    state = state_of(persona)
    assert (state.emotions["joy"].short.avg, state.emotions["joy"].short.weight) == (6.0, 1.0)
    assert state.persona_start_at == memory.created_at
    assert state.repeats[memory.id].first_at == memory.created_at  # a commit counts as the first showing
    assert not engine_files.queue_path(persona).exists()


def test_an_assigned_update_changes_the_state_and_writes_no_queue_line(tmp_path: Path) -> None:
    old = row(NOW - timedelta(days=3), emotions={}, mid="old")
    persona = make_persona(tmp_path / "persona", [old])
    install(persona, NOW)
    store = MemoryStore(persona / "memories.db")
    try:
        store.update("old", emotions={"grief": 4.0})
    finally:
        store.close()
    state = state_of(persona)
    assert (state.emotions["grief"].short.avg, state.emotions["grief"].short.weight) == (4.0, 1.0)
    assert not engine_files.queue_path(persona).exists()


def test_a_start_only_event_changes_the_state_and_writes_no_queue_line(world) -> None:
    persona, store = world
    install(persona, NOW)
    memory = row(NOW - timedelta(days=5), memory_type="journal_entry", emotions={})
    store.create(memory)
    assert state_of(persona).persona_start_at == memory.created_at
    assert not engine_files.queue_path(persona).exists()


def test_a_showing_and_a_change_write_a_queue_line_and_not_the_state(world) -> None:
    persona, store = world
    eng = install(persona, NOW)
    memory = row(NOW - timedelta(days=1), emotions={"joy": 6.0})
    eng.on_memory_shown(memory, NOW)
    eng.fold_change({"joy": 1.0}, NOW, "pass2")
    assert [line["kind"] for line in queue_text_lines(persona)] == ["showing", "change"]
    assert not engine_files.state_path(persona).exists()


# --- the lock (A2, A2b) -----------------------------------------------------------------------------------------


def test_every_acquisition_passes_the_owners_wait_and_no_engine_call_takes_the_blocking_form(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona, store = world
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 7.25)
    eng = install(persona, NOW)
    spy = LockSpy(monkeypatch)
    state_file = engine_files.state_path(persona)
    store.create(row(NOW - timedelta(hours=2)))  # direct apply
    assert spy.within == [(state_file, 7.25)]
    eng.on_memory_shown(row(NOW), NOW)  # a queue append
    assert spy.within[-1] == (state_file, 7.25) and len(spy.within) == 2
    eng.note_active(NOW)  # the drain
    assert len(spy.within) == 3
    with eng.batch():  # a batch's end
        store.create(row(NOW - timedelta(hours=1)))
    assert len(spy.within) == 4 and all(wait == 7.25 for _, wait in spy.within)
    assert spy.blocking == []


def test_a_direct_apply_whose_lock_wait_runs_out_logs_and_leaves_the_memory_saved(
    world, monkeypatch: pytest.MonkeyPatch, held_lock, caplog: pytest.LogCaptureFixture
) -> None:
    persona, store = world
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.2)
    install(persona, NOW)
    memory = row(NOW - timedelta(hours=2))
    with caplog.at_level(logging.ERROR, logger="brain.emotion.engine"), held_lock(engine_files.state_path(persona)):
        assert store.create(memory) == memory.id
    assert not engine_files.state_path(persona).exists()  # F-DA: the memory stays saved, its emotions are not applied
    assert store.get(memory.id, bump=False) is not None
    assert any("not applied" in r.getMessage() and "memory is saved" in r.getMessage() for r in caplog.records)


# --- recovery first (A3, A4, A9) -----------------------------------------------------------------------------


def test_a_direct_apply_on_a_backup_state_runs_the_catch_up_first_saves_once_and_counts_a_new_row_once_with_no_climax_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona, eng = _backup_world(tmp_path)
    store = MemoryStore(persona / "memories.db")
    spy = LockSpy(monkeypatch)
    saves = _count_saves(monkeypatch)
    try:
        # created after the backup's saved_at, stored climax 9: the replay folds it (no reset), its commit event is skipped
        store.create(row(D[5], emotions={"joy": 6.0, "arousal": 8.0, "climax": 9.0}, mid="x"))
    finally:
        store.close()
    state = state_of(persona)
    # joy: backup average 4 (weight 1, last update day 3). Day 4 is an off day and ages nothing, so at day 5 noon the weight
    # is 1 x 0.5^(24/24) = 0.5; the row adds 1: weight 1.5, average 4 + (6 - 4) / 1.5. Applied twice it would not be this.
    assert state.emotions["joy"].short.weight == pytest.approx(1.5)
    assert state.emotions["joy"].short.avg == pytest.approx(4.0 + 2.0 / 1.5)
    assert state.emotions["arousal"].short.avg == 8.0  # a climax reset would have made it 1.6
    assert (state.reset_at, state.saved_at, state.queue_line) == (D[8], D[8], 0)
    assert eng.status().caught_up
    assert saves == [0] and len(spy.within) == 1 and spy.blocking == []  # one save, one lock hold, the catch-up inside it


def test_an_assigned_event_on_a_row_created_before_saved_at_and_not_opened_since_is_applied_once(tmp_path: Path) -> None:
    persona, _ = _backup_world(tmp_path, [row(D[1], emotions={}, mid="old")])
    store = MemoryStore(persona / "memories.db")
    try:
        store.update("old", emotions={"sadness": 2.0})
    finally:
        store.close()
    sadness = state_of(persona).emotions["sadness"]
    # The window does not return the row, so the event applies, once: dated day 1, two days before the state's last update,
    # it is pre-aged to 0.5^(48/24) = 0.25. The day close at `now` then adds a quiet-day zero (weight 1, no entry on day 3)
    # dated half a day later, which ages the 0.25 by 0.5^(12/24): 1 + 0.25 x 0.7071. Applied twice it would be 1 + 0.5 x 0.7071.
    assert sadness.short.weight == pytest.approx(1.0 + 0.25 * 0.5**0.5)


def test_an_assigned_event_on_an_older_row_that_was_also_opened_since_is_skipped(tmp_path: Path) -> None:
    persona, _ = _backup_world(tmp_path, [row(D[1], accessed=D[5], emotions={}, mid="old")])
    store = MemoryStore(persona / "memories.db")
    try:
        store.update("old", emotions={"sadness": 2.0})
    finally:
        store.close()
    # The replay folds the row once at its access (day 5, a new emotion: weight 1); the event is skipped, not added again.
    # The day close at `now` closes day 5 with a quiet-day zero for joy dated half a day later, which ages sadness: 0.5^(12/24).
    assert state_of(persona).emotions["sadness"].short.weight == pytest.approx(0.5**0.5)


def test_an_assigned_event_on_an_older_row_opened_at_or_after_the_oldest_queued_line_is_applied_once(tmp_path: Path) -> None:
    """R3b (ii): the replay stops at the oldest queued line, so a row opened after it folds nothing and is NOT in the skip list."""
    persona, _ = _backup_world(tmp_path, [row(D[1], accessed=D[7], emotions={}, mid="old")])
    queue(persona, showing("elsewhere", D[6], {"joy": 1.0}))
    store = MemoryStore(persona / "memories.db")
    try:
        store.update("old", emotions={"sadness": 2.0})
    finally:
        store.close()
    # applied once, pre-aged by 2 days (0.25), then the day close at `now` adds a zero and ages it (as in the first case above)
    assert state_of(persona).emotions["sadness"].short.weight == pytest.approx(1.0 + 0.25 * 0.5**0.5)
    assert len(queue_text_lines(persona)) == 1  # a direct apply leaves the queue to the drain


def test_a_counted_row_created_exactly_at_saved_at_is_folded_by_the_replay_and_its_event_skipped_never_lost(tmp_path: Path) -> None:
    persona, _ = _backup_world(tmp_path)
    store = MemoryStore(persona / "memories.db")
    try:
        store.create(row(D[3], emotions={"grief": 3.0}, mid="edge"))
    finally:
        store.close()
    grief = state_of(persona).emotions["grief"]
    assert grief.short.weight == pytest.approx(1.0)  # folded once (a lost row would be absent, a doubled one 2.0)


def test_a_start_only_event_is_never_skipped(tmp_path: Path) -> None:
    persona, _ = _backup_world(tmp_path, start=False)
    store = MemoryStore(persona / "memories.db")
    try:
        store.create(row(D[5], memory_type="journal_entry", emotions={}))
    finally:
        store.close()
    assert state_of(persona).persona_start_at == D[5]


def test_a_batch_on_a_backup_state_recovers_first_and_applies_once_each_the_events_the_replay_did_not_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona, eng = _backup_world(tmp_path, [row(D[1], emotions={}, mid="old")])
    store = MemoryStore(persona / "memories.db")
    spy = LockSpy(monkeypatch)
    saves = _count_saves(monkeypatch)
    try:
        with eng.batch():
            store.create(row(D[5], emotions={"grief": 3.0}, mid="new"))  # in the window: counted by the replay, event skipped
            store.update("old", emotions={"sadness": 2.0})  # not in the window: applied once
            assert saves == []  # nothing applied inside the block
    finally:
        store.close()
    state = state_of(persona)
    # grief: folded by the replay at day 5 (weight 1), once; aged half a day by the close of day 5 (a zero for joy): 0.5^(12/24)
    assert state.emotions["grief"].short.weight == pytest.approx(0.5**0.5)
    # sadness: applied once, dated day 1 against a last update at day 5 (the replay moved it): 0.5^(96/24) = 0.0625; the close
    # of day 5 adds a zero (weight 1) dated half a day later: 1 + 0.0625 x 0.7071
    assert state.emotions["sadness"].short.weight == pytest.approx(1.0 + 0.0625 * 0.5**0.5)
    assert saves == [0] and len(spy.within) == 1


def test_the_drains_own_recovery_runs_the_catch_up_first_then_folds_every_queue_line_once(tmp_path: Path) -> None:
    persona, eng = _backup_world(
        tmp_path,
        [row(D[5], emotions={"grief": 3.0}, mid="r1"), row(D[1], accessed=D[7], emotions={"sadness": 2.0}, mid="r2")],
    )
    queue(persona, showing("s", D[6], {"joy": 1.0}))
    eng.note_active(D[8])
    state = state_of(persona)
    assert "sadness" not in state.emotions  # opened at day 7, after the oldest queued line (day 6): its queued showing folds it
    # grief: folded at its creation (day 5) with weight 1; the showing at day 6 noon is 24 h later, so the weight is 0.5 by then
    # (the close of day 5 in between is a zero for joy only); the close of day 6 at `now` adds a zero for grief (no entry on
    # day 6), dated half a day later: 1 + 0.5 x 0.7071
    assert state.emotions["grief"].short.weight == pytest.approx(1.0 + 0.5 * 0.5**0.5)
    assert (state.reset_at, state.queue_line, queue_events(persona)) == (D[8], 0, [])
    assert eng.status().caught_up


def test_a_queued_showing_for_a_skip_listed_row_still_folds_once_and_a_queued_commit_line_for_it_does_not(tmp_path: Path) -> None:
    """Round-4 MAJOR 3: the skip list is for commit and assigned events only; a showing is never skipped."""
    persona, eng = _backup_world(tmp_path, [row(D[5], accessed=D[8], emotions={"grief": 3.0}, mid="x")])
    queue(persona, commit_event("x", D[5], {"grief": 3.0}), showing("x", D[8], {"grief": 3.0}))
    eng.note_active(D[10])
    grief = state_of(persona).emotions["grief"]
    # replay: created day 5 (the oldest queued line is day 5, the access at day 8 is after it): weight 1. The commit line is
    # skipped. The showing at day 8 follows two off days: aged 0.5 (24 h) + 1 = 1.5. The close of day 8 at `now` adds a zero
    # for joy dated half a day later, which ages grief: 1.5 x 0.7071. A commit folded as well would add more.
    assert grief.short.weight == pytest.approx(1.5 * 0.5**0.5)


# --- failures of recovery and of the load (A5, A5b, A6, A7, A8c) ----------------------------------------------


def _failing_catch_up(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("the replay failed")

    monkeypatch.setattr(engine_rebuild, "catch_up", boom)


def test_a_failed_recovery_on_a_live_persona_applies_nothing_keeps_the_queue_flags_it_and_logs_once_per_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    persona, eng = _backup_world(tmp_path, live=True)
    queue(persona, showing("s", D[7]))
    _failing_catch_up(monkeypatch)
    store = MemoryStore(persona / "memories.db")
    before = tree(persona / "state")
    try:
        with caplog.at_level(logging.ERROR, logger="brain.emotion.engine"):
            first = row(D[7], emotions={"joy": 6.0})
            store.create(first)
            store.create(row(D[7], emotions={"joy": 6.0}))
            eng.note_active(D[8])
        assert store.get(first.id, bump=False) is not None  # each memory stays saved
    finally:
        store.close()
    assert tree(persona / "state") == before  # nothing written; the queue line and its position are kept
    assert eng.status().caught_up is False
    assert eng.read() == {}
    assert len([r for r in caplog.records if "catch-up of the backup state failed" in r.getMessage()]) == 1  # once per active day


def test_a_failed_recovery_on_a_persona_that_is_not_live_skips_the_pass_keeps_the_queue_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning B31 SW4: the failed-catch-up site follows the unreadable-state rule: no start from an empty state."""
    persona, eng = _backup_world(tmp_path)
    _failing_catch_up(monkeypatch)
    queue(persona, showing("s", D[7], {"joy": 6.0}))
    before = tree(persona / "state")
    eng.note_active(D[8])
    assert tree(persona / "state") == before  # no new main file, the queue line and its position kept
    assert engine_files.read_state(persona).source == "bak1" and len(queue_events(persona)) == 1


def test_a_direct_apply_on_an_unreadable_state_of_a_persona_this_instance_has_not_seen_switched_skips_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """Planning B31 SW4 (supersedes B25 edge (a)): with every copy of the state unreadable nothing can say whether the persona
    was switched over, so it reads as not switched, and the apply is skipped: the unreadable copies are left for the first
    rebuild at the next start, which decides from the database's x10 record. The memory is saved all the same."""
    persona = make_persona(tmp_path / "persona")
    save(persona, simple_state(saved_at=D[3], day=_day(3), switch_over_at=D[1]))
    ruin_the_state(persona)
    eng = install(persona, D[8])  # a new instance: it never saw the switch-over
    before = tree(persona / "state")
    store = MemoryStore(persona / "memories.db")
    try:
        memory = row(D[7], emotions={"joy": 6.0})
        store.create(memory)
        assert store.get(memory.id, bump=False) is not None
    finally:
        store.close()
    assert tree(persona / "state") == before
    assert engine_files.read_state(persona).problem == "unreadable"
    assert eng.is_live() is False


def test_an_unreadable_state_on_a_live_persona_is_never_overwritten_and_the_direct_apply_logs_and_flags(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    persona = make_persona(tmp_path / "persona")
    save(persona, simple_state(saved_at=D[3], day=_day(3), switch_over_at=D[1]))
    eng = install(persona, D[8])
    assert eng.is_live()  # a running engine has seen the switch-over; the state is damaged afterwards
    for suffix in ("", ".bak1", ".bak2", ".bak3"):
        target = engine_files.state_path(persona).with_name(engine_files.state_path(persona).name + suffix)
        if target.exists():
            target.write_bytes(b"\x00")
    store = MemoryStore(persona / "memories.db")
    before = tree(persona / "state")
    try:
        with caplog.at_level(logging.ERROR, logger="brain.emotion.engine"):
            memory = row(D[7], emotions={"joy": 6.0})
            store.create(memory)
        assert store.get(memory.id, bump=False) is not None
    finally:
        store.close()
    assert tree(persona / "state") == before
    assert eng.status() == engine.EngineStatus(False, None)
    assert any("unreadable" in r.getMessage() for r in caplog.records)


def test_a_live_persona_with_no_state_is_flagged_and_nothing_is_written(world, caplog: pytest.LogCaptureFixture) -> None:
    persona, store = world
    make_live(persona, D[1])
    eng = install(persona, NOW)
    assert eng.is_live()  # the running engine has seen the switch-over; the state is gone afterwards
    lose_the_state(persona)
    store.create(row(NOW - timedelta(hours=1)))
    assert not engine_files.state_path(persona).exists()
    assert eng.status().caught_up is False  # observed through status(), not through read()


def test_read_on_a_corrupted_state_recovers_nothing_writes_nothing_and_returns_the_backups_values(tmp_path: Path) -> None:
    persona, eng = _backup_world(tmp_path)
    before = tree(persona / "state")
    readings = eng.read()
    assert tree(persona / "state") == before
    assert readings["joy"].short == 4.0 and readings["joy"].long == 4.0  # a `{}` stub fails here
    assert eng.status().caught_up is False


def test_a_state_repaired_outside_the_engine_clears_the_failure_at_the_next_load_and_a_write_does_too(tmp_path: Path) -> None:
    persona = make_persona(tmp_path / "persona")
    save(persona, simple_state(saved_at=D[3], day=_day(3), switch_over_at=D[1]))
    eng = install(persona, D[8])
    assert eng.is_live()  # the running engine has seen the switch-over; the state is damaged afterwards
    for suffix in ("", ".bak1", ".bak2", ".bak3"):
        target = engine_files.state_path(persona).with_name(engine_files.state_path(persona).name + suffix)
        if target.exists():
            target.write_bytes(b"\x00")
    eng.note_active(D[8])
    assert eng.status().caught_up is False and eng.read() == {}
    save(persona, simple_state(saved_at=D[7], joy=5.0, day=_day(7)))  # repaired by someone else
    eng.note_active(D[7])  # a set flag defeats the early-out, so this drain loads the repaired main file
    assert eng.status().caught_up is True
    assert eng.read()["joy"].short == 5.0  # read() works again with nothing to save


# --- batch (A8, A8b) ------------------------------------------------------------------------------------------


def _three_rows() -> list:
    return [
        row(NOW - timedelta(hours=3), emotions={"joy": 6.0}),
        row(NOW - timedelta(hours=2), emotions={"joy": 2.0, "grief": 4.0}),
        row(NOW - timedelta(hours=1), emotions={"grief": 1.0}),
    ]


def test_n_creates_in_a_batch_save_once_and_give_the_numbers_n_separate_applies_give(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    separate = make_persona(tmp_path / "separate")
    batched = make_persona(tmp_path / "batched")
    rows = _three_rows()
    saves = _count_saves(monkeypatch)
    install(separate, NOW)
    with contextlib.closing(MemoryStore(separate / "memories.db")) as store:
        for memory in rows:
            store.create(memory)
    assert saves == [0, 0, 0]
    saves.clear()
    eng = install(batched, NOW)
    with contextlib.closing(MemoryStore(batched / "memories.db")) as store, eng.batch():
        for memory in rows:
            store.create(row(memory.created_at, emotions=memory.emotions))
        assert saves == []
    assert saves == [0]
    one, other = state_of(separate), state_of(batched)
    assert set(one.emotions) == set(other.emotions)
    for name in one.emotions:
        assert other.emotions[name].short.avg == pytest.approx(one.emotions[name].short.avg)
        assert other.emotions[name].short.weight == pytest.approx(one.emotions[name].short.weight)
        assert other.emotions[name].long.avg == pytest.approx(one.emotions[name].long.avg)
    assert other.last_update == one.last_update


def test_an_exception_inside_a_batch_still_applies_the_events_collected_and_goes_on_out(world) -> None:
    persona, store = world
    eng = install(persona, NOW)
    with pytest.raises(RuntimeError, match="from the block"):
        with eng.batch():
            store.create(row(NOW - timedelta(hours=2), emotions={"joy": 6.0}))
            raise RuntimeError("from the block")
    assert state_of(persona).emotions["joy"].short.avg == 6.0


def test_nested_batches_apply_once_when_the_outermost_ends(world, monkeypatch: pytest.MonkeyPatch) -> None:
    persona, store = world
    eng = install(persona, NOW)
    saves = _count_saves(monkeypatch)
    with eng.batch():
        with eng.batch():
            store.create(row(NOW - timedelta(hours=2)))
        assert saves == []  # the inner block's end applies nothing
        store.create(row(NOW - timedelta(hours=1)))
    assert saves == [0]


def test_a_hook_on_another_thread_applies_directly_while_this_thread_collects(world) -> None:
    persona, store = world
    eng = install(persona, NOW)
    other = row(NOW - timedelta(hours=2), emotions={"grief": 5.0})

    def work() -> None:
        with contextlib.closing(MemoryStore(persona / "memories.db")) as theirs:
            theirs.create(other)

    with eng.batch():
        store.create(row(NOW - timedelta(hours=1), emotions={"joy": 6.0}))
        thread = threading.Thread(target=work)
        thread.start()
        thread.join(10)
        assert "grief" in state_of(persona).emotions  # applied at once, outside this thread's block
        assert "joy" not in state_of(persona).emotions  # still collected
    assert "joy" in state_of(persona).emotions


def test_two_events_for_one_memory_in_a_batch_fold_once(world) -> None:
    persona, _ = world
    eng = install(persona, NOW)
    memory = row(NOW - timedelta(hours=2), emotions={"joy": 6.0})
    with eng.batch():
        eng.on_memory_committed(memory)
        eng.on_memory_committed(memory)
    assert state_of(persona).emotions["joy"].short.weight == 1.0


# --- validation of every route (M1, D8b, the partial fold) -----------------------------------------------------


def test_a_naive_and_an_offset_created_at_are_read_as_utc_instants_by_the_direct_apply(world) -> None:
    persona, store = world
    install(persona, NOW)
    naive = row(datetime(2026, 6, 10, 9, 0), emotions={"joy": 6.0})
    offset = row(datetime(2026, 6, 10, 18, 0, tzinfo=timezone(timedelta(hours=5))), emotions={"grief": 2.0})  # 13:00 UTC
    store.create(naive)
    assert state_of(persona).last_update == datetime(2026, 6, 10, 9, 0, tzinfo=UTC)
    store.create(offset)
    state = state_of(persona)
    assert state.last_update == datetime(2026, 6, 10, 13, 0, tzinfo=UTC)
    assert state.emotions["grief"].last_entry_day == engine_utils.day_of(datetime(2026, 6, 10, 13, 0, tzinfo=UTC))


def test_a_direct_event_and_a_batch_member_with_a_time_that_cannot_become_a_day_are_logged_and_skipped_the_others_fold(
    world, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    persona, store = world
    eng = install(persona, NOW)
    real = engine_utils.to_local

    def failing(dt: datetime) -> datetime:
        if dt.year < 1970:
            raise OverflowError("date value out of range")
        return real(dt)

    monkeypatch.setattr(engine_utils, "to_local", failing)
    bad = row(datetime(1969, 12, 31, 12, 0, tzinfo=UTC), emotions={"joy": 6.0})
    bad_in_batch = row(datetime(1969, 12, 31, 12, 0, tzinfo=UTC), emotions={"joy": 6.0})
    good = row(NOW - timedelta(hours=1), emotions={"grief": 2.0})
    with caplog.at_level(logging.WARNING):
        store.create(bad)  # a direct apply: nothing to fold, nothing raised
        assert not engine_files.state_path(persona).exists()
        with eng.batch():
            store.create(bad_in_batch)
            store.create(good)
    assert set(state_of(persona).emotions) == {"grief"}
    assert any("was dropped" in r.getMessage() for r in caplog.records)


def test_an_event_that_raises_after_it_changed_the_state_is_put_back_and_the_next_event_still_folds(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-4 MAJOR 2, through the drain: no trace of a half-folded event may reach the saved state."""
    persona, _ = world
    eng = install(persona, NOW)
    real = engine.fold_event

    def folds_then_raises(state, event, now):  # noqa: ANN001
        if event.memory_id == "bad":
            state.emotions["ghost"] = EmotionState(short=Average(9.0, 9.0), long=Average(9.0, 9.0))
            raise RuntimeError("a bug after the state changed")
        return real(state, event, now)

    monkeypatch.setattr(engine, "fold_event", folds_then_raises)
    queue(persona, showing("bad", NOW - timedelta(hours=2), {"joy": 6.0}), showing("good", NOW - timedelta(hours=1), {"grief": 2.0}))
    eng.note_active(NOW)
    assert set(state_of(persona).emotions) == {"grief"}


@pytest.mark.parametrize("route", ["drain", "direct"])
def test_a_day_close_that_raises_after_it_changed_the_state_is_put_back_and_the_next_event_still_folds(
    world, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """Round-4 MAJOR 2, the `close_days` half: the close and the fold of an event are ONE guarded step."""
    persona, store = world
    eng = install(persona, NOW)
    bad_at, good_at = NOW - timedelta(hours=2), NOW - timedelta(hours=1)
    real = engine.close_days

    def closes_then_raises(state, t):  # noqa: ANN001
        if t == bad_at:
            state.emotions["ghost"] = EmotionState(short=Average(9.0, 9.0), long=Average(9.0, 9.0))
            raise RuntimeError("a bug after the state changed")
        return real(state, t)

    monkeypatch.setattr(engine, "close_days", closes_then_raises)
    if route == "drain":
        queue(persona, showing("bad", bad_at, {"joy": 6.0}), showing("good", good_at, {"grief": 2.0}))
        eng.note_active(NOW)
    else:
        with eng.batch():
            store.create(row(bad_at, emotions={"joy": 6.0}))
            store.create(row(good_at, emotions={"grief": 2.0}))
    assert set(state_of(persona).emotions) == {"grief"}


def test_the_accepted_edge_comments_are_in_the_source() -> None:
    """D11, A10: the comments the spec asks for (S22, S42, F-DA, the climax-skip edge, the RD7 case, catch-up holding the lock)."""
    source = Path(engine.__file__).read_text(encoding="utf-8")
    for needle in ("S22", "S42", "F-DA", "climax reset", "RD7", "E5.13"):
        assert needle in source, needle
    rebuild = Path(engine_rebuild.__file__).read_text(encoding="utf-8")
    for needle in ("holds the caller's lock", "same-time order", "straddles"):
        assert needle in rebuild, needle
