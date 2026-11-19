"""`engine_rebuild.replay` and `catch_up` (Engine 4, #579): the recovery-first replay of a state loaded from a backup copy.

Every expected number is hand-worked in the comment beside it. Times are local noon of a civil day (`local_noon`).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files, engine_rebuild, engine_utils
from brain.emotion.engine_rebuild import CatchUpError, catch_up, replay
from brain.emotion.engine_utils import EngineState, RepeatEntry, day_of
from brain.memory.store import Memory, MemoryStore
from tests.unit.brain.emotion._engine_support import (
    change,
    local_day,
    local_noon,
    make_persona,
    queue_events,
    row,
    save,
    simple_state,
    tree,
)

D = [local_noon(local_day(n)) for n in range(12)]  # D[n] = local noon of day n


def _emotions_of(memory: Memory) -> dict:
    return dict(memory.emotions)


def _mods(**by_id: float):
    return lambda memory: by_id.get(memory.id, 1.0)


def _run(
    rows=(),
    stamps=None,
    changes=(),
    *,
    now=None,
    after=None,
    stop=None,
    modifiers=None,
    emotions_of=_emotions_of,
    state=None,
):
    state = EngineState() if state is None else state
    rows = list(rows)
    stamps = [r.created_at for r in rows] if stamps is None else stamps
    folded = replay(
        state,
        rows=rows,
        stamps=stamps,
        changes=list(changes),
        now=D[10] if now is None else now,
        after=D[1] if after is None else after,
        stop=stop,
        emotions_of=emotions_of,
        modifier_of=modifiers or _mods(),
    )
    return state, folded


# --- row points: one fold per row, at the right time ------------------------------------------------


def test_a_row_created_in_the_window_folds_once_at_its_creation() -> None:
    memory = row(D[3], emotions={"joy": 6.0}, mid="a")
    state, folded = _run([memory])
    # one entry of value 6 at weight 1 (modifier 1, no repeat factor): average 6, weight 1
    assert folded == {"a"}
    assert (state.emotions["joy"].short.avg, state.emotions["joy"].short.weight) == (6.0, 1.0)
    assert state.last_update == D[3]
    assert state.persona_start_at == D[3]
    assert state.emotions["joy"].last_entry_day == day_of(D[3])


def test_a_row_opened_inside_the_window_folds_once_at_its_access_time() -> None:
    memory = row(D[0], accessed=D[4], mid="a")  # created before the window, opened inside [D1, D6)
    state, folded = _run([memory], stop=D[6])
    assert folded == {"a"}
    assert state.last_update == D[4]


def test_a_row_created_in_the_window_and_opened_inside_it_folds_once_at_the_later_time() -> None:
    memory = row(D[2], accessed=D[4], mid="a")
    state, folded = _run([memory], stop=D[6])
    assert folded == {"a"}
    assert state.last_update == D[4]
    assert state.emotions["joy"].short.weight == 1.0  # once, not twice


def test_a_row_created_in_the_window_and_opened_at_or_after_the_stop_folds_once_at_its_creation() -> None:
    """R3b (i): created after `after`, accessed at or after `stop`: folded at creation, in the list; the access is the queued showing's."""
    memory = row(D[2], accessed=D[7], mid="a")
    state, folded = _run([memory], stop=D[6])
    assert folded == {"a"}
    assert state.last_update == D[2]


def test_a_row_created_before_the_window_and_opened_at_or_after_the_stop_folds_nothing_and_is_not_listed() -> None:
    """R3b (ii): the queued showing folds it; the skip list must not name it, so a later `assigned` event for it still applies."""
    memory = row(D[0], accessed=D[7], mid="a")
    state, folded = _run([memory], stop=D[6])
    assert folded == frozenset()
    assert state.emotions == {}


def test_the_lower_bound_is_inclusive_for_rows_stamps_and_changes_and_the_stop_is_exclusive() -> None:
    created_at_the_bound = row(D[1], mid="created")
    opened_at_the_bound = row(D[0], accessed=D[1], mid="opened")
    state, folded = _run([created_at_the_bound, opened_at_the_bound], stop=D[6])
    assert folded == {"created", "opened"}
    # a change line dated exactly at the lower bound is replayed (it can fold twice: accepted overlap)
    state, _ = _run(changes=[change(D[1], {"joy": 2.0})], stop=D[6])
    assert state.emotions["joy"].short.avg == 2.0  # no history: base 0, plus 2
    # a stamp exactly at the bound is replayed
    state, _ = _run(rows=[], stamps=[D[1]])
    assert state.persona_start_at == D[1]
    # an access exactly at the stop is not an access inside the window: created before the window, so nothing folds
    state, folded = _run([row(D[0], accessed=D[6], mid="late")], stop=D[6])
    assert folded == frozenset()
    # a change exactly at the stop is not replayed (it is a queued kind: the drain folds it)
    state, _ = _run(changes=[change(D[6], {"joy": 2.0})], stop=D[6])
    assert state.emotions == {}


def test_an_uncounted_row_folds_nothing_but_its_creation_stamp_still_keeps_the_start_date() -> None:
    memory = row(D[3], memory_type="reflex_arc", mid="u")
    state, folded = _run([memory], emotions_of=lambda m: {})
    assert folded == frozenset()
    assert state.emotions == {}
    assert state.persona_start_at == D[3]  # R4: min-folded from the stamp, through a start-only event
    assert state.active_day == day_of(D[3])


def test_the_start_date_is_the_earliest_stamp_of_the_window() -> None:
    state, _ = _run([row(D[5], mid="a"), row(D[3], mid="b")])
    assert state.persona_start_at == D[3]


def test_a_row_folds_at_its_source_modifier() -> None:
    memory = row(D[3], emotions={"joy": 6.0}, mid="peer")
    state, _ = _run([memory], modifiers=_mods(peer=0.3))
    assert (state.emotions["joy"].short.avg, state.emotions["joy"].short.weight) == (6.0, 0.3)


def test_changes_fold_as_change_entries_on_top_of_the_stored_average() -> None:
    state, _ = _run([row(D[2], emotions={"joy": 4.0}, mid="a")], changes=[change(D[3], {"joy": 1.0})], stop=D[8])
    # row: average 4 at weight 1 on day 2. A day later the change adds 1 to the stored average: value 5 at weight 1.
    # The weights age 1 day first (short: 0.5^(24/24)=0.5): short weight 0.5 + 1 = 1.5, average 4 + (5 - 4) * 1 / 1.5.
    assert state.emotions["joy"].short.avg == pytest.approx(4.0 + 1.0 / 1.5)


def test_ties_fold_in_read_order_a_row_before_a_change_at_the_same_instant() -> None:
    state, _ = _run([row(D[3], emotions={"joy": 4.0}, mid="a")], changes=[change(D[3], {"joy": 1.0})], stop=D[8])
    # row first: average 4, weight 1; the change adds 1 to it: value 5 at weight 1: average 4 + (5 - 4) / 2 = 4.5.
    # (change first would give: average 1, then the row: 1 + (4 - 1) / 2 = 2.5.)
    assert state.emotions["joy"].short.avg == pytest.approx(4.5)


def test_a_climax_reset_is_not_replayed() -> None:
    state, _ = _run([row(D[3], emotions={"arousal": 8.0, "climax": 9.0}, mid="a")])
    assert state.emotions["arousal"].short.avg == 8.0
    assert state.emotions["climax"].short.avg == 9.0


# --- repeats -------------------------------------------------------------------------------------------


def test_a_row_created_within_the_repeat_window_gets_one_first_showing_entry() -> None:
    now = D[10]
    fresh = row(now - timedelta(hours=3), mid="fresh")
    old = row(now - timedelta(hours=30), mid="old")
    opened_only = row(D[0], accessed=now - timedelta(hours=2), mid="opened")
    state, _ = _run([fresh, old, opened_only], now=now, stop=None)
    assert set(state.repeats) == {"fresh"}
    assert state.repeats["fresh"] == RepeatEntry(first_at=fresh.created_at, count=0)


# --- the hold at now, normalised times, and one bad point --------------------------------------------


def test_a_row_dated_ahead_of_now_folds_at_now_not_at_the_future_time() -> None:
    """Round-4 MAJOR 1: `fold_entry` does not hold a future time; replay does."""
    now = D[10]
    memory = row(now + timedelta(hours=2), mid="future")
    state, folded = _run([memory], now=now)
    assert folded == {"future"}
    assert state.last_update == now
    assert state.emotions["joy"].last_entry_day == day_of(now)
    assert state.repeats["future"].first_at == now  # the repeat record is dated like the fold, so the next showing is halved
    again, _ = _run([memory], now=now)
    assert engine_files.state_to_json(_stamped(again)) == engine_files.state_to_json(_stamped(state))


def _stamped(state: EngineState) -> EngineState:
    state.saved_at = D[10]
    return state


def test_a_naive_and_an_offset_created_at_are_read_as_utc_instants_and_folded_on_their_own_day() -> None:
    naive = row(datetime(2026, 5, 4, 12, 0), mid="naive")  # no zone: UTC
    offset = row(datetime(2026, 5, 5, 23, 30, tzinfo=timezone(timedelta(hours=5))), mid="offset")  # 18:30 UTC
    state, folded = _run([naive, offset], stamps=[naive.created_at, offset.created_at])
    assert folded == {"naive", "offset"}
    assert state.last_update == datetime(2026, 5, 5, 18, 30, tzinfo=UTC)
    assert state.emotions["joy"].last_entry_day == day_of(datetime(2026, 5, 5, 18, 30, tzinfo=UTC))


def test_a_row_whose_time_cannot_become_a_local_day_is_logged_and_skipped_and_the_others_fold(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = engine_utils.to_local

    def failing(dt: datetime) -> datetime:
        if dt.year < 1970:
            raise OverflowError("date value out of range")
        return real(dt)

    monkeypatch.setattr(engine_utils, "to_local", failing)
    bad = row(datetime(1969, 12, 31, 12, 0, tzinfo=UTC), mid="bad")
    good = row(D[3], mid="good")
    with caplog.at_level(logging.WARNING):
        state, folded = _run([bad, good], stamps=[bad.created_at, good.created_at], after=datetime(1960, 1, 1, tzinfo=UTC))
    assert folded == {"good"}
    assert state.emotions["joy"].short.weight == 1.0
    assert "bad" in " ".join(r.getMessage() for r in caplog.records)


def test_a_point_that_raises_midway_is_put_back_as_it_was_and_the_next_point_still_folds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Round-4 MAJOR 2: `joy` folds, then `grief` raises: no trace of the row may stay in the state."""
    real = engine_rebuild.fold_entry

    def folds_then_raises(state, name, value, w, at, **kwargs):
        if name == "grief":
            raise RuntimeError("a bug after one emotion was folded")
        return real(state, name, value, w, at, **kwargs)

    monkeypatch.setattr(engine_rebuild, "fold_entry", folds_then_raises)
    broken = row(D[2], emotions={"joy": 6.0, "grief": 3.0}, mid="broken")
    fine = row(D[3], emotions={"sadness": 2.0}, mid="fine")
    with caplog.at_level(logging.WARNING):
        state, folded = _run([broken, fine])
    assert folded == {"fine"}
    assert set(state.emotions) == {"sadness"}
    assert "joy" not in state.emotions
    assert any("unexpected RuntimeError" in r.getMessage() for r in caplog.records)


def test_replaying_the_same_window_twice_into_fresh_states_gives_the_same_state() -> None:
    rows = [row(D[2], emotions={"joy": 4.0}, mid="a"), row(D[3], accessed=D[5], emotions={"grief": 2.0}, mid="b")]
    first, _ = _run(rows, changes=[change(D[4])], stop=D[8])
    second, _ = _run(rows, changes=[change(D[4])], stop=D[8])
    assert engine_files.state_to_json(_stamped(first)) == engine_files.state_to_json(_stamped(second))


# --- catch_up -------------------------------------------------------------------------------------------


def _persona(tmp_path: Path, *rows: Memory) -> Path:
    return make_persona(tmp_path / "persona", list(rows))


def _catch(persona: Path, state: EngineState, now=None, queued=(), modifiers=None):
    return catch_up(
        persona,
        state,
        D[10] if now is None else now,
        queued=list(queued),
        emotions_of=_emotions_of,
        modifier_of=modifiers or _mods(),
    )


def test_catch_up_replays_the_rows_since_saved_at_sets_position_zero_returns_the_folded_ids_and_writes_nothing(
    tmp_path: Path,
) -> None:
    before_saved = row(D[1], mid="before")
    after_saved = row(D[4], emotions={"grief": 3.0}, mid="after")
    persona = _persona(tmp_path, before_saved, after_saved)
    state = simple_state(saved_at=D[3], queue_line=7)
    files_before = tree(persona / "state") if (persona / "state").exists() else {}
    folded = _catch(persona, state)
    assert folded == {"after"}
    assert state.queue_line == 0
    assert state.emotions["grief"].short.avg == 3.0
    assert state.emotions["joy"].short.avg == 4.0  # the saved entry is untouched; the earlier row was not replayed
    assert state.saved_at == D[3]  # the caller stamps saved_at and reset_at, not the catch-up
    assert (tree(persona / "state") if (persona / "state").exists() else {}) == files_before


def test_catch_up_stops_accesses_at_the_oldest_queued_line(tmp_path: Path) -> None:
    opened_early = row(D[1], accessed=D[4], mid="early")
    opened_late = row(D[1], accessed=D[7], mid="late")
    persona = _persona(tmp_path, opened_early, opened_late)
    state = simple_state(saved_at=D[3])
    folded = _catch(persona, state, queued=[change(D[6]), change(D[8])])
    assert folded == {"early"}  # the access at D7 is at or after the oldest queued line (D6): its queued showing folds it


def test_catch_up_with_no_queued_line_replays_accesses_up_to_now(tmp_path: Path) -> None:
    persona = _persona(tmp_path, row(D[1], accessed=D[9], mid="late"))
    assert _catch(persona, simple_state(saved_at=D[3])) == {"late"}


def test_a_saved_at_ahead_of_now_is_held_to_now(tmp_path: Path) -> None:
    now = D[5]
    persona = _persona(tmp_path, row(now, mid="at-now"), row(now - timedelta(hours=1), mid="before-now"))
    folded = _catch(persona, simple_state(saved_at=now + timedelta(days=1)), now=now)
    assert folded == {"at-now"}


def test_a_backup_exactly_thirty_days_old_replays_and_a_day_older_raises(tmp_path: Path) -> None:
    persona = _persona(tmp_path, row(D[10] - timedelta(days=1), mid="a"))
    limit = timedelta(days=dev_constants.EMOTION_LONG_HALF_LIFE_D)
    assert _catch(persona, simple_state(saved_at=D[10] - limit)) == {"a"}
    with pytest.raises(CatchUpError):
        _catch(persona, simple_state(saved_at=D[10] - limit - timedelta(seconds=1)))


def test_a_missing_saved_at_raises(tmp_path: Path) -> None:
    persona = _persona(tmp_path)
    state = simple_state(saved_at=D[3])
    state.saved_at = None
    with pytest.raises(CatchUpError):
        _catch(persona, state)


def test_a_missing_database_raises_and_no_database_is_created(tmp_path: Path) -> None:
    persona = tmp_path / "bare"
    persona.mkdir()
    with pytest.raises(CatchUpError):
        _catch(persona, simple_state(saved_at=D[3]))
    assert not (persona / "memories.db").exists()


def test_the_change_log_is_read_through_read_changes_and_a_line_it_cannot_load_does_not_fail_the_catch_up(
    tmp_path: Path,
) -> None:
    """R5 (Planning Q12): `read_changes` skips a line the JSON parser cannot load (#646); the catch-up completes."""
    persona = _persona(tmp_path)
    folder = engine_files.changes_dir(persona)
    folder.mkdir(parents=True)
    good = json.dumps(engine_files.event_to_json(change(D[5], {"joy": 2.0})))
    unloadable = '{"kind": "change", "at": "2026-05-06T12:00:00Z", "emotions": {"joy": ' + "9" * 5000 + "}}"
    (folder / f"{local_day(5).isoformat()}.jsonl").write_text(good + "\n" + unloadable + "\n", encoding="utf-8")
    state = simple_state(saved_at=D[3])
    _catch(persona, state)
    assert state.emotions["joy"].short.avg > 4.0  # the good change (+2 on a stored 4) folded; the other line was skipped


def test_a_failing_change_log_read_propagates_and_the_store_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path, row(D[5], mid="a"))
    monkeypatch.setattr(engine_files, "read_changes", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    closed: list[int] = []
    real_close = MemoryStore.close
    monkeypatch.setattr(MemoryStore, "close", lambda self: (closed.append(1), real_close(self))[1])
    with pytest.raises(OSError):
        _catch(persona, simple_state(saved_at=D[3]))
    assert closed == [1]


def test_a_failing_window_read_propagates_and_the_store_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path, row(D[5], mid="a"))

    def boom(self, since):  # noqa: ANN001
        raise RuntimeError("a database fault")

    monkeypatch.setattr(MemoryStore, "list_engine_window", boom)
    closed: list[int] = []
    real_close = MemoryStore.close
    monkeypatch.setattr(MemoryStore, "close", lambda self: (closed.append(1), real_close(self))[1])
    with pytest.raises(RuntimeError):
        _catch(persona, simple_state(saved_at=D[3]))
    assert closed == [1]


def test_the_queue_events_are_read_never_changed_by_the_catch_up(tmp_path: Path) -> None:
    persona = _persona(tmp_path, row(D[5], mid="a"))
    queued = [change(D[6])]
    _catch(persona, simple_state(saved_at=D[3]), queued=queued)
    assert queued == [change(D[6])]
    assert queue_events(persona) == []


def test_save_roundtrip_of_a_caught_up_state_is_valid(tmp_path: Path) -> None:
    persona = _persona(tmp_path, row(D[5], emotions={"joy": 5.0, "grief": 1.0}, mid="a"))
    state = simple_state(saved_at=D[3])
    _catch(persona, state)
    state.saved_at = D[10]
    save(persona, state)  # a caught-up state passes the writer's strict serialisation
    assert engine_files.read_state(persona).problem is None
