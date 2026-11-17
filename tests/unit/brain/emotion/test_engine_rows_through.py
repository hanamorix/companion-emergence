"""`EngineState.rows_through` and the rowid carrier (HR1 part 2 of #633, Hana's point 1).

`rows_through` is the highest `memories` rowid the engine has seen. It is saved with the state, read back as None unless it is
a whole number of 0 or more, carried by `Memory.rowid` and `Event.rowid` from the store's hooks to the apply step, and moved by
the apply, the first and recovery rebuilds and the catch-up. Nothing here reads it to decide anything: the start check is a
later step.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from brain.emotion import engine, engine_files, engine_rebuild
from brain.emotion.engine_utils import EngineState, Event, day_of, larger_rowid
from brain.memory.store import MemoryStore
from tests.unit.brain.emotion._engine_support import (
    NOW,
    install,
    local_day,
    local_noon,
    make_persona,
    row,
    save,
    save_as_backup,
    simple_state,
    state_of,
)

D = [local_noon(local_day(n)) for n in range(12)]


def _rowid(store: MemoryStore, memory_id: str) -> int:
    found = store._conn.execute("SELECT rowid FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return int(found[0])


def _state(saved_at, **kwargs) -> EngineState:  # noqa: ANN001, ANN003
    """A small readable state saved at `saved_at` (its active day set, as the file validator needs)."""
    return simple_state(saved_at=saved_at, day=day_of(saved_at), **kwargs)


def _round_trip(value: object) -> int | None:
    data = engine_files.state_to_json(EngineState(saved_at=NOW))
    data["rows_through"] = value
    return engine_files.state_from_json(json.loads(json.dumps(data))).rows_through


# --- the field in the state file -------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, 41, 10**12])
def test_a_whole_number_of_zero_or_more_is_saved_and_read_back(value: int) -> None:
    state = EngineState(saved_at=NOW, rows_through=value)
    assert engine_files.state_to_json(state)["rows_through"] == value
    assert _round_trip(value) == value


def test_an_unset_value_is_written_as_null_and_a_missing_key_reads_as_none() -> None:
    data = engine_files.state_to_json(EngineState(saved_at=NOW))
    assert "rows_through" in data and data["rows_through"] is None
    del data["rows_through"]
    assert engine_files.state_from_json(data).rows_through is None  # a file written before the field existed


@pytest.mark.parametrize("value", [True, False, -1, 2.0, 1.5, "7", [7], {"n": 7}, float("nan")])
def test_a_malformed_value_reads_as_none_and_the_state_stays_readable(value: object, tmp_path: Path) -> None:
    data = engine_files.state_to_json(EngineState(saved_at=NOW))
    data["rows_through"] = value
    engine_files.validate_state(json.loads(json.dumps(data)))  # never a reason to refuse the state
    assert _round_trip(value) is None
    path = engine_files.state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = engine_files.read_state(tmp_path)
    assert loaded.state is not None and loaded.state.rows_through is None
    assert loaded.state.saved_at == NOW


def test_larger_rowid_treats_none_as_no_value() -> None:
    assert larger_rowid(None, None) is None
    assert larger_rowid(None, 4) == 4
    assert larger_rowid(4, None) == 4
    assert larger_rowid(4, 9) == 9
    assert larger_rowid(9, 4) == 9


# --- the hooks carry the row's rowid ---------------------------------------------------------------------------


def test_a_counted_create_moves_rows_through_to_that_rows_rowid_and_does_not_change_the_callers_memory(tmp_path: Path) -> None:
    older = row(NOW - timedelta(days=5), mid="older")
    persona = make_persona(tmp_path / "persona", [older])  # rowid 1, written without telling the engine
    install(persona, NOW)
    store = MemoryStore(persona / "memories.db")
    try:
        new = row(NOW - timedelta(hours=2), mid="new")
        store.create(new)
        assert state_of(persona).rows_through == _rowid(store, "new") == 2
        assert new.rowid is None  # the hook was given a copy
    finally:
        store.close()


def test_a_later_smaller_rowid_does_not_lower_rows_through(tmp_path: Path) -> None:
    persona = make_persona(tmp_path / "persona")
    install(persona, NOW)
    save(persona, _state(NOW - timedelta(hours=1)))
    state = state_of(persona)
    state.rows_through = 50
    save(persona, state)
    store = MemoryStore(persona / "memories.db")
    try:
        store.create(row(NOW - timedelta(hours=2), mid="low"))  # rowid 1
    finally:
        store.close()
    assert state_of(persona).rows_through == 50


def test_the_assigned_hook_uses_the_updated_rows_own_rowid_not_the_newest(tmp_path: Path) -> None:
    old = row(NOW - timedelta(days=3), emotions={}, mid="old")  # rowid 1, no feelings yet
    newer = row(NOW - timedelta(days=1), mid="newer")  # rowid 2
    persona = make_persona(tmp_path / "persona", [old, newer])
    install(persona, NOW)
    store = MemoryStore(persona / "memories.db")
    try:
        store.update("old", emotions={"grief": 4.0})
        assert state_of(persona).rows_through == _rowid(store, "old") == 1
    finally:
        store.close()


def test_the_commit_and_assigned_hooks_and_the_grief_touch_hand_the_engine_the_rowid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = row(NOW - timedelta(days=3), emotions={}, mid="old")
    persona = make_persona(tmp_path / "persona", [old])
    eng = install(persona, NOW)
    seen: list[tuple[str, str, int | None]] = []
    monkeypatch.setattr(eng, "on_memory_committed", lambda memory: seen.append(("commit", memory.id, memory.rowid)))
    monkeypatch.setattr(eng, "on_emotions_assigned", lambda memory: seen.append(("assigned", memory.id, memory.rowid)))
    store = MemoryStore(persona / "memories.db")
    try:
        store.create(row(NOW, mid="made"))
        grief = row(NOW, mid="touch", metadata={"grief_referent_id": "gone"})
        assert store.create_grief_touch_unless_recent(grief, hours=1.0) == "touch"
        store.update("old", emotions={"joy": 3.0})
        assert seen == [
            ("commit", "made", _rowid(store, "made")),
            ("commit", "touch", _rowid(store, "touch")),
            ("assigned", "old", _rowid(store, "old")),
        ]
        assert [entry[2] for entry in seen] == [2, 3, 1]
    finally:
        store.close()


def test_a_caller_with_no_row_leaves_rows_through_as_it_was(tmp_path: Path) -> None:
    persona = make_persona(tmp_path / "persona")
    eng = install(persona, NOW)
    state = _state(NOW - timedelta(hours=1))
    state.rows_through = 7
    save(persona, state)
    eng.on_memory_committed(row(NOW - timedelta(minutes=5), mid="no-row"))  # rowid None, like consolidation's candidate
    after = state_of(persona)
    assert after.rows_through == 7
    assert after.emotions["joy"].short.weight > 1.0  # it did fold: only the position is left alone

    fresh = make_persona(tmp_path / "fresh")
    other = install(fresh, NOW)
    other.on_memory_committed(row(NOW - timedelta(minutes=5), mid="no-row-2"))
    assert state_of(fresh).rows_through is None


def test_an_event_the_apply_drops_does_not_move_rows_through(tmp_path: Path) -> None:
    """An event with no feelings does not survive the queue-form round trip, so nothing folds; the all-fail case is the batch test below."""
    persona = make_persona(tmp_path / "persona")
    eng = install(persona, NOW)
    state = _state(NOW - timedelta(hours=1))
    state.rows_through = 3
    save(persona, state)
    eng._apply_direct([Event(kind="commit", at=NOW, memory_id="x", emotions={}, rowid=99)])
    assert state_of(persona).rows_through == 3


def test_in_a_batch_only_the_events_that_folded_move_rows_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path / "persona")
    eng = install(persona, NOW)
    real = engine.fold_event

    def fold(state, event, now):  # noqa: ANN001
        if event.memory_id == "bad":
            raise ValueError("forced")
        return real(state, event, now)

    monkeypatch.setattr(engine, "fold_event", fold)
    start = NOW - timedelta(hours=3)
    good = Event(kind="commit", at=start, memory_id="good", emotions={"joy": 5.0}, rowid=5)
    bad = Event(kind="commit", at=start + timedelta(minutes=1), memory_id="bad", emotions={"joy": 5.0}, rowid=9)
    eng._apply_direct([good, bad])
    assert state_of(persona).rows_through == 5  # the folded event's, not the failed one's


def test_an_event_carries_its_rowid_without_taking_part_in_equality() -> None:
    a = Event(kind="commit", at=NOW, memory_id="m", rowid=3)
    b = Event(kind="commit", at=NOW, memory_id="m")
    assert a == b and a.rowid == 3 and b.rowid is None


# --- the rebuilds and the catch-up -----------------------------------------------------------------------------


def _modifier(_memory) -> float:  # noqa: ANN001
    return 1.0


def _stored(memory) -> dict:  # noqa: ANN001
    return dict(memory._stored_emotions)


def test_the_first_rebuild_sets_rows_through_to_the_highest_rowid_of_its_window(tmp_path: Path) -> None:
    rows = [
        row(NOW - timedelta(days=2), mid="a"),  # rowid 1
        row(NOW - timedelta(days=1), mid="b"),  # rowid 2
        row(NOW - timedelta(days=90), mid="ancient"),  # rowid 3: outside the 30-day window, so not read, not counted
    ]
    persona = make_persona(tmp_path / "persona", rows)
    engine_rebuild.rebuild(persona, "first", NOW, faded_emotions_of=_stored, modifier_of=_modifier)
    assert state_of(persona).rows_through == 2


def test_a_first_rebuild_of_an_empty_window_leaves_rows_through_unset(tmp_path: Path) -> None:
    persona = make_persona(tmp_path / "persona")
    engine_rebuild.rebuild(persona, "first", NOW, faded_emotions_of=_stored, modifier_of=_modifier)
    assert state_of(persona).rows_through is None


@pytest.mark.parametrize(("old", "expected"), [(None, 2), (1, 2), (50, 50)])
def test_a_recovery_rebuild_takes_the_larger_of_the_old_states_value_and_its_windows(tmp_path: Path, old: int | None, expected: int) -> None:
    persona = make_persona(tmp_path / "persona", [row(NOW - timedelta(days=2), mid="a"), row(NOW - timedelta(days=1), mid="b")])
    state = _state(NOW - timedelta(days=1), switch_over_at=NOW - timedelta(days=10))
    state.rows_through = old
    save(persona, state)
    engine_rebuild.rebuild(persona, "recovery", NOW, faded_emotions_of=_stored, modifier_of=_modifier)
    assert state_of(persona).rows_through == expected


def test_catch_up_advances_rows_through_and_never_lowers_it(tmp_path: Path) -> None:
    persona = make_persona(tmp_path / "persona", [row(D[1], mid="before"), row(D[5], mid="x"), row(D[6], mid="y")])
    state = simple_state(saved_at=D[3], day=local_day(3).isoformat())
    engine_rebuild.catch_up(persona, state, D[8], queued=[], emotions_of=_stored, modifier_of=_modifier)
    assert state.rows_through == 3  # "before" (rowid 1) is outside the window; the rows read are x and y, the highest is y's
    higher = simple_state(saved_at=D[3], day=local_day(3).isoformat())
    higher.rows_through = 40
    engine_rebuild.catch_up(persona, higher, D[8], queued=[], emotions_of=_stored, modifier_of=_modifier)
    assert higher.rows_through == 40


def test_the_drains_catch_up_of_a_backup_state_saves_the_advanced_rows_through(tmp_path: Path) -> None:
    persona = make_persona(tmp_path / "persona", [row(D[5], mid="x"), row(D[6], mid="y")])
    save_as_backup(persona, simple_state(saved_at=D[3], day=local_day(3).isoformat(), switch_over_at=D[1]))
    eng = install(persona, D[8])
    eng.note_active(D[8])
    assert state_of(persona).rows_through == 2


def test_a_direct_apply_on_a_backup_state_catches_up_and_saves_the_advanced_rows_through(tmp_path: Path) -> None:
    persona = make_persona(tmp_path / "persona", [row(D[5], mid="x")])
    save_as_backup(persona, simple_state(saved_at=D[3], day=local_day(3).isoformat(), switch_over_at=D[1]))
    install(persona, D[8])
    store = MemoryStore(persona / "memories.db")
    try:
        store.create(row(D[7], mid="z"))  # rowid 2; the catch-up reads it too, so this proves the saved state, not the event path
    finally:
        store.close()
    assert state_of(persona).rows_through == 2


def test_the_window_rowid_read_skips_nothing_the_store_filters(tmp_path: Path) -> None:
    """The highest rowid is taken over the rows the window read returned, through the same reader the rebuild uses."""
    persona = make_persona(tmp_path / "persona", [row(NOW - timedelta(days=1), mid="a")])
    with sqlite3.connect(persona / "memories.db") as conn:
        assert conn.execute("SELECT MAX(rowid) FROM memories").fetchone()[0] == 1
    engine_rebuild.rebuild(persona, "first", NOW, faded_emotions_of=_stored, modifier_of=_modifier)
    assert state_of(persona).rows_through == 1
