"""test_grief_touch_lookup.py — MemoryStore.create_grief_touch_unless_recent (the grief-touch race, #342).

The method checks for a recent touch of one referent and inserts a new one as one step: a read-only look first (a debounced
turn takes no write lock), then one write transaction that repeats the check before it inserts. Its window cases are the
three the old read-only lookup had; the rest are the method's own contract.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.memory.store import Memory, MemoryStore

HOURS = 2.0


def _make_grief_event(*, referent_id: object, created_at: datetime, mid: str | None = None) -> Memory:
    m = Memory.create_new(
        content=f"reached for {referent_id} — gone",
        memory_type="grief_event",
        domain="grief",
        emotions={"memory_grief": 3.5},
        metadata={"grief_referent_id": referent_id, "grief_subtype": "recall_touch"},
    )
    object.__setattr__(m, "created_at", created_at)
    if mid is not None:
        object.__setattr__(m, "id", mid)
    return m


def _touch(referent_id: object = "mem-x", *, ago_hours: float = 0.0, mid: str | None = None) -> Memory:
    return _make_grief_event(referent_id=referent_id, created_at=datetime.now(UTC) - timedelta(hours=ago_hours), mid=mid)


def _grief_rows(store: MemoryStore, referent_id: str | None = None) -> list[str]:
    rows = store._conn.execute("SELECT id, metadata_json FROM memories WHERE memory_type = 'grief_event'").fetchall()
    return [
        r["id"]
        for r in rows
        if referent_id is None or json.loads(r["metadata_json"]).get("grief_referent_id") == referent_id
    ]


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def test_a_first_touch_is_inserted_and_its_id_returned(store: MemoryStore) -> None:
    memory = _touch()
    assert store.create_grief_touch_unless_recent(memory, hours=HOURS) == memory.id
    assert _grief_rows(store, "mem-x") == [memory.id]


def test_a_touch_within_the_window_is_not_written_again(store: MemoryStore) -> None:
    first = _touch()
    assert store.create_grief_touch_unless_recent(first, hours=HOURS) == first.id  # the positive control
    assert store.create_grief_touch_unless_recent(_touch(), hours=HOURS) is None
    assert _grief_rows(store, "mem-x") == [first.id]


def test_a_touch_outside_the_window_does_not_block_a_new_one(store: MemoryStore) -> None:
    store.create(_touch(ago_hours=5))
    fresh = _touch()
    assert store.create_grief_touch_unless_recent(fresh, hours=HOURS) == fresh.id
    assert len(_grief_rows(store, "mem-x")) == 2


def test_a_touch_of_another_referent_does_not_block(store: MemoryStore) -> None:
    store.create(_touch("mem-y"))
    mine = _touch("mem-x")
    assert store.create_grief_touch_unless_recent(mine, hours=HOURS) == mine.id
    again = _touch("mem-y")
    assert store.create_grief_touch_unless_recent(again, hours=HOURS) is None  # mem-y is debounced, mem-x is not
    assert len(_grief_rows(store, "mem-y")) == 1 and len(_grief_rows(store, "mem-x")) == 1


def test_the_engine_is_told_once_for_an_insert_and_not_for_a_skip_or_fold_false(
    tmp_path: Path, emotion_engine_stub
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    in_transaction_when_told: list[bool] = []
    record = emotion_engine_stub.on_memory_committed

    def told(memory: Memory) -> None:
        in_transaction_when_told.append(store._conn.in_transaction)
        record(memory)

    emotion_engine_stub.on_memory_committed = told  # type: ignore[method-assign]
    try:
        inserted = _touch("mem-x")
        assert store.create_grief_touch_unless_recent(inserted, hours=HOURS) == inserted.id
        assert [c[0] for c in emotion_engine_stub.recorded_calls] == ["on_memory_committed"]
        assert emotion_engine_stub.recorded_calls[0][1].id == inserted.id
        assert in_transaction_when_told == [False]  # told after the commit, never inside the write transaction
        assert store.create_grief_touch_unless_recent(_touch("mem-x"), hours=HOURS) is None  # a skip tells nobody
        silent = _touch("mem-y")
        assert store.create_grief_touch_unless_recent(silent, hours=HOURS, fold=False) == silent.id
        assert len(emotion_engine_stub.recorded_calls) == 1
    finally:
        store.close()


@pytest.mark.parametrize("referent", [None, 123, ""])
def test_a_memory_with_no_text_referent_is_refused_and_nothing_is_written(store: MemoryStore, referent: object) -> None:
    bad = _make_grief_event(referent_id=referent, created_at=datetime.now(UTC))
    with pytest.raises(ValueError):
        store.create_grief_touch_unless_recent(bad, hours=HOURS)
    assert _grief_rows(store) == [] and not store._conn.in_transaction
    good = _touch()
    assert store.create_grief_touch_unless_recent(good, hours=HOURS) == good.id  # the positive control


def test_a_memory_with_no_referent_key_at_all_is_refused(store: MemoryStore) -> None:
    bare = Memory.create_new(content="x", memory_type="grief_event", domain="grief", emotions={"memory_grief": 1.0})
    with pytest.raises(ValueError):
        store.create_grief_touch_unless_recent(bare, hours=HOURS)
    assert _grief_rows(store) == []
    good = _touch()
    assert store.create_grief_touch_unless_recent(good, hours=HOURS) == good.id  # the positive control


def test_a_duplicate_id_raises_rolls_back_and_leaves_no_transaction_open(store: MemoryStore) -> None:
    taken = _touch("mem-y", mid="dup-id")
    store.create(taken)
    clash = _touch("mem-x", mid="dup-id")
    with pytest.raises(sqlite3.IntegrityError):
        store.create_grief_touch_unless_recent(clash, hours=HOURS)
    assert not store._conn.in_transaction
    assert _grief_rows(store) == ["dup-id"] and _grief_rows(store, "mem-x") == []
    ok = _touch("mem-x")
    assert store.create_grief_touch_unless_recent(ok, hours=HOURS) == ok.id  # the next touch is not refused


def test_a_call_inside_an_open_transaction_is_refused_and_the_transaction_is_left_alone(store: MemoryStore) -> None:
    store._conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(RuntimeError):
        store.create_grief_touch_unless_recent(_touch(), hours=HOURS)
    assert store._conn.in_transaction
    store._conn.rollback()
    assert _grief_rows(store) == []
    mine = _touch()
    assert store.create_grief_touch_unless_recent(mine, hours=HOURS) == mine.id  # the positive control


def _statements(store: MemoryStore) -> list[str]:
    seen: list[str] = []
    store._conn.set_trace_callback(lambda statement: seen.append(statement.strip()))
    return seen


def test_an_insert_is_one_write_transaction_begun_before_the_insert(store: MemoryStore) -> None:
    seen = _statements(store)
    assert store.create_grief_touch_unless_recent(_touch(), hours=HOURS) is not None
    begins = [i for i, s in enumerate(seen) if s.upper().startswith("BEGIN")]
    inserts = [i for i, s in enumerate(seen) if s.startswith("INSERT INTO memories (")]
    commits = [i for i, s in enumerate(seen) if s.upper().startswith("COMMIT")]
    assert [seen[i].upper() for i in begins] == ["BEGIN IMMEDIATE"]
    # sqlite reports the INSERT's text again for the work its triggers do, so the same statement can appear more than once
    assert inserts and len({seen[i] for i in inserts}) == 1 and len(commits) == 1
    assert begins[0] < inserts[0] and inserts[-1] < commits[0]


def test_a_debounced_touch_takes_no_write_lock(store: MemoryStore) -> None:
    store.create(_touch())
    seen = _statements(store)
    assert store.create_grief_touch_unless_recent(_touch(), hours=HOURS) is None
    assert not any(s.upper().startswith("BEGIN") or s.startswith("INSERT INTO memories (") for s in seen)
    assert any("SELECT" in s.upper() for s in seen)  # the look ran
    seen.clear()
    assert store.create_grief_touch_unless_recent(_touch("mem-other"), hours=HOURS) is not None  # the positive control
    assert any(s.upper() == "BEGIN IMMEDIATE" for s in seen) and any(s.startswith("INSERT INTO memories (") for s in seen)


def test_a_failed_create_rolls_back_what_it_opened_and_leaves_a_callers_transaction_alone(store: MemoryStore) -> None:
    first = _touch("mem-y", mid="dup-id")
    store.create(first)
    with pytest.raises(sqlite3.IntegrityError):
        store.create(_touch("mem-x", mid="dup-id"))
    assert not store._conn.in_transaction  # nothing is left open for the next writer
    assert _grief_rows(store) == ["dup-id"]
    later = _touch("mem-x")
    assert store.create_grief_touch_unless_recent(later, hours=HOURS) == later.id  # not refused by a dangling transaction
    # a transaction the caller had open before the call is not the call's to roll back
    store._conn.execute("BEGIN")
    store._conn.execute("UPDATE memories SET importance = 9 WHERE id = ?", (first.id,))
    with pytest.raises(sqlite3.IntegrityError):
        store.create(_touch("mem-z", mid="dup-id"))
    assert store._conn.in_transaction
    store._conn.commit()
    assert store._conn.execute("SELECT importance FROM memories WHERE id = ?", (first.id,)).fetchone()[0] == 9


def test_a_touch_that_appears_between_the_look_and_the_lock_is_skipped_and_the_transaction_is_ended(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def appears_after_the_look(referent_id: str, hours: float) -> bool:
        calls.append(referent_id)
        return len(calls) > 1  # nothing at the look, a touch by the time the lock is held

    monkeypatch.setattr(store, "_recent_grief_touch_exists", appears_after_the_look)
    assert store.create_grief_touch_unless_recent(_touch(), hours=HOURS) is None
    assert len(calls) == 2  # the look and the check inside the write transaction
    assert _grief_rows(store) == []
    assert not store._conn.in_transaction  # the write lock was given up, not held until the next commit
    other = MemoryStore(store.db_path)
    try:
        other._conn.execute("PRAGMA busy_timeout = 200")
        other._conn.execute("BEGIN IMMEDIATE")  # would time out if the first connection still held the write lock
        other._conn.rollback()
    finally:
        other.close()
    # the positive control: when neither check finds a touch the same call inserts
    calls.clear()
    monkeypatch.setattr(store, "_recent_grief_touch_exists", lambda referent_id, hours: calls.append(referent_id) or False)
    mine = _touch()
    assert store.create_grief_touch_unless_recent(mine, hours=HOURS) == mine.id
    assert len(calls) == 2 and _grief_rows(store) == [mine.id]


def test_an_in_memory_store_takes_the_same_one_step_path() -> None:
    memory_store = MemoryStore(":memory:")
    try:
        first = _touch()
        assert memory_store.create_grief_touch_unless_recent(first, hours=HOURS) == first.id
        assert memory_store.create_grief_touch_unless_recent(_touch(), hours=HOURS) is None
        assert _grief_rows(memory_store) == [first.id] and not memory_store._conn.in_transaction
    finally:
        memory_store.close()
