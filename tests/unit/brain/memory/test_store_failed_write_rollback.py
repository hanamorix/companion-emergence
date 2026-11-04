"""A failed single-statement write on MemoryStore leaves no transaction open (Engine 3, #578, BC1).

Python's sqlite3 opens a transaction before a write. When the write raises, the connection stays inside that transaction unless the method ends it, and
the next writer's commit would take the failed call's earlier statements with it. `create`, `update`, `update_emotions_batch` and `deactivate` each
roll back the transaction they opened. The failure is forced with a trigger that aborts the write on one row, through the store's own connection.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from brain.memory.store import Memory, MemoryStore


def _mem(content: str, *, emotions: dict[str, float] | None = None) -> Memory:
    return Memory.create_new(content=content, memory_type="conversation", domain="us", emotions=emotions or {"love": 4.0})


def _fail_on(store: MemoryStore, event: str, row_id: str) -> None:
    """Make the next INSERT or UPDATE that touches `row_id` abort."""
    store._conn.execute(
        f"CREATE TEMP TRIGGER forced_failure BEFORE {event} ON memories WHEN NEW.id = '{row_id}' BEGIN SELECT RAISE(ABORT, 'forced failure'); END"
    )


def _stop_failing(store: MemoryStore) -> None:
    store._conn.execute("DROP TRIGGER forced_failure")


def _durable(path: Path, row_id: str) -> sqlite3.Row | None:
    """The row as a second, read-only connection sees it: only what has been committed."""
    with closing(sqlite3.connect(path, timeout=0.2)) as other:
        other.row_factory = sqlite3.Row
        return other.execute("SELECT * FROM memories WHERE id = ?", (row_id,)).fetchone()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def test_a_failed_create_leaves_no_open_transaction_and_the_next_write_succeeds(store: MemoryStore) -> None:
    failing = _mem("never stored")
    _fail_on(store, "INSERT", failing.id)
    with pytest.raises(sqlite3.DatabaseError, match="forced failure"):
        store.create(failing)
    assert not store._conn.in_transaction
    _stop_failing(store)
    nxt = _mem("stored next")
    assert store.create(nxt) == nxt.id
    assert _durable(store.db_path, nxt.id) is not None
    assert _durable(store.db_path, failing.id) is None


def test_a_failed_update_leaves_no_open_transaction_and_a_callers_own_is_left_alone(store: MemoryStore) -> None:
    a, b = _mem("a"), _mem("b")
    store.create(a)
    store.create(b)
    _fail_on(store, "UPDATE", b.id)
    with pytest.raises(sqlite3.DatabaseError, match="forced failure"):
        store.update(b.id, importance=9.0)
    assert not store._conn.in_transaction
    store.update(a.id, importance=7.0)  # the next write on the same connection
    row_a, row_b = _durable(store.db_path, a.id), _durable(store.db_path, b.id)
    assert row_a is not None and row_a["importance"] == 7.0
    assert row_b is not None and row_b["importance"] == b.importance
    # a transaction the caller already had open is not the failed call's to end
    store._conn.execute("BEGIN")
    store._conn.execute("UPDATE memories SET importance = 3.0 WHERE id = ?", (a.id,))
    with pytest.raises(sqlite3.DatabaseError, match="forced failure"):
        store.update(b.id, importance=9.0)
    assert store._conn.in_transaction
    store._conn.commit()
    row_a = _durable(store.db_path, a.id)
    assert row_a is not None and row_a["importance"] == 3.0


def test_a_failed_batch_leaves_no_open_transaction_and_its_earlier_rows_do_not_become_durable(store: MemoryStore) -> None:
    a, b = _mem("a", emotions={"love": 4.0}), _mem("b", emotions={"love": 4.0})
    store.create(a)
    store.create(b)
    _fail_on(store, "UPDATE", b.id)
    with pytest.raises(sqlite3.DatabaseError, match="forced failure"):
        store.update_emotions_batch([(a.id, {"love": 1.0}), (b.id, {"love": 1.0})])
    assert not store._conn.in_transaction
    _stop_failing(store)
    store.update(b.id, importance=8.0)  # the next write on the same connection commits
    row_a, row_b = _durable(store.db_path, a.id), _durable(store.db_path, b.id)
    assert row_a is not None and json.loads(row_a["emotions_json"]) == {"love": 4.0}  # not the batch's first row
    assert row_b is not None and row_b["importance"] == 8.0


def test_a_failed_deactivate_leaves_no_open_transaction_and_the_next_write_succeeds(store: MemoryStore) -> None:
    a, b = _mem("a"), _mem("b")
    store.create(a)
    store.create(b)
    _fail_on(store, "UPDATE", b.id)
    with pytest.raises(sqlite3.DatabaseError, match="forced failure"):
        store.deactivate(b.id)
    assert not store._conn.in_transaction
    store.deactivate(a.id)  # the next write on the same connection
    row_a, row_b = _durable(store.db_path, a.id), _durable(store.db_path, b.id)
    assert row_a is not None and row_a["active"] == 0
    assert row_b is not None and row_b["active"] == 1
