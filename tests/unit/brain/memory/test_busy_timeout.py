"""memories.db busy timeout — C20, C33 (RAM-spike-fix INC-1, S48/S58).

Fail-first proof (ST1.5f) + the new behavior, both against REAL sqlite3
connections over a real file, so this is an execution check, not an
inspection of the pragma string. `test_old_5s_timeout_fails_...` proves the
pre-change mechanism (5s) really does raise "database is locked" against a
transaction held longer than 5s — the O16 failure mode (2-plan.md §4.2).
`test_new_30s_timeout_waits_...` proves `MemoryStore` (which now sources
`dev_constants.MEMORIES_DB_BUSY_TIMEOUT_S` = 30s at both the `sqlite3.connect`
timeout and the `PRAGMA busy_timeout`) waits out the same hold instead of
failing, and that it actually WAITED (elapsed tracks the hold), not merely
that no exception happened to surface.

C33 (build-order check — the 2-plan.md §4.2 report exists with file:line for
every memories.db connect site, written before this file's timeout change)
is satisfied by that document itself + this run's decisions.md gate log, not
by an executable check here; noted for the harness table.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from brain.memory.store import MemoryStore

# Longer than the pre-change 5s busy_timeout, comfortably under the new 30s
# one, and close to (but a hair under, for test determinism) the largest
# measured memories.db write transaction (clustering's set_cluster_
# memberships, 6.7s on F-bob20k, 2-plan.md §4.3).
_HOLD_SECONDS = 6.0


def _hold_write_lock(db_path: str, ready: threading.Event, hold_seconds: float) -> None:
    """Open a second, independent connection and hold a write transaction
    for `hold_seconds` — simulates a long in-progress memories.db writer
    (clustering / a heartbeat decay batch) that another connection must
    wait behind. `_lock_probe` must already exist and be committed (WAL
    readers/writers only ever see committed schema) before this runs."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE _lock_probe SET x = x + 1")
    ready.set()
    time.sleep(hold_seconds)
    conn.commit()
    conn.close()


def _make_probe_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("CREATE TABLE _lock_probe (x INTEGER)")
    conn.execute("INSERT INTO _lock_probe VALUES (0)")
    conn.commit()
    conn.close()


def test_old_5s_timeout_fails_against_a_long_held_write_txn(tmp_path) -> None:
    db_path = str(tmp_path / "memories.db")
    MemoryStore(db_path).close()  # creates the file + schema + WAL mode
    _make_probe_table(db_path)

    ready = threading.Event()
    holder = threading.Thread(target=_hold_write_lock, args=(db_path, ready, _HOLD_SECONDS))
    holder.start()
    assert ready.wait(timeout=5.0), "holder never acquired its write transaction"

    waiter = sqlite3.connect(db_path, timeout=5.0)
    waiter.execute("PRAGMA busy_timeout = 5000")
    try:
        with pytest.raises(sqlite3.OperationalError, match="(?i)locked"):
            waiter.execute("UPDATE _lock_probe SET x = 2")
    finally:
        waiter.close()
        holder.join(timeout=_HOLD_SECONDS + 2.0)
        assert not holder.is_alive()


def test_new_30s_timeout_waits_out_the_same_hold(tmp_path) -> None:
    db_path = str(tmp_path / "memories.db")
    store = MemoryStore(db_path)
    _make_probe_table(db_path)

    ready = threading.Event()
    holder = threading.Thread(target=_hold_write_lock, args=(db_path, ready, _HOLD_SECONDS))
    holder.start()
    assert ready.wait(timeout=5.0), "holder never acquired its write transaction"

    start = time.monotonic()
    store.bump_recall("nonexistent-memory-id-for-this-test", 1.0)  # must NOT raise
    elapsed = time.monotonic() - start

    holder.join(timeout=_HOLD_SECONDS + 2.0)
    assert not holder.is_alive()
    store.close()

    assert elapsed >= _HOLD_SECONDS * 0.5, (
        f"bump_recall returned in {elapsed:.2f}s, too fast to have actually "
        "waited behind the held transaction — the 30s busy_timeout may not "
        "be reaching this connection"
    )
