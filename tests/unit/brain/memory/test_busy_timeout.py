"""memories.db busy timeout — C20, C33 (RAM-spike-fix INC-1, S48/S58).

Fail-first proof (ST1.5f) + the new behavior, both against REAL sqlite3
connections over a real file, so this is an execution check, not an
inspection of the pragma string. `test_short_timeout_fails_...` proves the
busy_timeout MECHANISM really does raise "database is locked" when a
connection's own timeout is shorter than a transaction held against it — the
O16 failure mode (2-plan.md §4.2), demonstrated with a SHORT explicit
timeout rather than the literal pre-change 5s value (see "Why a separate
process + a short/long pair" below). `test_new_30s_timeout_waits_...` proves
`MemoryStore` (which now sources `dev_constants.MEMORIES_DB_BUSY_TIMEOUT_S` =
30s at both the `sqlite3.connect` timeout and the `PRAGMA busy_timeout`)
waits out a hold instead of failing, and that it actually WAITED (elapsed
tracks the hold), not merely that no exception happened to surface.

Why a separate PROCESS + a short/long pair, not literal 5s/30s in one
process (CI fix, 2026-09-26): the original version held the write
transaction on a background THREAD in the same process for 6.0s against a
literal 5.0s "old" timeout — a 1.0s margin. That passed on Linux/Windows CI
but failed on macOS (`DID NOT RAISE OperationalError`): under CI scheduling
jitter, the gap between the holder thread actually acquiring its lock and
the waiter issuing its statement (or the holder's own remaining hold time by
the time the waiter got CPU) could eat most of that 1.0s margin. The fix
widens the margin by an order of magnitude (an explicit 0.5s timeout against
a 3.0s hold — a 2.5s margin) and moves the lock holder into its own OS
PROCESS (a `python -c` subprocess), synchronized via a ready-file the
subprocess writes only AFTER it has actually acquired the write transaction
— removing any same-process thread-scheduling/GIL ambiguity about when the
lock is really held. The mechanism under test (sqlite's busy_timeout) is
identical; only the timing margin and the isolation are different, and the
oracle still fails against a too-short timeout and passes against
`MemoryStore`'s real (30s) one.

C33 (build-order check — the 2-plan.md §4.2 report exists with file:line for
every memories.db connect site, written before this file's timeout change)
is satisfied by that document itself + this run's decisions.md gate log, not
by an executable check here; noted for the harness table.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from brain.memory.store import MemoryStore

# A short EXPLICIT timeout standing in for "any timeout shorter than the
# hold" (the mechanism under test, not the literal pre-change 5.0s value —
# see the module docstring) and a hold several times longer, so CI
# scheduling jitter of even a second or two cannot close the gap.
_SHORT_TIMEOUT_S = 0.5
_HOLD_SECONDS = 3.0
_READY_POLL_S = 0.05
_READY_WAIT_TIMEOUT_S = 15.0

_HOLDER_SCRIPT = """
import sqlite3, sys, time
db_path, ready_path, hold_seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
conn = sqlite3.connect(db_path, timeout=30.0)
conn.execute("PRAGMA busy_timeout = 30000")
conn.execute("BEGIN IMMEDIATE")
conn.execute("UPDATE _lock_probe SET x = x + 1")
# Write the ready-file only AFTER the write transaction is genuinely open —
# the parent process treats its existence as proof the lock is held.
with open(ready_path, "w") as f:
    f.write("1")
time.sleep(hold_seconds)
conn.commit()
conn.close()
"""


def _make_probe_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("CREATE TABLE _lock_probe (x INTEGER)")
    conn.execute("INSERT INTO _lock_probe VALUES (0)")
    conn.commit()
    conn.close()


def _start_holder(db_path: str, ready_path: Path, hold_seconds: float) -> subprocess.Popen:
    """Launch a genuinely separate OS process that holds a write
    transaction on `db_path` for `hold_seconds`, once it has actually
    acquired it (signaled by creating `ready_path`)."""
    return subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, db_path, str(ready_path), str(hold_seconds)],
    )


def _wait_for_ready(ready_path: Path, holder: subprocess.Popen) -> None:
    deadline = time.monotonic() + _READY_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        if ready_path.exists():
            return
        assert holder.poll() is None, (
            f"holder process exited early (returncode {holder.returncode}) "
            "before acquiring its write transaction"
        )
        time.sleep(_READY_POLL_S)
    raise AssertionError("holder process never signaled it acquired its write transaction")


def test_short_timeout_fails_against_a_longer_held_write_txn(tmp_path) -> None:
    """Fail-first oracle: a connection whose own busy_timeout is SHORTER
    than a transaction held against it by another process must raise
    "database is locked" — the mechanism the 30s production timeout exists
    to avoid tripping on legitimate long writers (clustering, a heartbeat
    decay batch)."""
    db_path = str(tmp_path / "memories.db")
    ready_path = tmp_path / "holder.ready"
    MemoryStore(db_path).close()  # creates the file + schema + WAL mode
    _make_probe_table(db_path)

    holder = _start_holder(db_path, ready_path, _HOLD_SECONDS)
    try:
        _wait_for_ready(ready_path, holder)

        waiter = sqlite3.connect(db_path, timeout=_SHORT_TIMEOUT_S)
        waiter.execute(f"PRAGMA busy_timeout = {int(_SHORT_TIMEOUT_S * 1000)}")
        start = time.monotonic()
        try:
            with pytest.raises(sqlite3.OperationalError, match="(?i)locked"):
                waiter.execute("UPDATE _lock_probe SET x = 2")
        finally:
            elapsed = time.monotonic() - start
            waiter.close()
        # Sanity bound: it must have actually waited close to its own
        # timeout (not failed instantly for an unrelated reason, and not
        # suspiciously slow either).
        assert elapsed < _HOLD_SECONDS, (
            f"waiter took {elapsed:.2f}s — longer than the full hold; "
            "something other than busy_timeout may be at play"
        )
    finally:
        returncode = holder.wait(timeout=_HOLD_SECONDS + 10.0)
        assert returncode == 0, f"holder process exited with {returncode}"


def test_new_30s_timeout_waits_out_the_same_hold(tmp_path) -> None:
    db_path = str(tmp_path / "memories.db")
    ready_path = tmp_path / "holder.ready"
    store = MemoryStore(db_path)
    _make_probe_table(db_path)

    holder = _start_holder(db_path, ready_path, _HOLD_SECONDS)
    try:
        _wait_for_ready(ready_path, holder)

        start = time.monotonic()
        store.bump_recall("nonexistent-memory-id-for-this-test", 1.0)  # must NOT raise
        elapsed = time.monotonic() - start
    finally:
        returncode = holder.wait(timeout=_HOLD_SECONDS + 10.0)
        assert returncode == 0, f"holder process exited with {returncode}"
        store.close()

    assert elapsed >= _HOLD_SECONDS * 0.5, (
        f"bump_recall returned in {elapsed:.2f}s, too fast to have actually "
        "waited behind the held transaction — the 30s busy_timeout may not "
        "be reaching this connection"
    )
