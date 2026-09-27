"""Once-per-process FTS5 health check (INC-3 — spec §6c, S60/S61/S62/S74).

Criteria covered: C37 (AC22, S60, S61, S62, S74) parts (a)-(f); C34 (S61,
S48/S58 side effect) — a per-turn `MemoryStore(integrity_check=False)` open
no longer takes any transaction for the retired `_boot_fts_backstop` and
returns fast even while another connection holds a long write transaction
open, because that check no longer runs from `MemoryStore.__init__` at all.

Each test states the fail-first oracle it proves against the pre-change
`_boot_fts_backstop` (store.py, ran unconditionally on every open, and
treated a lock timeout the same as real corruption).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from brain.memory import db_health
from brain.memory.store import Memory, MemoryStore

F_BOB20K = Path(
    "/home/zero/Desktop/companion-emergence/.claude/worktrees/dragonfly-ram-spike/"
    "changes/dragonfly-ram-spike/persona/Bob/memories.db"
)


def _mem(content: str) -> Memory:
    return Memory.create_new(content=content, memory_type="event", domain="d")


@pytest.fixture(autouse=True)
def _reset_db_health_state():
    """Every test starts with a clean once-per-process checked-paths set —
    otherwise a path resolved by an earlier test (or a colliding tmp_path)
    would silently short-circuit this test's own check."""
    db_health._reset_for_tests()
    yield
    db_health._reset_for_tests()


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    return tmp_path


def _seeded_db(tmp_path: Path, n: int = 5) -> Path:
    db = tmp_path / "memories.db"
    store = MemoryStore(db)
    for i in range(n):
        store.create(_mem(f"apple {i}"))
    store.close()
    return db


def _health_log_lines(tmp_home: Path, persona: str) -> list[dict]:
    log_path = tmp_home / "logs" / f"db-health-{persona}.jsonl"
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# C37(a) — runs at most once per process; non-bridge processes run it zero
# times because they never call it (only `MemoryStore()` construction, never
# `run_fts_health_check_once`).
# ---------------------------------------------------------------------------


def test_c37a_check_runs_at_most_once_across_many_opens(tmp_home, monkeypatch):
    db = _seeded_db(tmp_home)
    calls: list[None] = []
    orig = db_health._classify_and_run

    def counting(conn):
        calls.append(None)
        return orig(conn)

    monkeypatch.setattr(db_health, "_classify_and_run", counting)

    # The lifespan's one explicit call...
    db_health.run_fts_health_check_once(db)
    # ...plus >=20 per-turn/supervisor MemoryStore opens against the same
    # file, none of which run the check themselves (that's the INC-3 fix)...
    for _ in range(20):
        s = MemoryStore(db, integrity_check=False)
        s.close()
    # ...plus a second explicit call (e.g. a stray supervisor invocation) —
    # still a no-op for the same resolved path.
    db_health.run_fts_health_check_once(db)

    # Fail-first: pre-change, the retired `_boot_fts_backstop` ran on EVERY
    # one of the 20 `MemoryStore()` opens above — this assertion fails
    # against that code (>=1 call from opens alone, before even counting the
    # two explicit calls).
    assert len(calls) == 1


def test_c37a_non_bridge_opens_never_run_the_check(tmp_home, monkeypatch):
    """MCP-child / CLI processes only ever construct `MemoryStore()` — they
    never import or call `run_fts_health_check_once`. Proving the check
    never fires from a bare `MemoryStore()` open (with no explicit call at
    all) proves that architectural gap (S74)."""
    db = _seeded_db(tmp_home)
    calls: list[None] = []
    monkeypatch.setattr(
        db_health, "_classify_and_run", lambda conn: (calls.append(None), ("healthy", None))[1]
    )

    for _ in range(20):
        s = MemoryStore(db, integrity_check=False)
        s.close()
    for _ in range(5):
        s = MemoryStore(db)  # even the integrity_check=True (CLI-style) path
        s.close()

    assert calls == []


def test_c37f_two_different_paths_each_checked_once(tmp_home):
    (tmp_home / "a").mkdir()
    (tmp_home / "b").mkdir()
    db_a = _seeded_db(tmp_home / "a")
    db_b = _seeded_db(tmp_home / "b")

    for db in (db_a, db_b, db_a, db_b):
        db_health.run_fts_health_check_once(db)

    assert db_a.resolve() in db_health._checked_paths
    assert db_b.resolve() in db_health._checked_paths
    assert len(db_health._checked_paths) == 2


# ---------------------------------------------------------------------------
# C37(b) — a forced lock timeout classifies as could_not_check, never a
# rebuild attempt, exactly one log line, and the check's own connection ends
# with no open transaction.
# ---------------------------------------------------------------------------


def test_c37b_lock_timeout_is_could_not_check_no_rebuild(tmp_home, monkeypatch):
    persona = tmp_home.name
    db = _seeded_db(tmp_home)

    # Hold a write transaction open on a second connection (WAL: one writer
    # at a time) so the health check's own connection contends for the lock.
    blocker = sqlite3.connect(str(db))
    blocker.execute("PRAGMA journal_mode = WAL")
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute("CREATE TABLE IF NOT EXISTS _lock_holder(x)")
    try:
        monkeypatch.setattr(db_health.dev_constants, "MEMORIES_DB_BUSY_TIMEOUT_S", 0.3)
        t0 = time.time()
        db_health.run_fts_health_check_once(db)
        elapsed = time.time() - t0
    finally:
        blocker.rollback()
        blocker.close()

    # Fail-first: pre-change, a lock timeout was caught by the same
    # `except sqlite3.DatabaseError` as real corruption and treated as
    # "rebuild the index" (S60) — this asserts the opposite outcome.
    lines = _health_log_lines(tmp_home, persona)
    assert len(lines) == 1
    assert lines[0]["result"] == "could_not_check"
    assert "lock" in lines[0]["error"].lower() or "busy" in lines[0]["error"].lower()
    # Blocked for roughly the (patched) busy timeout, not instantly, and not
    # forever.
    assert 0.2 <= elapsed <= 5.0


def test_c37b_classify_leaves_no_open_transaction_on_lock_timeout(tmp_path, monkeypatch):
    db = _seeded_db(tmp_path)
    blocker = sqlite3.connect(str(db))
    blocker.execute("PRAGMA journal_mode = WAL")
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute("CREATE TABLE IF NOT EXISTS _lock_holder(x)")
    try:
        checker = sqlite3.connect(str(db), timeout=0.3)
        checker.execute("PRAGMA busy_timeout = 300")
        result, error = db_health._classify_and_run(checker)
        assert result == "could_not_check"
        assert not checker.in_transaction
        checker.close()
    finally:
        blocker.rollback()
        blocker.close()


# ---------------------------------------------------------------------------
# C37(c) — a forced count mismatch rebuilds, logs `count_mismatch`, and the
# NEXT process start (a fresh `_reset_for_tests`, matching a new process)
# finds it healthy and logs nothing further.
# ---------------------------------------------------------------------------


def test_c37c_count_mismatch_rebuilds_and_logs_then_next_start_is_healthy(tmp_home):
    persona = tmp_home.name
    db = _seeded_db(tmp_home, n=5)

    conn = sqlite3.connect(str(db))
    conn.execute(
        "DELETE FROM memories_fts_docsize WHERE rowid = (SELECT MIN(rowid) FROM memories_fts_docsize)"
    )
    conn.commit()
    conn.close()

    db_health.run_fts_health_check_once(db)
    lines = _health_log_lines(tmp_home, persona)
    assert len(lines) == 1
    assert lines[0]["result"] == "count_mismatch"

    conn2 = sqlite3.connect(str(db))
    assert conn2.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == conn2.execute(
        "SELECT COUNT(*) FROM memories_fts_docsize"
    ).fetchone()[0]
    conn2.close()

    # Next process start: fresh once-per-process state.
    db_health._reset_for_tests()
    db_health.run_fts_health_check_once(db)
    lines_after = _health_log_lines(tmp_home, persona)
    assert len(lines_after) == 1, "healthy re-check must not append another line"


# ---------------------------------------------------------------------------
# C37(d) — real, reproduced FTS corruption (not a mocked exception): a
# damaged shadow table classifies `damaged`, rebuilds, and logs.
# ---------------------------------------------------------------------------


def test_c37d_damaged_fts_shadow_table_rebuilds_and_logs(tmp_home):
    persona = tmp_home.name
    db = _seeded_db(tmp_home, n=10)

    # Corrupt the FTS5 shadow *content* (not its table structure) so the
    # integrity-check genuinely raises a non-"malformed" DatabaseError AND
    # `rebuild` (which just re-scans `memories`, not the old shadow rows) can
    # actually repair it — a real, reproduced FTS5 corruption, not a mock.
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories_fts_data WHERE rowid IN "
                 "(SELECT rowid FROM memories_fts_data LIMIT 2)")
    conn.commit()
    conn.close()

    db_health.run_fts_health_check_once(db)

    lines = _health_log_lines(tmp_home, persona)
    assert len(lines) == 1
    assert lines[0]["result"] == "damaged"
    assert lines[0]["error"]
    assert "malformed" not in lines[0]["error"].lower()

    # Rebuild actually restored the index — re-open and confirm FTS is usable
    # again (mirrors test_fts_sync.py's C2 no-memory-lost assertion).
    store = MemoryStore(db)
    try:
        rows = store._conn.execute(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'apple'"
        ).fetchone()[0]
        assert rows == 10
    finally:
        store.close()


def test_c37d_malformed_disk_image_logs_and_reports_rebuild_failure(tmp_home):
    """A truncated file is unrecoverable — `rebuild` cannot fix a malformed
    disk image, so this checks the OTHER documented branch: a rebuild
    attempt that itself fails logs `rebuild_failed` (db_health.py), in
    addition to the original `malformed` classification."""
    persona = tmp_home.name
    db = _seeded_db(tmp_home, n=10)

    size = db.stat().st_size
    with db.open("r+b") as fh:
        fh.truncate(size // 2)

    db_health.run_fts_health_check_once(db)

    lines = _health_log_lines(tmp_home, persona)
    results = [line["result"] for line in lines]
    assert "malformed" in results
    malformed_line = next(line for line in lines if line["result"] == "malformed")
    assert "malformed" in malformed_line["error"].lower()
    assert "rebuild_failed" in results


# ---------------------------------------------------------------------------
# no_schema vs. legacy-persona-missing-FTS-table (code-red-team pass 1, M2):
# a brand-new persona (no `memories` table either) is a true no-op; a legacy
# persona whose `memories` table already has rows but predates the FTS table
# must be backfilled now, not silently skipped for a whole process lifetime.
# ---------------------------------------------------------------------------


def test_no_schema_is_a_true_noop_for_a_brand_new_persona(tmp_home):
    persona = tmp_home.name
    db = tmp_home / "memories.db"  # file doesn't exist yet at all

    db_health.run_fts_health_check_once(db)

    assert db.exists()  # sqlite3.connect creates the (empty) file
    assert _health_log_lines(tmp_home, persona) == []
    conn = sqlite3.connect(str(db))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master").fetchall()}
    conn.close()
    assert "memories" not in tables, "a true no-op must not create any schema"


def test_legacy_persona_missing_fts_table_is_backfilled_not_skipped(tmp_home):
    """Reproduces a persona whose `memories.db` predates the FTS5 table:
    `memories` already has rows, `memories_fts` (and its shadow tables /
    triggers) don't exist. Fail-first: before this fix, this hit the same
    'no such table' branch as a brand-new persona and was classified
    `no_schema` — silently skipped, not logged, and the once-per-process
    flag then prevented any retry for the rest of this bridge's lifetime."""
    persona = tmp_home.name
    db = _seeded_db(tmp_home, n=3)

    conn = sqlite3.connect(str(db))
    conn.execute("DROP TRIGGER IF EXISTS memories_fts_ai")
    conn.execute("DROP TRIGGER IF EXISTS memories_fts_ad")
    conn.execute("DROP TRIGGER IF EXISTS memories_fts_au")
    conn.execute("DROP TABLE IF EXISTS memories_vocab")
    conn.execute("DROP TABLE IF EXISTS memories_fts")
    conn.commit()
    conn.close()

    db_health.run_fts_health_check_once(db)

    lines = _health_log_lines(tmp_home, persona)
    assert len(lines) == 1
    assert lines[0]["result"] == "fts_missing_backfilled"

    conn2 = sqlite3.connect(str(db))
    try:
        rows = conn2.execute(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'apple'"
        ).fetchone()[0]
        assert rows == 3, "pre-existing rows must be backfilled into the rebuilt FTS index"
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# C37(e) — healthy → no log line.
# ---------------------------------------------------------------------------


def test_c37e_healthy_logs_nothing(tmp_home):
    persona = tmp_home.name
    db = _seeded_db(tmp_home, n=5)
    db_health.run_fts_health_check_once(db)
    assert _health_log_lines(tmp_home, persona) == []


# ---------------------------------------------------------------------------
# C34 — with the once-per-process check already done, a per-turn
# `MemoryStore(integrity_check=False)` open takes no transaction of its own
# and returns fast even while another connection holds a long write
# transaction open (the retired `_boot_fts_backstop` used to contend for
# that same write lock on every open).
# ---------------------------------------------------------------------------


def test_c34_per_turn_open_takes_no_transaction_while_a_writer_holds_one(tmp_home):
    db = _seeded_db(tmp_home)
    db_health.run_fts_health_check_once(db)  # "already done" precondition

    holder = sqlite3.connect(str(db))
    holder.execute("PRAGMA journal_mode = WAL")
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("CREATE TABLE IF NOT EXISTS _writer_holds_this(x)")
    try:
        t0 = time.time()
        s = MemoryStore(db, integrity_check=False)
        elapsed = time.time() - t0
        try:
            # No BEGIN/INSERT for the retired FTS check: the store's own
            # constructor-time work (schema DDL) already committed before
            # returning, so it should not be sitting in an open transaction.
            assert not s._conn.in_transaction
        finally:
            s.close()
    finally:
        holder.rollback()
        holder.close()

    # Fail-first: pre-change, `_boot_fts_backstop` ran its integrity-check
    # INSERT on this very open and blocked behind `holder`'s write lock for
    # up to the busy timeout (30 s) — this open must return fast instead.
    assert elapsed <= 1.0


@pytest.mark.skipif(not F_BOB20K.exists(), reason="F-bob20k fixture not present (local Linux only)")
def test_c34_per_turn_open_fast_during_a_real_clustering_write(tmp_path):
    """Same property as above, proven against a real (not synthetic-tiny)
    long write transaction: clustering's `set_cluster_memberships` on the
    Phoebe-sized F-bob20k fixture (2-plan §4.3 measured 6.7 s locally)."""
    db = tmp_path / "memories.db"
    shutil.copy(F_BOB20K, db)
    db_health.run_fts_health_check_once(db)  # "already done" precondition

    probe = MemoryStore(db, integrity_check=False)
    try:
        ids = [row[0] for row in probe._conn.execute("SELECT id FROM memories LIMIT 20000").fetchall()]
    finally:
        probe.close()
    import numpy as np

    memberships = {mid: i % 8 for i, mid in enumerate(ids)}
    centroids = np.random.default_rng(0).random((8, 1024)).astype(np.float32)

    done = threading.Event()
    started = threading.Event()

    def _run_clustering():
        # A fresh MemoryStore/connection — sqlite3 connections cannot cross
        # threads, so the writer opens its own inside the thread it runs on.
        writer_store = MemoryStore(db, integrity_check=False)
        try:
            started.set()
            writer_store.set_cluster_memberships(memberships, centroids, model_id="test-model")
        finally:
            writer_store.close()
        done.set()

    t = threading.Thread(target=_run_clustering)
    t.start()
    started.wait(timeout=5)
    time.sleep(0.05)  # let the write actually begin

    t0 = time.time()
    reader = MemoryStore(db, integrity_check=False)
    elapsed = time.time() - t0
    reader.close()

    t.join(timeout=30)

    assert done.is_set(), "clustering write should have completed within the join timeout"
    assert elapsed <= 1.0
