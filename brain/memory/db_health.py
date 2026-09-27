"""Once-per-process memories.db FTS5 health check (INC-3, spec §6c, S60-S62, S74).

Before this fix, `MemoryStore.__init__` ran the FTS5 integrity-check +
conditional rebuild backstop on EVERY open (`_boot_fts_backstop`, retired) —
including the ~15 concurrent per-turn/CLI opens against a live bridge's
`memories.db`, each one taking an implicit write transaction for a check that
only ever needs to happen once per process. Two bugs followed from running it
unconditionally: (1) a lock timeout ("database is locked") was caught by the
same `except sqlite3.DatabaseError` as real corruption and treated as
"rebuild the index" (S60); (2) every open paid the write-lock cost, widening
the window in which a per-turn store construction could stall behind a
concurrent writer (O16).

This module runs the check explicitly, at most once per process per resolved
database path, classifies the result, rebuilds the FTS index only on real
corruption, and appends every non-`healthy` result to a persistent,
append-only db-health log in the persona's log directory. The bridge lifespan
(`brain/bridge/server.py`) is the ONLY caller, and calls it once, before the
first `MemoryStore` open for that persona (S74): the per-turn MCP child, CLI
commands, `nell chat --no-bridge` and `cmd_start`'s parent-side recovery never
call this — a problem on one of those paths is caught at the next bridge
start instead.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from brain import dev_constants
from brain.paths import get_log_dir

logger = logging.getLogger(__name__)

_CHECK_NAME = "memories_fts"

# Module-level, lock-guarded set of resolved db paths already checked in this
# process (stage-3 round-2 red-team m2: keyed by `Path(db_path).resolve()` so
# two different on-disk paths are each checked once, and the same path opened
# twice is checked only the first time — C37(a)/(f)).
_lock = threading.Lock()
_checked_paths: set[Path] = set()

# Outcomes that trigger a rebuild attempt (S60): real corruption or a stale
# index, never a lock timeout.
_REBUILD_ON = {"damaged", "malformed", "count_mismatch"}


def _classify_error_message(msg: str) -> str:
    """Map a `DatabaseError` message to a health-check outcome.

    Spec §6c / S60 (the spec, which outranks 2-plan.md's own looser "any
    other DatabaseError" wording — orchestrator ruling 2026-09-27): only a
    genuine corruption signal may trigger a rebuild — the FTS5
    integrity-check itself reporting damage/corruption, or a "database disk
    image is malformed" error. A lock timeout ("database is locked"/"busy")
    is `could_not_check` per S60. Anything ELSE — an environmental or
    unexpected error this check didn't anticipate (permissions, I/O, "unable
    to open database file", a future SQLite error string not seen here) —
    is ALSO `could_not_check`, never corruption: rebuilding the FTS index in
    response to an error that has nothing to do with the FTS index's own
    integrity is the exact over-broad-`except sqlite3.DatabaseError`
    conflation this increment exists to fix, just for a different error
    family than the lock-timeout one S60 names explicitly. `could_not_check`
    is always safe to fall back to: it changes nothing and is retried at the
    next process start (S60)."""
    low = msg.lower()
    if "malformed" in low:
        return "malformed"
    if "corrupt" in low:
        return "damaged"
    return "could_not_check"


def _log_health_event(persona: str, result: str, error: str | None) -> None:
    """Append one JSON line to the persona's db-health log (S62). Append-only,
    never rotated by this change; a failure to write is logged, not raised —
    this check must never be the reason a bridge fails to start."""
    line = {
        "ts": datetime.now(UTC).isoformat(),
        "check": _CHECK_NAME,
        "result": result,
        "error": error,
    }
    log_path = get_log_dir() / f"db-health-{persona}.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    except OSError as exc:
        logger.warning("db-health log append failed (result=%s): %s", result, exc)


def _end_txn(conn: sqlite3.Connection) -> None:
    """Commit (or roll back) unconditionally so no boot leaves a lock held
    against the other concurrent `memories.db` connections — the same
    guarantee the retired `_boot_fts_backstop` made on every branch,
    including `could_not_check` (HIST 0d301ae0; C37(b))."""
    if not conn.in_transaction:
        return
    try:
        conn.commit()
    except sqlite3.DatabaseError:
        try:
            conn.rollback()
        except sqlite3.DatabaseError:
            pass


def _create_and_backfill_fts(conn: sqlite3.Connection) -> tuple[str, str | None]:
    """A legacy persona: `memories` already has rows but `memories_fts`
    doesn't exist yet (this check now runs BEFORE the first `MemoryStore`
    open, so it can observe this state — the retired `_boot_fts_backstop`
    never could, since it ran after schema creation). Build the FTS schema
    (the same idempotent statements `MemoryStore.__init__` runs — imported
    from there, not duplicated, so the two can never drift) and backfill
    every existing row via `rebuild`, so this persona's pre-existing
    memories don't silently sit outside the FTS index for this process's
    whole lifetime (code-red-team pass 1, M2)."""
    from brain.memory.store import _SCHEMA

    try:
        conn.executescript(_SCHEMA)
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
    except sqlite3.DatabaseError as exc:
        _end_txn(conn)
        msg = str(exc)
        return _classify_error_message(msg), msg
    _end_txn(conn)
    return "fts_missing_backfilled", None


def _classify_and_run(conn: sqlite3.Connection) -> tuple[str, str | None]:
    """Run the FTS5 integrity-check in its own short transaction and
    classify the outcome (S60). Ends the transaction on every branch."""
    try:
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')")
    except sqlite3.DatabaseError as exc:
        msg = str(exc)
        low = msg.lower()
        if "no such table" in low:
            # Two distinct cases share this error text (stage-6 red-team
            # finding, code-red-team pass 1 M2): a brand-new persona (NO
            # schema at all yet — `memories` doesn't exist either, the
            # first MemoryStore open right after this check creates
            # everything) vs. a legacy persona whose `memories` table
            # already has rows but predates the `memories_fts` table (this
            # check now runs BEFORE the first store open, so — unlike the
            # retired `_boot_fts_backstop`, which ran AFTER schema creation
            # and so never saw this state — it can observe `memories_fts`
            # missing while `memories` is already populated). Only the
            # first case is a true no-op; the second must build the FTS
            # schema and backfill it now, or the persona's pre-existing
            # memories silently never enter the FTS index for this entire
            # process lifetime (the once-per-process flag would prevent any
            # later retry).
            mem_exists = (
                conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
                ).fetchone()
                is not None
            )
            if not mem_exists:
                result, error = "no_schema", None
            else:
                result, error = _create_and_backfill_fts(conn)
        else:
            # A lock timeout, or any other non-corruption error, is
            # `could_not_check` — never corruption (S60; see
            # `_classify_error_message`'s docstring for why the catch-all
            # case is could_not_check, not "damaged").
            result, error = _classify_error_message(msg), msg
        _end_txn(conn)
        return result, error

    # integrity-check passed; compare `memories` against the FTS shadow
    # table's own row count (`_docsize`, not the external-content FTS table
    # itself — see the retired `_boot_fts_backstop` docstring for why).
    try:
        mem_rows = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        fts_rows = conn.execute("SELECT COUNT(*) FROM memories_fts_docsize").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        _end_txn(conn)
        msg = str(exc)
        return _classify_error_message(msg), msg

    _end_txn(conn)
    if mem_rows != fts_rows:
        return "count_mismatch", None
    return "healthy", None


def run_fts_health_check_once(db_path: str | Path) -> None:
    """Run the FTS5 health check against `db_path` at most once per process.

    A second (or 21st) call for the same resolved path in this process is a
    no-op. Rebuilds the index only on `damaged`/`malformed`/`count_mismatch`;
    a `could_not_check` (lock timeout) is retried at the next process start,
    not later in this one (S60) — the once-per-process flag is set here
    regardless of outcome.
    """
    resolved = Path(db_path).resolve()
    with _lock:
        if resolved in _checked_paths:
            return
        _checked_paths.add(resolved)

    persona = resolved.parent.name
    conn = sqlite3.connect(str(resolved), timeout=dev_constants.MEMORIES_DB_BUSY_TIMEOUT_S)
    try:
        conn.execute(
            f"PRAGMA busy_timeout = {int(dev_constants.MEMORIES_DB_BUSY_TIMEOUT_S * 1000)}"
        )
        result, error = _classify_and_run(conn)
        if result in _REBUILD_ON:
            try:
                conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                _end_txn(conn)
            except sqlite3.DatabaseError as exc:
                _end_txn(conn)
                logger.warning("memories_fts rebuild failed: %s", exc)
                _log_health_event(persona, "rebuild_failed", str(exc))
        if result not in ("healthy", "no_schema"):
            _log_health_event(persona, result, error)
    finally:
        conn.close()


def _reset_for_tests() -> None:
    """Test-only: clear the once-per-process checked-paths set so a test can
    force the check to run again against the same resolved path."""
    with _lock:
        _checked_paths.clear()
