"""Shared bounded-retry `PRAGMA integrity_check` helper for all 4 sqlite
stores (`MemoryStore`, `HebbianMatrix`, `KindledLinkStore`, `SoulStore`).

C16 Windows CI flake fix (ram-spike-fix INC-10 follow-up), generalized from
`MemoryStore`-only (commit 06847436) to all 4 stores per orchestrator
directive 2026-09-28: "same known, cheap defect class... fix it here rather
than defer."

Every one of the 4 stores used to convert ANY `sqlite3.DatabaseError` from
its own `__init__`'s `PRAGMA integrity_check` straight into
`BrainIntegrityError` — a "the brain is corrupted, unrecoverable" Layer-3
alarm (`brain.health.anomaly.BrainIntegrityError`). That conflates genuine
corruption with a transient OS-level condition: on Windows, `Popen.kill()`
-> `TerminateProcess` can leave a killed process's WAL/-shm memory-mapped
section released slightly after `proc.wait()` returns, so the very next
store open against the same file (a gated job resuming after a hard-kill,
or a bridge's dirty-restart recovery opening `MemoryStore` then
`HebbianMatrix` back-to-back against the same persona dir) can see a
`disk I/O error` that clears within milliseconds and has nothing to do
with the database's actual health.

This module is the ONE place the transient-vs-corruption classification and
the retry loop live, so the four stores can't drift out of sync with each
other (each used to carry an independent, easy-to-diverge copy of the same
try/except block).

Design note (why a small, POSITIVE allowlist, not a negative one): retrying
only a few independently-verified known-transient message substrings is
deliberately much narrower than "retry anything not explicitly known to be
corruption" — that broader default was tried and rejected during this
change's own plan red-team, because `PRAGMA integrity_check`'s error
vocabulary is wide (a garbled/overwritten file raises `"file is not a
database"`, which must NOT retry — that's real corruption, not a transient
hiccup, and contains neither "malformed" nor "corrupt"). A message NOT on
the allowlist is NOT assumed transient and raises immediately, exactly as
every store's constructor behaved before this retry mechanism existed.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from brain import dev_constants

logger = logging.getLogger(__name__)

# Known-transient (never corruption) SQLite error-message substrings for
# PRAGMA integrity_check, matched lower-cased. Each entry independently
# verified against real SQLite output (CPython sqlite3 + libsqlite3), not
# copied from another module's docstring:
#   - "disk i/o error": the observed root cause (C16 Windows CI flake) --
#     see this module's docstring.
#   - "database is locked": SQLITE_BUSY's actual message text (verified via
#     a live two-connection repro) -- a concurrent writer, not corruption.
#   - "unable to open database file": a transient can't-open condition
#     (permissions race, antivirus scan, file not yet flushed to disk).
# "database disk image is malformed" is DELIBERATELY excluded: genuine
# corruption, must never retry.
TRANSIENT_INTEGRITY_ERROR_SUBSTRINGS = (
    "disk i/o error",
    "database is locked",
    "unable to open database file",
)


def run_integrity_check_with_retry(
    conn: sqlite3.Connection, db_path: str | Path, *, caller: str = ""
) -> list:
    """Run `PRAGMA integrity_check` on `conn`, retrying a bounded number of
    times if (and only if) the failure is a known-transient condition.

    Retries on the SAME `conn` — the condition being waited out is external
    OS/filesystem state, not anything cached in the Python-level connection
    object, and `PRAGMA integrity_check` is a read-only probe with no
    retained transaction state across attempts.

    On a non-allowlisted message (real or unrecognized corruption) — raises
    `BrainIntegrityError` immediately, closing `conn` first, byte-for-byte
    the same shape (`BrainIntegrityError(str(db_path), str(exc)) from exc`)
    every store raised before this helper existed. On an allowlisted message
    that never clears within the retry budget — same immediate-raise shape,
    just after the budget is exhausted rather than on the first attempt.

    On success (the pragma does not raise) — returns the raw result rows
    (e.g. `[("ok",)]`); the caller is responsible for its own
    `result != [("ok",)]` check for a non-exception corruption signal (a
    clean-but-actually-corrupt result), unchanged from before this helper.
    Logs a warning if a retry was needed and then succeeded, so a production
    occurrence is observable without building a persistent log mechanism.

    `caller` (e.g. `"MemoryStore"`, `"HebbianMatrix"`) is cosmetic only —
    included in the retry-success log line for readability, no behavior
    effect.
    """
    attempts = dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            result = conn.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.DatabaseError as exc:
            msg = str(exc).lower()
            is_transient = any(
                substr in msg for substr in TRANSIENT_INTEGRITY_ERROR_SUBSTRINGS
            )
            if not is_transient or attempt == attempts:
                conn.close()
                from brain.health.anomaly import BrainIntegrityError

                raise BrainIntegrityError(str(db_path), str(exc)) from exc
            time.sleep(dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_DELAY_S)
            continue
        if attempt > 1:
            logger.warning(
                "%s integrity check for %s succeeded after retry (attempt "
                "%d/%d) — a transient condition cleared, not a corruption "
                "alarm.",
                caller or "sqlite store",
                db_path,
                attempt,
                attempts,
            )
        return result
    # Unreachable (the loop always returns or raises), but keeps type
    # checkers happy about a guaranteed return.
    raise AssertionError("unreachable")
