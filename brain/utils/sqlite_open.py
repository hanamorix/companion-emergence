"""One way to open a store's SQLite connection (consolidation C9 of #465).

MemoryStore, SoulStore, HebbianMatrix, KindledLinkStore and WorksStore each used
to carry their own copy of the same open sequence. It lives here once, in the
order the stores have always run it:

    connect -> integrity check (optional) -> WAL -> busy_timeout -> store setup

The order is semantic. The pragmas come after the integrity check so that a
corrupt file raises `BrainIntegrityError`, not a pragma error, and the integrity
rows are read before any `row_factory` exists, so they are plain tuples.

A named `OpenProfile` holds the two things that differ between stores: the busy
timeout and whether a WAL failure is swallowed. Whether to run the integrity
check is NOT part of a profile: it is a required keyword on every call, so this
module decides no check cadence (each store passes its own, as it always did).

Each store keeps its schema, migrations and `row_factory`; it hands them over as
the `setup` callable, which runs inside the close guard. If anything after the
connect raises, the connection is closed and the same exception propagates.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from brain import dev_constants
from brain.health import integrity_retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenProfile:
    """What differs between stores: the busy timeout and the WAL-failure policy."""

    busy_timeout_s: float
    # True: a failing `PRAGMA journal_mode = WAL` is ignored (in-memory databases
    # reject WAL). False: the failure propagates.
    wal_fallback: bool = True


DEFAULT_PROFILE = OpenProfile(dev_constants.SQLITE_STORE_BUSY_TIMEOUT_S)
STRICT_WAL_PROFILE = OpenProfile(dev_constants.SQLITE_STORE_BUSY_TIMEOUT_S, wal_fallback=False)


@contextmanager
def close_on_error(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Close `conn` and re-raise the same exception if the body raises anything.

    `BaseException`, so a KeyboardInterrupt during an open releases the handle too.
    A second `close()` on a closed sqlite3 connection does nothing, so this is safe
    after a path that already closed it. A `sqlite3.Error` from the close itself is
    swallowed (and logged at debug, so a leaked handle can be found) so it cannot
    replace the exception being propagated.
    """
    try:
        yield conn
    except BaseException:
        try:
            conn.close()
        except sqlite3.Error:
            logger.debug("closing the connection after a failed open also failed", exc_info=True)
        raise


def check_integrity(conn: sqlite3.Connection, db_path: str | Path, *, caller: str) -> None:
    """Run `PRAGMA integrity_check` (with the transient-error retry) and raise
    `BrainIntegrityError` if the result is not `[("ok",)]`.

    Does not close `conn`; the caller's close guard does. Exported so that a
    scheduler can run the same check without copying this block.
    """
    result = integrity_retry.run_integrity_check_with_retry(conn, db_path, caller=caller)
    if result != [("ok",)]:
        detail = "; ".join(str(row[0]) for row in result)
        from brain.health.anomaly import BrainIntegrityError

        raise BrainIntegrityError(str(db_path), detail)


def open_store_connection(
    db_path: str | Path,
    *,
    caller: str,
    integrity_check: bool,
    profile: OpenProfile = DEFAULT_PROFILE,
    setup: Callable[[sqlite3.Connection], None] | None = None,
) -> sqlite3.Connection:
    """Open `db_path` and return the connection; on any failure close it and re-raise.

    `caller` names the store in the integrity-retry log line. `integrity_check` has
    no default on purpose (see the module docstring). `setup(conn)` is the store's own
    tail (row factory, schema, migrations, commits) and runs last, inside the guard.
    """
    conn = sqlite3.connect(str(db_path), timeout=profile.busy_timeout_s)
    with close_on_error(conn):
        if integrity_check:
            check_integrity(conn, db_path, caller=caller)
        if profile.wal_fallback:
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass
        else:
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA busy_timeout = {int(profile.busy_timeout_s * 1000)}")
        if setup is not None:
            setup(conn)
    return conn
