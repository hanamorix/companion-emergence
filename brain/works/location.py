"""brain.works.location — the one place that names the works database (#606).

Every opener of the works index resolves its path here. The live file is ``<persona>/works.db``. Before #606 the works
tools (``save_work``, ``list_works``, ``read_work``, ``search_works``) and the integrity walker opened
``<persona>/data/works.db`` instead, so a persona can still hold that older file with works in it (the migration import,
anything ``save_work`` stored). The first time the path is resolved, those rows are merged into the live file by id,
checked, and the old file is kept beside itself as ``data/works.db.migrated`` (``.migrated.1`` and on if a rolled-back
older brain wrote a fresh one later) rather than deleted.

The merge is best-effort: if the old file can't be read or the check fails, nothing is committed, the old file stays
where it is, a ``RuntimeWarning`` names it, and the live path is still returned. It is retried in the next process.
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

from brain.utils.file_lock import file_lock

WORKS_DB_FILENAME = "works.db"
_LEGACY_SUBDIR = "data"
_RETIRED_SUFFIX = ".migrated"
_SQLITE_SIDECARS = ("-wal", "-shm")
_BUSY_TIMEOUT_SECONDS = 5.0

# Old files whose merge failed in this process: warned about once, not retried on every resolve.
_FAILED: set[str] = set()


class WorksMergeCheckError(RuntimeError):
    """The merged file is missing a work (or its search row) that the old file holds."""


def works_db_path(persona_dir: Path) -> Path:
    """Return ``<persona>/works.db``, first merging in an older ``<persona>/data/works.db`` if one is there."""
    persona_dir = Path(persona_dir)
    live = persona_dir / WORKS_DB_FILENAME
    legacy = persona_dir / _LEGACY_SUBDIR / WORKS_DB_FILENAME
    if legacy.is_file() and str(legacy) not in _FAILED:
        _merge_legacy(live, legacy)
    return live


def _merge_legacy(live: Path, legacy: Path) -> None:
    with file_lock(live):
        if not legacy.is_file():  # another process merged it while this one waited
            return
        try:
            _copy_rows(live, legacy)
            _retire(legacy)
        except (sqlite3.Error, OSError, WorksMergeCheckError) as exc:
            _FAILED.add(str(legacy))
            warnings.warn(
                f"works_db_path: could not merge {legacy} into {live}; the old file is left in place ({exc})",
                RuntimeWarning,
                stacklevel=3,
            )


def _copy_rows(live: Path, legacy: Path) -> None:
    """Copy every work the live file lacks, with its search row, in one transaction, then check none is missing."""
    from brain.works.store import WorksStore

    WorksStore(live)  # the live schema (and its v1 → v2 columns) exists before anything is copied into it
    conn = sqlite3.connect(str(live), timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    try:
        conn.execute("ATTACH DATABASE ? AS old", (str(legacy),))
        old_tables = {r[0] for r in conn.execute("SELECT name FROM old.sqlite_master WHERE type = 'table'")}
        if "works" not in old_tables:
            raise WorksMergeCheckError("it has no works table")
        live_cols = [r[1] for r in conn.execute("PRAGMA main.table_info(works)")]
        old_cols = {r[1] for r in conn.execute("PRAGMA old.table_info(works)")}
        # An old file from before the maker columns lacks them; the live defaults fill them in.
        cols = ", ".join(c for c in live_cols if c in old_cols)
        has_fts = "works_fts" in old_tables
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(f"INSERT OR IGNORE INTO main.works ({cols}) SELECT {cols} FROM old.works")
            if has_fts:
                conn.execute(
                    "INSERT INTO main.works_fts (id, title, summary, content) "
                    "SELECT id, title, summary, content FROM old.works_fts "
                    "WHERE id NOT IN (SELECT id FROM main.works_fts)"
                )
            _check(conn, has_fts=has_fts)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        conn.execute("DETACH DATABASE old")
    finally:
        conn.close()


def _check(conn: sqlite3.Connection, *, has_fts: bool) -> None:
    missing = conn.execute("SELECT COUNT(*) FROM old.works WHERE id NOT IN (SELECT id FROM main.works)").fetchone()[0]
    if missing:
        raise WorksMergeCheckError(f"{missing} work(s) did not copy")
    if has_fts:
        unindexed = conn.execute(
            "SELECT COUNT(*) FROM old.works_fts WHERE id NOT IN (SELECT id FROM main.works_fts)"
        ).fetchone()[0]
        if unindexed:
            raise WorksMergeCheckError(f"{unindexed} search row(s) did not copy")


def _retire(legacy: Path) -> None:
    """Fold the old file's write-ahead log into it, then keep it as ``.migrated`` (or the next free ``.migrated.N``)."""
    conn = sqlite3.connect(str(legacy), timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        try:
            conn.execute("PRAGMA journal_mode = DELETE")
        except sqlite3.OperationalError:
            pass  # another handle still has it open; its -wal and -shm then move with it below
    finally:
        conn.close()
    target = legacy.with_name(legacy.name + _RETIRED_SUFFIX)
    n = 0
    while target.exists():
        n += 1
        target = legacy.with_name(f"{legacy.name}{_RETIRED_SUFFIX}.{n}")
    for suffix in ("", *_SQLITE_SIDECARS):  # the database first, then any -wal / -shm left beside it
        source = legacy.with_name(legacy.name + suffix)
        if suffix == "" or source.exists():
            source.replace(target.with_name(target.name + suffix))  # atomic-write: ok -- a one-time rename of a finished database file (and its sidecars), not a write
