"""brain.works.store — SQLite index for the works portfolio.

Mirrors brain/memory/store.py conventions:
- Open per-call (no long-lived connection that crosses threads).
- WAL mode for concurrent readers + single writer.
- Schema versioning via PRAGMA user_version.

The store is the index; the canonical content lives in markdown files
under persona/<name>/data/works/<id>.md (see brain.works.storage).
The store transaction is atomic internally; callers that also write markdown
must roll back the index row if the file write fails.

FTS5 virtual table (works_fts) powers search by title + summary +
content. Memory store uses LIKE-based search; works establishes the
FTS5 pattern as the project's first FTS user. Future memory work may
backport.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from brain.works import Work

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS works (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    type          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    session_id    TEXT,
    content_path  TEXT NOT NULL,
    word_count    INTEGER NOT NULL,
    summary       TEXT
);

CREATE INDEX IF NOT EXISTS idx_works_created_at ON works(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_works_type       ON works(type);

CREATE VIRTUAL TABLE IF NOT EXISTS works_fts USING fts5(
    id UNINDEXED,
    title,
    summary,
    content,
    tokenize='porter unicode61'
);

PRAGMA user_version = {_SCHEMA_VERSION};
"""


class WorksStore:
    """SQLite index over works.

    Use as a transient object: construct, call methods, drop. Each method
    opens a fresh sqlite3 connection so handles never cross threads.

    M-3: schema init runs once per (process, db_path). Subsequent
    WorksStore(same_path) instances skip the CREATE IF NOT EXISTS dance,
    so the hot-path (per-tool-call) is just a connect + query.
    """

    _INITIALISED_PATHS: set[str] = set()

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self._path.resolve())
        if key not in self._INITIALISED_PATHS:
            with self._connect() as conn:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            self._INITIALISED_PATHS.add(key)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def schema_version(self) -> int:
        with self._connect() as conn:
            row = conn.execute("PRAGMA user_version").fetchone()
            return int(row[0]) if row else 0

    # ----- writes -----

    def insert(self, work: Work, *, content: str) -> bool:
        """Insert a work + index its content for FTS. Idempotent on id.

        Returns True when a new row was inserted, False when the work id was
        already present. Duplicate ids must not rewrite the markdown sidecar
        with different metadata.

        Wrapped in BEGIN IMMEDIATE so:
          - The SELECT-then-INSERT compound op is serialized (writers can't
            both observe "row absent" and both insert).
          - If either INSERT raises mid-tx, the rollback is total — works
            and works_fts stay in sync (no orphan row in either table).
        """
        content_path = f"data/works/{work.id}.md"
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    "SELECT 1 FROM works WHERE id = ?", (work.id,)
                )
                if cur.fetchone() is not None:
                    conn.execute("ROLLBACK")
                    return False  # idempotent duplicate
                conn.execute(
                    """
                    INSERT INTO works
                      (id, title, type, created_at, session_id,
                       content_path, word_count, summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        work.id,
                        work.title,
                        work.type,
                        work.created_at.isoformat(),
                        work.session_id,
                        content_path,
                        work.word_count,
                        work.summary,
                    ),
                )
                conn.execute(
                    "INSERT INTO works_fts (id, title, summary, content) "
                    "VALUES (?, ?, ?, ?)",
                    (work.id, work.title, work.summary or "", content),
                )
                conn.execute("COMMIT")
                return True
            except Exception:
                # Roll back the entire transaction so neither table has
                # the row. The default sqlite3 connection __exit__ does
                # roll back on exception, but explicit ROLLBACK before
                # close makes the contract obvious and works regardless
                # of how the connection is later disposed.
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
        finally:
            conn.close()

    def delete(self, work_id: str) -> None:
        """Delete a work index row and its FTS row.

        Used by save_work to roll back the SQLite index if the markdown sidecar
        write fails after a successful insert.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM works_fts WHERE id = ?", (work_id,))
            conn.execute("DELETE FROM works WHERE id = ?", (work_id,))
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    # ----- reads -----

    def get(self, work_id: str) -> Work | None:
        with self._connect() as conn:
            try:
                row = conn.execute(
                    "SELECT * FROM works WHERE id = ?", (work_id,)
                ).fetchone()
            except sqlite3.OperationalError as exc:
                # Audit 2026-05-07 P3-5: corrupt DB / partial WAL /
                # schema mismatch is operationally distinct from
                # "no work with that id." Log loudly + still return
                # None so the chat path stays graceful, but operators
                # tailing the log + future health-anomaly readers can
                # spot a real data-integrity problem.
                logger.warning(
                    "WorksStore.get operational error for id=%s: %s", work_id, exc
                )
                return None
            return _row_to_work(row) if row else None

    def list_recent(
        self, *, limit: int = 20, type: str | None = None
    ) -> list[Work]:
        sql = "SELECT * FROM works"
        params: list[object] = []
        if type is not None:
            sql += " WHERE type = ?"
            params.append(type)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                # Audit 2026-05-07 P3-5: log + return empty rather
                # than silent. Same posture as get() / search().
                logger.warning(
                    "WorksStore.list_recent operational error: %s", exc
                )
                return []
            return [_row_to_work(r) for r in rows]

    def search(
        self, query: str, *, limit: int = 20, type: str | None = None
    ) -> list[Work]:
        if not query.strip():
            return []
        sql = """
        SELECT works.* FROM works
        JOIN works_fts ON works.id = works_fts.id
        WHERE works_fts MATCH ?
        """
        params: list[object] = [query]
        if type is not None:
            sql += " AND works.type = ?"
            params.append(type)
        sql += " ORDER BY works.created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                # Audit 2026-05-07 P3-5: malformed FTS5 query is the
                # common case (user-typed query). Distinguish it from
                # a real DB problem in the log: the FTS5 messages
                # mention 'syntax' / 'fts5'. Both still return empty
                # to keep chat graceful.
                msg = str(exc).lower()
                if "fts5" in msg or "syntax" in msg or "malformed" in msg:
                    logger.debug("WorksStore.search FTS5 syntax: %s", exc)
                else:
                    logger.warning(
                        "WorksStore.search operational error (likely "
                        "data-integrity, not user input): %s",
                        exc,
                    )
                return []
            return [_row_to_work(r) for r in rows]


def _row_to_work(row: sqlite3.Row) -> Work:
    return Work(
        id=row["id"],
        title=row["title"],
        type=row["type"],
        created_at=datetime.fromisoformat(row["created_at"]),
        session_id=row["session_id"],
        word_count=int(row["word_count"]),
        summary=row["summary"],
    )
