"""brain.works.store — SQLite index for the works portfolio.

Mirrors brain/memory/store.py conventions:
- Open per-call (no long-lived connection that crosses threads).
- WAL mode for concurrent readers + single writer.
- Schema versioning via PRAGMA user_version.

The store is the index; the canonical content lives in markdown files
under persona/<name>/data/works/<id>.md (see brain.works.storage).
The store and the markdown files are kept in sync by the same
transaction in WorksStore.insert.

FTS5 virtual table (works_fts) powers search by title + summary +
content. Memory store uses LIKE-based search; works establishes the
FTS5 pattern as the project's first FTS user. Future memory work may
backport.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from brain.works import Work


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
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

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

    def insert(self, work: Work, *, content: str) -> None:
        """Insert a work + index its content for FTS. Idempotent on id.

        The transaction inserts into both `works` and `works_fts` so the
        FTS index never lags the canonical row.
        """
        content_path = f"data/works/{work.id}.md"
        with self._connect() as conn:
            cur = conn.execute("SELECT 1 FROM works WHERE id = ?", (work.id,))
            if cur.fetchone() is not None:
                return  # idempotent
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
                "INSERT INTO works_fts (id, title, summary, content) VALUES (?, ?, ?, ?)",
                (work.id, work.title, work.summary or "", content),
            )
            conn.commit()

    # ----- reads -----

    def get(self, work_id: str) -> Work | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM works WHERE id = ?", (work_id,)
            ).fetchone()
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
            rows = conn.execute(sql, params).fetchall()
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
            except sqlite3.OperationalError:
                # Malformed FTS5 query — return empty results rather than
                # raise. The LLM's tool path and the bridge endpoint both
                # benefit from a clean "no matches" instead of a stack trace.
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
