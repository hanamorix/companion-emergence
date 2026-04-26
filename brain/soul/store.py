"""SoulStore — SQLite-backed persistence for Crystallization records.

Follows the same PRAGMA integrity_check + row_factory pattern as MemoryStore.
Soul crystallizations are permanent — the only mutation is marking revoked,
which sets revoked_at + revoked_reason but keeps the row in the DB (soft delete).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from brain.soul.crystallization import Crystallization

_SCHEMA = """
CREATE TABLE IF NOT EXISTS crystallizations (
    id              TEXT PRIMARY KEY,
    moment          TEXT NOT NULL,
    love_type       TEXT NOT NULL,
    why_it_matters  TEXT NOT NULL,
    who_or_what     TEXT NOT NULL DEFAULT '',
    resonance       INTEGER NOT NULL DEFAULT 8,
    crystallized_at TEXT NOT NULL,
    permanent       INTEGER NOT NULL DEFAULT 1,
    revoked_at      TEXT,
    revoked_reason  TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_crystal_revoked ON crystallizations(revoked_at);
CREATE INDEX IF NOT EXISTS idx_crystal_created ON crystallizations(crystallized_at);
"""


class SoulStore:
    """SQLite-backed store for Crystallization records.

    Pass `":memory:"` as db_path for in-memory databases (used in tests).
    Any filesystem path creates or opens a persistent database.

    Opens with PRAGMA integrity_check on init — raises BrainIntegrityError
    if the database is corrupt, matching the MemoryStore pattern.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)

        # Integrity check BEFORE row_factory — result rows must be plain tuples.
        try:
            result = self._conn.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.DatabaseError as exc:
            self._conn.close()
            from brain.health.anomaly import BrainIntegrityError

            raise BrainIntegrityError(self._db_path, str(exc)) from exc

        if result != [("ok",)]:
            detail = "; ".join(str(row[0]) for row in result)
            self._conn.close()
            from brain.health.anomaly import BrainIntegrityError

            raise BrainIntegrityError(self._db_path, detail)

        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection. Safe to call multiple times."""
        self._conn.close()

    def create(self, c: Crystallization) -> None:
        """Insert a crystallization. Raises on duplicate id."""
        self._conn.execute(
            """
            INSERT INTO crystallizations (
                id, moment, love_type, why_it_matters, who_or_what,
                resonance, crystallized_at, permanent, revoked_at, revoked_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                c.id,
                c.moment,
                c.love_type,
                c.why_it_matters,
                c.who_or_what,
                c.resonance,
                c.crystallized_at.isoformat(),
                1 if c.permanent else 0,
                c.revoked_at.isoformat() if c.revoked_at else None,
                c.revoked_reason,
            ),
        )
        self._conn.commit()

    def get(self, id: str) -> Crystallization | None:
        """Return the Crystallization with the given id, or None."""
        row = self._conn.execute("SELECT * FROM crystallizations WHERE id = ?", (id,)).fetchone()
        return _row_to_crystal(row) if row else None

    def list_active(self) -> list[Crystallization]:
        """Return active crystallizations (not revoked), oldest-first."""
        rows = self._conn.execute(
            "SELECT * FROM crystallizations WHERE revoked_at IS NULL ORDER BY crystallized_at ASC"
        ).fetchall()
        return [_row_to_crystal(row) for row in rows]

    def list_revoked(self) -> list[Crystallization]:
        """Return revoked crystallizations, most-recently-revoked first."""
        rows = self._conn.execute(
            "SELECT * FROM crystallizations WHERE revoked_at IS NOT NULL ORDER BY revoked_at DESC"
        ).fetchall()
        return [_row_to_crystal(row) for row in rows]

    def mark_revoked(self, id: str, reason: str) -> Crystallization | None:
        """Move a crystallization to revoked state (soft delete).

        Sets revoked_at = now UTC, revoked_reason = reason.
        Returns the updated Crystallization, or None if id not found.
        """
        existing = self.get(id)
        if existing is None:
            return None

        now_iso = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE crystallizations SET revoked_at = ?, revoked_reason = ? WHERE id = ?",
            (now_iso, reason, id),
        )
        self._conn.commit()
        return self.get(id)

    def count(self) -> int:
        """Return count of active (non-revoked) crystallizations."""
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM crystallizations WHERE revoked_at IS NULL"
            ).fetchone()[0]
        )

    def __enter__(self) -> SoulStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _row_to_crystal(row: sqlite3.Row) -> Crystallization:
    """Materialise a sqlite row into a Crystallization dataclass."""
    from brain.soul.crystallization import _coerce_utc

    crystallized_at = _coerce_utc(row["crystallized_at"])
    revoked_at = _coerce_utc(row["revoked_at"]) if row["revoked_at"] else None

    return Crystallization(
        id=row["id"],
        moment=row["moment"],
        love_type=row["love_type"],
        why_it_matters=row["why_it_matters"],
        crystallized_at=crystallized_at,
        who_or_what=row["who_or_what"] or "",
        resonance=int(row["resonance"]),
        permanent=bool(row["permanent"]),
        revoked_at=revoked_at,
        revoked_reason=row["revoked_reason"] or "",
    )
