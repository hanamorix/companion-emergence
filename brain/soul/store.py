"""SoulStore — SQLite-backed persistence for Crystallization records.

Follows the same PRAGMA integrity_check + row_factory pattern as MemoryStore.
Soul crystallizations are permanent — the only mutation is marking revoked,
which sets revoked_at + revoked_reason but keeps the row in the DB (soft delete).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.soul.crystallization import Crystallization


@dataclass
class VoiceEvolution:
    """An accepted voice-template edit, recorded for posterity.

    `accepted_at` is an ISO-8601 timestamp string (UTC). `evidence` is a list
    of opaque source identifiers (dream ids, crystallization ids, tone-shift
    ids) that motivated the edit. `audit_id` links back to the initiate
    audit row. `user_modified` distinguishes edits the user re-wrote before
    accepting from edits accepted verbatim.
    """

    id: str
    accepted_at: str
    diff: str
    old_text: str
    new_text: str
    rationale: str
    evidence: list[str]
    audit_id: str
    user_modified: bool

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

        # WAL + 5s busy_timeout — soul crystallizations are durable
        # and accessed concurrently (review path + autonomous soul
        # acceptance + chat tools). Without WAL, concurrent writes
        # surface 'database is locked' more readily than the memory
        # store. Set AFTER integrity check so corrupt-file probes
        # surface BrainIntegrityError, not a pragma crash. In-memory
        # dbs reject WAL silently; same fallback as MemoryStore.
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
        self._conn.execute("PRAGMA busy_timeout = 5000")
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

    def _ensure_voice_evolution_table(self) -> None:
        """Create the voice_evolution table lazily on first use."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_evolution (
                id              TEXT PRIMARY KEY,
                accepted_at     TEXT NOT NULL,
                diff            TEXT NOT NULL,
                old_text        TEXT NOT NULL,
                new_text        TEXT NOT NULL,
                rationale       TEXT NOT NULL,
                evidence_json   TEXT NOT NULL,
                audit_id        TEXT NOT NULL,
                user_modified   INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def save_voice_evolution(self, ev: VoiceEvolution) -> None:
        """Persist a voice_evolution record. Idempotent on id (INSERT OR REPLACE)."""
        self._ensure_voice_evolution_table()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO voice_evolution
                (id, accepted_at, diff, old_text, new_text, rationale,
                 evidence_json, audit_id, user_modified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ev.id,
                ev.accepted_at,
                ev.diff,
                ev.old_text,
                ev.new_text,
                ev.rationale,
                json.dumps(ev.evidence),
                ev.audit_id,
                1 if ev.user_modified else 0,
            ),
        )
        self._conn.commit()

    def list_voice_evolution(self) -> list[VoiceEvolution]:
        """Return all voice_evolution records, oldest accepted_at first."""
        self._ensure_voice_evolution_table()
        rows = self._conn.execute(
            """
            SELECT id, accepted_at, diff, old_text, new_text,
                   rationale, evidence_json, audit_id, user_modified
            FROM voice_evolution
            ORDER BY accepted_at ASC
            """
        ).fetchall()
        return [
            VoiceEvolution(
                id=r[0],
                accepted_at=r[1],
                diff=r[2],
                old_text=r[3],
                new_text=r[4],
                rationale=r[5],
                evidence=json.loads(r[6]) if r[6] else [],
                audit_id=r[7],
                user_modified=bool(r[8]),
            )
            for r in rows
        ]

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
