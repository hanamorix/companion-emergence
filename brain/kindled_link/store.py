"""KindledLinkStore — SQLite peers + consent state machine + consumed-invite
ledger. Phase 1 scope: pairing/consent only (sessions/messages/relationship are
later phases). Connection idiom mirrors brain/memory/store.py (integrity check →
WAL + 5s busy_timeout → Row → executescript)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS peers (
    peer_id        TEXT PRIMARY KEY,
    identity_pub   TEXT NOT NULL,
    fingerprint    TEXT NOT NULL,
    consent_state  TEXT NOT NULL,
    relay_url      TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS consumed_invites (
    invite_id   TEXT PRIMARY KEY,
    consumed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS seq_high_water (
    peer_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    high_water  INTEGER NOT NULL,
    PRIMARY KEY (peer_id, session_id)
);
"""

CONSENT_STATES = frozenset(
    {"pending_local", "pending_remote", "paired", "paused", "revoked", "blocked"}
)
_ALLOWED_TRANSITIONS = {
    "pending_local": {"pending_remote", "paired", "revoked", "blocked"},
    "pending_remote": {"paired", "revoked", "blocked"},
    "paired": {"paused", "revoked", "blocked"},
    "paused": {"paired", "revoked", "blocked"},
    "revoked": {"blocked"},
    "blocked": set(),
}


class ConsentTransitionError(ValueError):
    """An illegal consent transition or a re-consumed invite."""


class KindledLinkStore:
    def __init__(self, db_path: str | Path, *, integrity_check: bool = True) -> None:
        self._conn = sqlite3.connect(str(db_path))
        if integrity_check:
            try:
                result = self._conn.execute("PRAGMA integrity_check").fetchall()
            except sqlite3.DatabaseError as exc:
                self._conn.close()
                from brain.health.anomaly import BrainIntegrityError

                raise BrainIntegrityError(str(db_path), str(exc)) from exc
            if result != [("ok",)]:
                detail = "; ".join(str(row[0]) for row in result)
                self._conn.close()
                from brain.health.anomaly import BrainIntegrityError

                raise BrainIntegrityError(str(db_path), detail)
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def upsert_peer(
        self,
        *,
        peer_id: str,
        identity_pub_hex: str,
        fingerprint: str,
        consent_state: str,
        relay_url: str | None,
        now: datetime,
    ) -> None:
        ts = now.isoformat()
        self._conn.execute(
            """
            INSERT INTO peers (peer_id, identity_pub, fingerprint, consent_state,
                               relay_url, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(peer_id) DO UPDATE SET
                identity_pub=excluded.identity_pub,
                fingerprint=excluded.fingerprint,
                consent_state=excluded.consent_state,
                relay_url=excluded.relay_url,
                updated_at=excluded.updated_at
            """,
            (peer_id, identity_pub_hex, fingerprint, consent_state, relay_url, ts, ts),
        )
        self._conn.commit()

    def get_peer(self, peer_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM peers WHERE peer_id = ?", (peer_id,)
        ).fetchone()
        return dict(row) if row else None

    def set_consent(self, peer_id: str, new_state: str, now: datetime) -> None:
        if new_state not in CONSENT_STATES:
            raise ConsentTransitionError(f"unknown consent state: {new_state!r}")
        peer = self.get_peer(peer_id)
        if peer is None:
            raise ConsentTransitionError(f"no such peer: {peer_id!r}")
        current = peer["consent_state"]
        if new_state not in _ALLOWED_TRANSITIONS[current]:
            raise ConsentTransitionError(f"{current} -> {new_state} not allowed")
        self._conn.execute(
            "UPDATE peers SET consent_state = ?, updated_at = ? WHERE peer_id = ?",
            (new_state, now.isoformat(), peer_id),
        )
        self._conn.commit()

    def is_invite_consumed(self, invite_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM consumed_invites WHERE invite_id = ?", (invite_id,)
        ).fetchone()
        return row is not None

    def mark_invite_consumed(self, invite_id: str, now: datetime) -> None:
        try:
            self._conn.execute(
                "INSERT INTO consumed_invites (invite_id, consumed_at) VALUES (?, ?)",
                (invite_id, now.isoformat()),
            )
        except sqlite3.IntegrityError as exc:
            raise ConsentTransitionError(
                f"invite already consumed: {invite_id!r}"
            ) from exc
        self._conn.commit()

    def get_seq_high_water(self, peer_id: str, session_id: str) -> int:
        """The highest accepted per-(peer, session) sequence (0 if none). Used by
        the receiver to reject replayed/duplicate envelopes (protocol §8 rule 5)."""
        row = self._conn.execute(
            "SELECT high_water FROM seq_high_water WHERE peer_id = ? AND session_id = ?",
            (peer_id, session_id),
        ).fetchone()
        return int(row["high_water"]) if row else 0

    def set_seq_high_water(self, peer_id: str, session_id: str, value: int) -> None:
        self._conn.execute(
            """
            INSERT INTO seq_high_water (peer_id, session_id, high_water)
            VALUES (?, ?, ?)
            ON CONFLICT(peer_id, session_id) DO UPDATE SET high_water = excluded.high_water
            """,
            (peer_id, session_id, value),
        )
        self._conn.commit()
