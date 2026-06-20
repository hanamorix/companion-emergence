"""KindledLinkStore — SQLite peers + consent state machine + consumed-invite
ledger. Phase 1 scope: pairing/consent only (sessions/messages/relationship are
later phases). Connection idiom mirrors brain/memory/store.py (integrity check →
WAL + 5s busy_timeout → Row → executescript)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from brain.kindled_link import limits

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
CREATE TABLE IF NOT EXISTS sessions (
    peer_id          TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    state            TEXT NOT NULL,
    msg_count        INTEGER NOT NULL,
    started_at       TEXT NOT NULL,
    ended_at         TEXT,
    last_outbound_at TEXT,
    cooldown_until   TEXT,
    PRIMARY KEY (peer_id, session_id)
);
CREATE TABLE IF NOT EXISTS peer_counters (
    peer_id             TEXT PRIMARY KEY,
    reset_date          TEXT NOT NULL,
    outbound_count      INTEGER NOT NULL,
    provider_call_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS outbound_drafts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    peer_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transcript (
    peer_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    direction   TEXT NOT NULL,
    text        TEXT NOT NULL,
    provenance  TEXT NOT NULL,
    ts          TEXT NOT NULL,
    PRIMARY KEY (peer_id, session_id, seq)
);
CREATE TABLE IF NOT EXISTS disclosure_budget (
    peer_id     TEXT PRIMARY KEY,
    budget      REAL NOT NULL,
    updated_at  TEXT NOT NULL
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

    # --- sessions (Phase 3) ---
    def create_session(self, peer_id: str, session_id: str, now: datetime) -> None:
        self._conn.execute(
            """INSERT INTO sessions
               (peer_id, session_id, state, msg_count, started_at)
               VALUES (?, ?, 'open', 0, ?)""",
            (peer_id, session_id, now.isoformat()),
        )
        self._conn.commit()

    def get_session(self, peer_id: str, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE peer_id = ? AND session_id = ?",
            (peer_id, session_id),
        ).fetchone()
        return dict(row) if row else None

    def get_active_session(self, peer_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE peer_id = ? AND state = 'open' "
            "ORDER BY started_at DESC LIMIT 1",
            (peer_id,),
        ).fetchone()
        return dict(row) if row else None

    def bump_session_outbound(
        self, peer_id: str, session_id: str, now: datetime
    ) -> None:
        self._conn.execute(
            """UPDATE sessions
               SET msg_count = msg_count + 1, last_outbound_at = ?
               WHERE peer_id = ? AND session_id = ?""",
            (now.isoformat(), peer_id, session_id),
        )
        self._conn.commit()

    def end_session(
        self, peer_id: str, session_id: str, *, now: datetime,
        cooldown_until: datetime,
    ) -> None:
        self._conn.execute(
            """UPDATE sessions
               SET state = 'ended', ended_at = ?, cooldown_until = ?
               WHERE peer_id = ? AND session_id = ?""",
            (now.isoformat(), cooldown_until.isoformat(), peer_id, session_id),
        )
        self._conn.commit()

    # --- peer_counters (Phase 3, daily caps; fail-safe-permissive reset) ---
    def get_counters(self, peer_id: str, today: str) -> dict:
        row = self._conn.execute(
            "SELECT * FROM peer_counters WHERE peer_id = ?", (peer_id,)
        ).fetchone()
        if row is None or row["reset_date"] != today:
            return {"outbound_count": 0, "provider_call_count": 0}
        return {
            "outbound_count": int(row["outbound_count"]),
            "provider_call_count": int(row["provider_call_count"]),
        }

    def _ensure_counter_row(self, peer_id: str, today: str) -> None:
        row = self._conn.execute(
            "SELECT reset_date FROM peer_counters WHERE peer_id = ?", (peer_id,)
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO peer_counters "
                "(peer_id, reset_date, outbound_count, provider_call_count) "
                "VALUES (?, ?, 0, 0)",
                (peer_id, today),
            )
        elif row["reset_date"] != today:
            self._conn.execute(
                "UPDATE peer_counters SET reset_date = ?, outbound_count = 0, "
                "provider_call_count = 0 WHERE peer_id = ?",
                (today, peer_id),
            )

    def incr_outbound_count(self, peer_id: str, today: str) -> None:
        self._ensure_counter_row(peer_id, today)
        self._conn.execute(
            "UPDATE peer_counters SET outbound_count = outbound_count + 1 "
            "WHERE peer_id = ?",
            (peer_id,),
        )
        self._conn.commit()

    def incr_provider_count(self, peer_id: str, today: str) -> None:
        self._ensure_counter_row(peer_id, today)
        self._conn.execute(
            "UPDATE peer_counters SET provider_call_count = provider_call_count + 1 "
            "WHERE peer_id = ?",
            (peer_id,),
        )
        self._conn.commit()

    # --- outbound_drafts (Phase 3, recovery re-gate) ---
    def save_draft(
        self, *, peer_id: str, session_id: str, payload_json: str,
        now: datetime, status: str = "pending",
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO outbound_drafts
               (peer_id, session_id, payload_json, status, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (peer_id, session_id, payload_json, status, now.isoformat()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_pending_drafts(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM outbound_drafts WHERE status = 'pending' ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def set_draft_status(self, draft_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE outbound_drafts SET status = ? WHERE id = ?",
            (status, draft_id),
        )
        self._conn.commit()

    # --- transcript (Phase 3, provenance-marked) ---
    def append_transcript(
        self, *, peer_id: str, session_id: str, seq: int, direction: str,
        text: str, now: datetime, provenance: str,
    ) -> None:
        self._conn.execute(
            """INSERT INTO transcript
               (peer_id, session_id, seq, direction, text, provenance, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (peer_id, session_id, seq, direction, text, provenance,
             now.isoformat()),
        )
        self._conn.commit()

    def recent_transcript(self, peer_id: str, *, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM transcript WHERE peer_id = ? "
            "ORDER BY seq DESC LIMIT ?",
            (peer_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def latest_cooldown_until(self, peer_id: str) -> str | None:
        """The latest cooldown expiry (ISO string) across ALL ended sessions for
        a peer, or None if none is set. Uses MAX so a session that ended later
        but with an earlier cooldown cannot hide a still-cooling earlier-ended
        session (ISO-8601 UTC strings sort chronologically)."""
        row = self._conn.execute(
            "SELECT MAX(cooldown_until) AS cu FROM sessions "
            "WHERE peer_id = ? AND state = 'ended' AND cooldown_until IS NOT NULL",
            (peer_id,),
        ).fetchone()
        return row["cu"] if row and row["cu"] else None

    # --- disclosure budget (Phase 4, parent §12; pull-computed refill) ---
    def get_disclosure_budget(self, peer_id: str, now: datetime) -> float:
        row = self._conn.execute(
            "SELECT budget, updated_at FROM disclosure_budget WHERE peer_id = ?",
            (peer_id,),
        ).fetchone()
        if row is None:
            return limits.BUDGET_MAX
        return self._refilled(row["budget"], row["updated_at"], now)

    def debit_disclosure_budget(
        self, peer_id: str, amount: float, now: datetime
    ) -> None:
        current = self.get_disclosure_budget(peer_id, now)  # refill-to-now
        new_budget = max(0.0, current - amount)
        self._conn.execute(
            """INSERT INTO disclosure_budget (peer_id, budget, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(peer_id) DO UPDATE SET
                   budget = excluded.budget, updated_at = excluded.updated_at""",
            (peer_id, new_budget, now.isoformat()),
        )
        self._conn.commit()

    @staticmethod
    def _refilled(stored: float, updated_at: str, now: datetime) -> float:
        try:
            elapsed_days = (
                now - datetime.fromisoformat(updated_at)
            ).total_seconds() / 86400.0
        except (ValueError, TypeError):
            elapsed_days = 0.0
        refilled = stored + max(0.0, elapsed_days) * limits.BUDGET_REFILL_PER_DAY
        return min(limits.BUDGET_MAX, max(0.0, refilled))
