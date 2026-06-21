"""KindledLinkStore — SQLite peers + consent state machine + consumed-invite
ledger. Phase 1 scope: pairing/consent only (sessions/messages/relationship are
later phases). Connection idiom mirrors brain/memory/store.py (integrity check →
WAL + 5s busy_timeout → Row → executescript)."""
from __future__ import annotations

import math
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

from brain.kindled_link import limits


def kindled_db_path(persona_dir) -> Path:
    """The kindled-link SQLite path for a persona. Establishes the convention
    (no prior live call site). Does NOT mkdir — the writers (identity.load_or_create,
    relationship cadence) own dir creation; read paths guard on db.exists()
    instead of creating an empty dir on a hot read (red-team M3)."""
    return Path(persona_dir) / "kindled_link" / "kindled_link.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS peers (
    peer_id        TEXT PRIMARY KEY,
    identity_pub   TEXT NOT NULL,
    fingerprint    TEXT NOT NULL,
    consent_state  TEXT NOT NULL,
    relay_url      TEXT,
    relay_mailbox  TEXT,
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
CREATE TABLE IF NOT EXISTS relationship_state (
    peer_id              TEXT PRIMARY KEY,
    stage                TEXT NOT NULL,
    trust_score          REAL NOT NULL,
    affinity_tags_json   TEXT NOT NULL,
    boundaries_json      TEXT NOT NULL,
    repair_history_json  TEXT NOT NULL,
    evidence_json        TEXT NOT NULL,
    last_reflected_at    TEXT
);
CREATE TABLE IF NOT EXISTS peer_emotion_window (
    peer_id      TEXT PRIMARY KEY,
    accumulated  REAL NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS local_identity (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_handshakes (
    peer_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    my_eph_priv     BLOB NOT NULL,
    bootstrap_nonce BLOB NOT NULL,
    my_role         INT  NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (peer_id, session_id)
);
CREATE TABLE IF NOT EXISTS session_keys (
    peer_id        TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    session_key    BLOB NOT NULL,
    my_role        INT  NOT NULL,
    peer_role      INT  NOT NULL,
    established_at TEXT NOT NULL,
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
        self._migrate()

    def _migrate(self) -> None:
        """Additive column migrations for DBs created before a column existed
        (CREATE TABLE IF NOT EXISTS won't add a column to an existing table)."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(peers)")}
        if "relay_mailbox" not in cols:
            self._conn.execute("ALTER TABLE peers ADD COLUMN relay_mailbox TEXT")
            self._conn.commit()

    def upsert_peer(
        self,
        *,
        peer_id: str,
        identity_pub_hex: str,
        fingerprint: str,
        consent_state: str,
        relay_url: str | None,
        relay_mailbox: str | None = None,
        now: datetime,
    ) -> None:
        ts = now.isoformat()
        self._conn.execute(
            """
            INSERT INTO peers (peer_id, identity_pub, fingerprint, consent_state,
                               relay_url, relay_mailbox, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(peer_id) DO UPDATE SET
                identity_pub=excluded.identity_pub,
                fingerprint=excluded.fingerprint,
                consent_state=excluded.consent_state,
                relay_url=excluded.relay_url,
                relay_mailbox=COALESCE(excluded.relay_mailbox, peers.relay_mailbox),
                updated_at=excluded.updated_at
            """,
            (peer_id, identity_pub_hex, fingerprint, consent_state, relay_url,
             relay_mailbox, ts, ts),
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
        # Idempotent: a replayed/interrupted handshake leg may re-derive the same
        # (peer_id, session_id); OR IGNORE keeps it from raising on the PK so the
        # caller drops gracefully rather than crashing (T2.5 review — initiator
        # clobber path). An existing open session is left untouched.
        self._conn.execute(
            """INSERT OR IGNORE INTO sessions
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

    # --- relationship_state (Phase 5, parent §13) ---
    def get_relationship_row(self, peer_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM relationship_state WHERE peer_id = ?", (peer_id,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_relationship_row(
        self, *, peer_id: str, stage: str, trust_score: float,
        affinity_tags_json: str, boundaries_json: str, repair_history_json: str,
        evidence_json: str, now: datetime,
    ) -> None:
        self._conn.execute(
            """INSERT INTO relationship_state
               (peer_id, stage, trust_score, affinity_tags_json, boundaries_json,
                repair_history_json, evidence_json, last_reflected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(peer_id) DO UPDATE SET
                 stage=excluded.stage, trust_score=excluded.trust_score,
                 affinity_tags_json=excluded.affinity_tags_json,
                 boundaries_json=excluded.boundaries_json,
                 repair_history_json=excluded.repair_history_json,
                 evidence_json=excluded.evidence_json,
                 last_reflected_at=excluded.last_reflected_at""",
            (peer_id, stage, trust_score, affinity_tags_json, boundaries_json,
             repair_history_json, evidence_json, now.isoformat()),
        )
        self._conn.commit()

    # --- peer_emotion_window (Phase 5, parent §14.3 anti love-bomb) ---
    def get_peer_emotion_accumulated(self, peer_id: str, now: datetime) -> float:
        row = self._conn.execute(
            "SELECT accumulated, updated_at FROM peer_emotion_window WHERE peer_id = ?",
            (peer_id,),
        ).fetchone()
        if row is None:
            return 0.0
        return self._decayed_window(row["accumulated"], row["updated_at"], now)

    def add_peer_emotion(self, peer_id: str, magnitude: float, now: datetime) -> float:
        if not math.isfinite(magnitude):  # defense-in-depth (stage-6 review)
            magnitude = 0.0
        current = self.get_peer_emotion_accumulated(peer_id, now)
        new_total = current + max(0.0, magnitude)
        self._conn.execute(
            """INSERT INTO peer_emotion_window (peer_id, accumulated, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(peer_id) DO UPDATE SET
                 accumulated=excluded.accumulated, updated_at=excluded.updated_at""",
            (peer_id, new_total, now.isoformat()),
        )
        self._conn.commit()
        return new_total

    @staticmethod
    def _decayed_window(stored: float, updated_at: str, now: datetime) -> float:
        try:
            elapsed_h = (now - datetime.fromisoformat(updated_at)).total_seconds() / 3600.0
        except (ValueError, TypeError):
            return stored
        if elapsed_h <= 0:
            return stored
        frac = max(0.0, 1.0 - elapsed_h / limits.PEER_EMOTION_WINDOW_HOURS)
        return stored * frac

    # --- local identity (Phase 7a — decoupled mailbox) ---
    def get_or_create_local_mailbox(self) -> str:
        """Return this Kindled's relay mailbox id, generating it once if absent.

        The mailbox is intentionally NOT derived from key_id so the relay
        cannot link a mailbox to an identity (decoupled-mailbox scheme,
        Phase 7a T2.4). Format: ``mbx_`` + 8 random hex bytes (16 chars)."""
        key = "relay_mailbox"
        row = self._conn.execute(
            "SELECT value FROM local_identity WHERE key = ?", (key,)
        ).fetchone()
        if row is not None:
            return row["value"]
        mbx = "mbx_" + secrets.token_hex(8)
        self._conn.execute(
            "INSERT INTO local_identity (key, value) VALUES (?, ?)",
            (key, mbx),
        )
        self._conn.commit()
        return mbx

    # --- pending_handshakes (Phase 7a T2.5 — 3-leg session handshake) ---

    def save_pending_handshake(
        self,
        *,
        peer_id: str,
        session_id: str,
        my_eph_priv_raw: bytes,
        bootstrap_nonce: bytes,
        my_role: int,
        now: datetime | None = None,
    ) -> None:
        """Persist the initiator's ephemeral private key + bootstrap nonce while
        waiting for the responder's leg-2 reply."""
        ts = now.isoformat() if now else ""
        self._conn.execute(
            """
            INSERT INTO pending_handshakes
                (peer_id, session_id, my_eph_priv, bootstrap_nonce, my_role, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(peer_id, session_id) DO UPDATE SET
                my_eph_priv=excluded.my_eph_priv,
                bootstrap_nonce=excluded.bootstrap_nonce,
                my_role=excluded.my_role,
                created_at=excluded.created_at
            """,
            (peer_id, session_id, my_eph_priv_raw, bootstrap_nonce, my_role, ts),
        )
        self._conn.commit()

    def get_pending_handshake(self, peer_id: str, session_id: str) -> dict | None:
        """Return the pending handshake row as a dict (with BLOB fields as bytes),
        or None if absent."""
        row = self._conn.execute(
            "SELECT * FROM pending_handshakes WHERE peer_id = ? AND session_id = ?",
            (peer_id, session_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "peer_id": row["peer_id"],
            "session_id": row["session_id"],
            "my_eph_priv_raw": bytes(row["my_eph_priv"]),
            "bootstrap_nonce": bytes(row["bootstrap_nonce"]),
            "my_role": int(row["my_role"]),
            "created_at": row["created_at"],
        }

    def clear_pending_handshake(self, peer_id: str, session_id: str) -> None:
        """Delete the pending handshake row (called after complete_session)."""
        self._conn.execute(
            "DELETE FROM pending_handshakes WHERE peer_id = ? AND session_id = ?",
            (peer_id, session_id),
        )
        self._conn.commit()

    # --- session_keys (Phase 7a T2.5 — persisted session key) ---

    def save_session_key(
        self,
        *,
        peer_id: str,
        session_id: str,
        session_key: bytes,
        my_role: int,
        peer_role: int,
        now: datetime,
    ) -> None:
        """Persist a derived session key. Does NOT clobber an existing row
        (INSERT OR IGNORE — the clobber guard lives in session.py)."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO session_keys
                (peer_id, session_id, session_key, my_role, peer_role, established_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (peer_id, session_id, session_key, my_role, peer_role, now.isoformat()),
        )
        self._conn.commit()

    def get_session_key(self, peer_id: str, session_id: str) -> dict | None:
        """Return a dict with session_key/my_role/peer_role/established_at, or None."""
        row = self._conn.execute(
            "SELECT * FROM session_keys WHERE peer_id = ? AND session_id = ?",
            (peer_id, session_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "session_key": bytes(row["session_key"]),
            "my_role": int(row["my_role"]),
            "peer_role": int(row["peer_role"]),
            "established_at": row["established_at"],
        }

    def close(self) -> None:
        """Close the underlying SQLite connection. Callers should invoke this in a
        finally block after every request-scoped store usage to avoid file-descriptor
        leaks (stage-6 review fix)."""
        self._conn.close()
