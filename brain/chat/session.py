"""SP-6 chat session state — per-session in-memory registry.

Ported from OG NellBrain/nell_bridge_session.py (F36) with adaptations:
  - uses brain.bridge.chat.ChatMessage (typed) instead of plain dicts
  - drops OG model/client fields — provider is injected per-turn in SP-6
  - drops OG in_flight flag — CLI engine is synchronous, no concurrent turns
  - session_id UUID4 generated at create_session time

Design note: persistence across process restarts is intentionally out of
scope (matching OG). Sessions are in-memory only; on CLI exit the REPL
flushes the buffer via close_session → ingest pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from brain.bridge.chat import ChatMessage

# Truncate at 20 user+assistant pairs (40 messages total), matching OG.
HISTORY_MAX_TURNS = 20


@dataclass
class SessionState:
    """In-memory state for one chat session.

    Attributes
    ----------
    session_id:
        UUIDv4 generated at creation. Stable for the session lifetime.
    persona_name:
        Name of the persona directory this session is addressing.
    created_at:
        When the session was opened (UTC).
    history:
        Rolling message window — last HISTORY_MAX_TURNS user+assistant pairs.
    turns:
        Count of completed turn pairs (user + assistant).
    last_turn_at:
        UTC timestamp of the last completed turn pair (None until first turn).
    """

    session_id: str
    persona_name: str
    created_at: datetime
    history: list[ChatMessage] = field(default_factory=list)
    turns: int = 0
    last_turn_at: datetime | None = None

    def append_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Append one user+assistant pair and maintain the rolling window.

        Truncates to the last HISTORY_MAX_TURNS pairs (2 * HISTORY_MAX_TURNS
        ChatMessage entries) after appending. Increments turns. Sets
        last_turn_at to now (UTC).
        """
        self.history.append(ChatMessage(role="user", content=user_msg))
        self.history.append(ChatMessage(role="assistant", content=assistant_msg))
        max_msgs = HISTORY_MAX_TURNS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]
        self.turns += 1
        self.last_turn_at = datetime.now(UTC)


# ---------------------------------------------------------------------------
# Module-level in-memory registry
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, SessionState] = {}


def create_session(persona_name: str) -> SessionState:
    """Create a new session with a fresh UUIDv4 id and register it.

    Parameters
    ----------
    persona_name:
        Name of the persona this session speaks as.

    Returns
    -------
    The newly created SessionState (already in the registry).
    """
    sid = str(uuid.uuid4())
    state = SessionState(
        session_id=sid,
        persona_name=persona_name,
        created_at=datetime.now(UTC),
    )
    _SESSIONS[sid] = state
    return state


def get_session(session_id: str) -> SessionState | None:
    """Return the session for the given id, or None if unknown."""
    return _SESSIONS.get(session_id)


def all_sessions() -> list[SessionState]:
    """Return all registered sessions.

    H-B hardening (2026-04-28): with `remove_session` now wired into
    `/sessions/close`, this returns only LIVE sessions — closed ones are
    pulled from the registry. Long-lived bridges no longer accumulate stale.
    """
    return list(_SESSIONS.values())


def remove_session(session_id: str) -> bool:
    """Remove a session from the registry. Returns True if it was present.

    Called by `/sessions/close` after the ingest pipeline succeeds, so
    `sessions_active` count stays accurate and in_flight_locks can be
    cleaned up by the caller. Idempotent — calling on an unknown id is
    a silent False return, not an error.
    """
    return _SESSIONS.pop(session_id, None) is not None


def reset_registry() -> None:
    """Clear the in-memory registry. Test-only helper."""
    _SESSIONS.clear()
