"""SP-6 chat session state — per-session in-memory registry.

Ported from OG NellBrain/nell_bridge_session.py (F36) with adaptations:
  - uses brain.bridge.chat.ChatMessage (typed) instead of plain dicts
  - drops OG model/client fields — provider is injected per-turn in SP-6
  - drops OG in_flight flag — CLI engine is synchronous, no concurrent turns
  - session_id UUID4 generated at create_session time

Design note: persistence across process restarts is intentionally out of
scope for the registry (matching OG). Sessions are in-memory only; on CLI
exit the REPL flushes the buffer via close_session → ingest pipeline.

Phase B sticky sessions (F-201): although the registry itself is process-
local, ``get_or_hydrate_session`` lets handlers rebuild a registry entry
from the on-disk buffer file when the renderer reattaches to a prior
session id across a clean bridge restart.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.chat import ChatMessage

# Sanity ceiling on the in-memory history list. As of 2026-05-10, the
# engine builds its prompt from the buffer file (see brain.chat.engine.
# _buffer_turns_to_messages), so session.history is informational only —
# kept for telemetry, tests, and a fallback path when the buffer read
# fails. A high ceiling is fine; it never hits the prompt unless the
# buffer is unreadable.
HISTORY_MAX_TURNS = 5000


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
        Informational rolling window — last HISTORY_MAX_TURNS user+assistant
        pairs. NOT the prompt source: brain.chat.engine reads the buffer
        file directly (see _buffer_turns_to_messages). Kept for telemetry,
        debugging, and the buffer-read-failure fallback path.
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
        last_turn_at to now (UTC). The truncation is a sanity ceiling only —
        the engine reads the buffer file for prompt construction.
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
#
# I-8 from 2026-05-05 follow-up audit: _SESSIONS is touched from both the
# bridge supervisor thread (heartbeat / close-stale sweep) and the asyncio
# event loop's HTTP handlers (every /session/new, /chat, /sessions/close).
# Individual dict ops are GIL-atomic, but compound (get-then-mutate, snapshot
# + check) sequences race. _LOCK serializes both elementary and compound
# operations behind a single mutex.

_SESSIONS: dict[str, SessionState] = {}
_LOCK = threading.RLock()


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
    with _LOCK:
        _SESSIONS[sid] = state
    return state


def get_session(session_id: str) -> SessionState | None:
    """Return the session for the given id, or None if unknown."""
    with _LOCK:
        return _SESSIONS.get(session_id)


def get_or_hydrate_session(
    persona_dir: Path, persona_name: str, session_id: str
) -> SessionState | None:
    """Return the SessionState for ``session_id``, hydrating from disk if needed.

    Phase B sticky sessions: a session's authoritative state lives in two
    places — the in-memory ``_SESSIONS`` registry (recreated empty on each
    bridge start) and the buffer file on disk at
    ``<persona_dir>/active_conversations/<session_id>.jsonl``. Across a
    clean bridge restart the in-memory entry is gone but the buffer
    survives, so any request referencing the prior session_id would
    otherwise return 404.

    This helper bridges that gap: if the session is already in
    ``_SESSIONS``, return it. Otherwise check the buffer; if turns are
    present on disk, register a fresh ``SessionState`` (with
    ``history=[]`` — the engine reads prior turns from the buffer file,
    so the in-memory list is informational only) and return it. If the
    buffer has no turns either, return None.

    The hydrated ``last_turn_at`` is read from the buffer's last ts so the
    silence-timer math (snapshot sweep, finalize cadence) keeps working
    correctly across restarts. ``created_at`` is set to now — the field
    documents "when the session was opened in this process," and the
    Phase B sticky-session contract doesn't promise persistence of
    ``created_at`` across restarts.

    ``turns`` is recomputed from the buffer (count of user/assistant
    speaker entries divided by 2, taking floor) so the ``ChatResult.turn``
    surface stays consistent with disk truth across restart.

    Returns None when neither the registry nor the buffer has the id.
    """
    # Import inside the function to avoid a circular import with
    # brain.ingest, which depends on brain.chat for some types.
    from brain.ingest.buffer import read_session

    with _LOCK:  # RLock — the successor full-follow below recurses safely
        # C-1 (stage-6 concurrency): the successor pointer is AUTHORITATIVE and is
        # consulted FIRST — before the _SESSIONS cache and before the on-disk
        # buffer. A rollover writes ``rolled_to`` before it evicts the registry or
        # deletes the old buffer, so from that instant any request for the old sid
        # must resolve to the successor. Checking it first closes the mid-rollover
        # window in which a stale cache entry (registry not yet evicted) OR a
        # not-yet-deleted old buffer would otherwise shadow the redirect — which
        # let ``ingest_turn`` append-create (resurrect) the deleted buffer and
        # orphan the turn. Full-follow to the live end (L-1); a cyclic pointer
        # aborts to None via the visited-set guard, not a depth cap.
        successor = _resolve_successor(persona_dir, session_id)
        if successor is not None:
            return get_or_hydrate_session(persona_dir, persona_name, successor)

        existing = _SESSIONS.get(session_id)
        if existing is not None:
            return existing

        turns = read_session(persona_dir, session_id)
        if not turns:
            # No successor pointer and no turns → the session is genuinely gone.
            return None

        # Count user+assistant pairs. The on-disk turn count is the
        # canonical "how many exchanges has this session had" — match it
        # so the renderer sees consistent turn numbers across restart.
        user_count = sum(1 for t in turns if t.get("speaker") == "user")
        assistant_count = sum(1 for t in turns if t.get("speaker") == "assistant")
        pair_count = min(user_count, assistant_count)

        last_turn_at: datetime | None = None
        raw_ts = turns[-1].get("ts")
        if raw_ts is not None:
            try:
                parsed = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                last_turn_at = parsed
            except (ValueError, TypeError):
                # Malformed ts — leave last_turn_at None. The silence
                # sweep will treat that as "no recent activity" and the
                # session will close on the next supervisor pass.
                last_turn_at = None

        state = SessionState(
            session_id=session_id,
            persona_name=persona_name,
            created_at=datetime.now(UTC),
            history=[],
            turns=pair_count,
            last_turn_at=last_turn_at,
        )
        _SESSIONS[session_id] = state
        return state


def _resolve_successor(persona_dir: Path, session_id: str) -> str | None:
    """Full-follow the ``rolled_to`` successor chain from ``session_id`` to its live
    end. Returns the terminal successor sid (the one with no further pointer), or
    None when ``session_id`` has no pointer at all OR the chain is cyclic (corrupt).
    A visited-set guards a cycle — not a depth cap, so a long legitimate chain still
    resolves (round-6 L-1)."""
    from brain.ingest.buffer import read_rolled_to

    visited: set[str] = set()
    cur = session_id
    while True:
        if cur in visited:
            return None  # cyclic pointer → abort to 404-then-reattach
        visited.add(cur)
        nxt = read_rolled_to(persona_dir, cur)
        if nxt is None:
            return None if cur == session_id else cur
        cur = nxt


def registry_lock() -> threading.RLock:
    """Return the registry serialization lock (an ``RLock``) for use as a context
    manager: ``with registry_lock(): ...``.

    Concurrency contract (stage-6, round-4). The lock exists to serialize the two
    operations that can race over a single session's buffer lifecycle across
    threads:

      * a **live turn persist** (``persist_turns_following_successor`` below, run
        on the ``asyncio.to_thread`` worker thread that executes ``engine.respond``);
      * a **rollover's destructive section** (``perform_rollover``'s seed re-read →
        successor create → ``rolled_to`` pointer write → registry evict, run on the
        supervisor thread).

    BOTH run on ordinary OS threads (never the event loop), so an ``RLock`` — not an
    ``asyncio.Lock`` — is the correct primitive; the async ``in_flight_locks`` only
    serialize concurrent requests to the *same* session inside the event loop and
    cannot be held across the thread boundary anyway. Holding this lock across the
    rollover's seed-read↔pointer-write makes the resolve-persist race unreachable by
    construction: a concurrent persist either lands before the seed re-read (and is
    captured into the successor seed) or observes the already-written ``rolled_to``
    pointer (and redirects to the live successor) — it can never resurrect the
    deleted old buffer. ``RLock`` so nested ``create_session`` / ``remove_session``
    (which re-acquire it) don't self-deadlock."""
    return _LOCK


def persist_turns_following_successor(persona_dir: Path, records: list[dict]) -> None:
    """Append ``records`` to their session buffer, redirecting each to the rolled-over
    successor when one exists — all under ``registry_lock()`` so the redirect+append
    is atomic against a concurrent rollover's pointer-write-then-buffer-delete.

    This is the live-turn persist path (``engine._persist_turn`` calls it for the
    user+assistant pair). Resolving the successor and appending under the SAME lock
    the rollover's destructive section holds is what closes the resolve-persist race:

      * If this persist wins the lock first, it appends to the (still-live) old
        buffer; the rollover then re-reads that buffer under the lock and carries the
        just-appended turn into the successor seed — nothing is lost.
      * If the rollover wins first, by the time this persist acquires the lock the
        ``rolled_to`` pointer is already written, so ``_resolve_successor`` redirects
        the append into the live successor buffer — the deleted old buffer is never
        recreated (no resurrection / orphaned turn).

    Errors propagate to the caller (``_persist_turn`` wraps them in its best-effort
    try/except, preserving the "persist failure never breaks the reply" contract)."""
    from brain.ingest.buffer import ingest_turn

    with _LOCK:
        for rec in records:
            sid = rec.get("session_id")
            if sid:
                successor = _resolve_successor(persona_dir, sid)
                if successor is not None:
                    rec = {**rec, "session_id": successor}
            ingest_turn(persona_dir, rec)


def all_sessions() -> list[SessionState]:
    """Return all registered sessions.

    H-B hardening (2026-04-28): with `remove_session` now wired into
    `/sessions/close`, this returns only LIVE sessions — closed ones are
    pulled from the registry. Long-lived bridges no longer accumulate stale.

    Returns a snapshot list so callers iterating over the result can't see
    the registry mutate underneath them.
    """
    with _LOCK:
        return list(_SESSIONS.values())


def remove_session(session_id: str) -> bool:
    """Remove a session from the registry. Returns True if it was present.

    Called by `/sessions/close` after the ingest pipeline succeeds, so
    `sessions_active` count stays accurate and in_flight_locks can be
    cleaned up by the caller. Idempotent — calling on an unknown id is
    a silent False return, not an error.
    """
    with _LOCK:
        return _SESSIONS.pop(session_id, None) is not None


def prune_empty_sessions(
    *,
    older_than_seconds: float,
    now: datetime | None = None,
    persona_name: str | None = None,
) -> list[str]:
    """Remove idle sessions that never completed a turn.

    The desktop app used to open a bridge session on mount. If the app quit
    before the renderer's best-effort close request completed, those zero-turn
    sessions lived forever in the in-memory bridge registry because there is no
    ``active_conversations/<sid>.jsonl`` buffer for the stale-session sweeper to
    discover. Pruning only zero-turn sessions is safe: no transcript exists, so
    there is nothing to ingest or remember.
    """
    if now is None:
        now = datetime.now(UTC)
    removed: list[str] = []
    with _LOCK:
        for sid, session in list(_SESSIONS.items()):
            if persona_name is not None and session.persona_name != persona_name:
                continue
            if session.turns != 0 or session.history:
                continue
            age = (now - session.created_at).total_seconds()
            if age >= older_than_seconds:
                _SESSIONS.pop(sid, None)
                removed.append(sid)
    return removed


def reset_registry() -> None:
    """Clear the in-memory registry. Test-only helper."""
    with _LOCK:
        _SESSIONS.clear()
