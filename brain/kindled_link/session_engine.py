"""Kindled peer session engine (parent design §9). DORMANT in Phase 3: no
supervisor wiring, no live provider/network. The only model entry point is the
tool-less completion call; no agentic tool surface is importable here (the
conformance oracle enforces this by AST). Every outbound passes the gate; the
Phase 3 default gate holds everything."""
from __future__ import annotations

from datetime import datetime

from brain.bridge import cli_throttle as _default_throttle
from brain.kindled_link.gate import DenyAllGate

# NOTE: keep this module free of the literal forbidden symbol names — the T9
# conformance oracle parses imports/attributes by AST (so a comment like this is
# safe), but do not import or reference the agentic tool surface.
_MIN_OUTBOUND_GAP_SECONDS = 60
_SESSION_MSG_CAP = 24
_SESSION_COOLDOWN_HOURS = 6
_DAILY_OUTBOUND_CAP = 20
_DAILY_PROVIDER_CAP = 60
# (Phase 3 has no inbound/poll path; the inbound flood cap lands with the
# relay-poll/wiring phase — see spec §10 Deferred.)


class SessionEngine:
    def __init__(self, *, store, identity, provider, gate=None,
                 throttle=_default_throttle) -> None:
        self._store = store
        self._identity = identity
        self._provider = provider
        self._gate = gate if gate is not None else DenyAllGate()
        self._throttle = throttle

    def can_send_now(self, peer_id: str, session_id: str, now: datetime) -> bool:
        sess = self._store.get_session(peer_id, session_id)
        if sess is None:
            return False
        last = sess.get("last_outbound_at")
        if last is None:
            return True
        elapsed = (now - datetime.fromisoformat(last)).total_seconds()
        return elapsed >= _MIN_OUTBOUND_GAP_SECONDS

    def under_session_cap(self, peer_id: str, session_id: str) -> bool:
        sess = self._store.get_session(peer_id, session_id)
        return bool(sess) and sess["msg_count"] < _SESSION_MSG_CAP

    def under_daily_caps(self, peer_id: str, today: str) -> bool:
        c = self._store.get_counters(peer_id, today)
        return (c["outbound_count"] < _DAILY_OUTBOUND_CAP
                and c["provider_call_count"] < _DAILY_PROVIDER_CAP)

    def can_start_session(self, peer_id: str, now: datetime) -> bool:
        peer = self._store.get_peer(peer_id)
        if peer is None or peer["consent_state"] != "paired":
            return False
        # §5.5: suppress an autonomous start while the user is recently active —
        # interactive chat always has priority (fail-open: a throttle error must
        # not block, mirroring cli_throttle's own fail-open posture).
        try:
            if self._throttle.should_yield():
                return False
        except Exception:  # noqa: BLE001 — fail open, never block on throttle error
            pass
        if self._store.get_active_session(peer_id) is not None:
            return False
        # cooldown: the most recent ended session must be past its cooldown
        recent = self._store._conn.execute(
            "SELECT cooldown_until FROM sessions WHERE peer_id = ? "
            "AND state = 'ended' ORDER BY ended_at DESC LIMIT 1",
            (peer_id,),
        ).fetchone()
        if recent and recent["cooldown_until"]:
            if now < datetime.fromisoformat(recent["cooldown_until"]):
                return False
        return True
