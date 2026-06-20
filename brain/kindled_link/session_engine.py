"""Kindled peer session engine (parent design §9). DORMANT in Phase 3: no
supervisor wiring, no live provider/network. The only model entry point is the
tool-less completion call; no agentic tool surface is importable here (the
conformance oracle enforces this by AST). Every outbound passes the gate; the
Phase 3 default gate holds everything."""
from __future__ import annotations

import logging
from datetime import datetime

from brain.bridge import cli_throttle as _default_throttle
from brain.kindled_link.gate import DenyAllGate
from brain.kindled_link.peer_prompt import build_peer_prompt

log = logging.getLogger(__name__)

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

    def generate_draft(
        self, *, peer_id: str, session_id: str, persona_voice: str,
        ambient: str, peer_stage: str, transcript_summary: str, today: str,
    ) -> str | None:
        # Daily provider-call cap (parent §9): GENERATION itself is bounded, not
        # only sends — else background draft calls escape the 60/day cap
        # (re-red-team Major B). Returns None when spent; caller defers.
        if (self._store.get_counters(peer_id, today)["provider_call_count"]
                >= _DAILY_PROVIDER_CAP):
            return None
        prompt = build_peer_prompt(
            persona_voice=persona_voice, ambient=ambient,
            peer_stage=peer_stage, transcript_summary=transcript_summary,
        )
        with self._throttle.background_slot() as granted:
            if not granted:
                return None
            draft = self._provider.complete(prompt)
        self._store.incr_provider_count(peer_id, today)
        return draft

    def _send_allowed(self, peer_id: str, session_id: str, now: datetime,
                      today: str) -> bool:
        """A draft may be sent only if its session is OPEN and pacing + caps
        permit. Shared by process_outbound (live path) and recover (so a pacing-
        pre-empted draft is deferred, not dropped)."""
        sess = self._store.get_session(peer_id, session_id)
        if sess is None or sess["state"] != "open":
            return False
        return (
            self.can_send_now(peer_id, session_id, now)
            and self.under_session_cap(peer_id, session_id)
            and self.under_daily_caps(peer_id, today)
        )

    def process_outbound(
        self, *, peer_id: str, session_id: str, payload, reason: str,
        now: datetime, today: str, send_fn, transcript_summary: str = "",
    ) -> str:
        """Review payload via the gate; send only on 'send'. Any gate failure
        fails closed to 'hold' — nothing leaves ungated (parent §5 inv. 127/134).

        Pacing/cap guard runs FIRST (via _send_allowed): if the session is not
        open, the per-message gap is unmet, or a cap is exhausted, the payload is
        held without calling the gate or send_fn. This keeps recovery (T8) and any
        future real gate from firing a burst of sends that bypasses §5.3 — the
        predicates are wired into the send path, not merely unit-tested (red-team
        Major #2 + re-red-team session-state Minor)."""
        if not self._send_allowed(peer_id, session_id, now, today):
            return "hold"
        peer = self._store.get_peer(peer_id)
        stage = (peer or {}).get("consent_state", "stranger")
        try:
            decision = self._gate.review(
                payload, peer_id=peer_id, stage=stage,
                transcript_summary=transcript_summary, reason=reason,
            )
            action = decision.action
        except Exception:  # noqa: BLE001 — fail closed
            log.warning("kindled gate raised; holding (fail-closed)", exc_info=True)
            action = "hold"

        if action == "send":
            send_fn(payload)
            self._store.bump_session_outbound(peer_id, session_id, now)
            self._store.incr_outbound_count(peer_id, today)
        # hold / revise / end_or_pause: nothing leaves this phase (DenyAllGate
        # never returns 'send'; a real gate's revise/end handling lands in Phase 4)
        return action
