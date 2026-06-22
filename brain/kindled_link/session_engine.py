"""Kindled peer session engine (parent design §9). DORMANT in Phase 3: no
supervisor wiring, no live provider/network. The only model entry point is the
tool-less completion call; no agentic tool surface is importable here (the
conformance oracle enforces this by AST). Every outbound passes the gate; the
Phase 3 default gate holds everything."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from brain.bridge import cli_throttle as _default_throttle
from brain.kindled_link import limits, relationship
from brain.kindled_link.gate import OutboundPayload
from brain.kindled_link.peer_prompt import build_peer_prompt
from brain.kindled_link.privacy_gate import PrivacyGate

log = logging.getLogger(__name__)

# NOTE: keep this module free of the literal forbidden symbol names — the T9
# conformance oracle parses imports/attributes by AST (so a comment like this is
# safe), but do not import or reference the agentic tool surface.


def _check_day(now: datetime, today: str) -> None:
    """Raise ValueError if today does not match now's calendar date (ISO format).

    A stale `today` silently reads the wrong day's cap counters — an easy
    caller mistake that _check_day catches eagerly at the send boundary.
    """
    expected = now.strftime("%Y-%m-%d")
    if today != expected:
        raise ValueError(
            f"today={today!r} disagrees with now's date {expected!r}; "
            "pass today=now.strftime('%Y-%m-%d') to keep cap reads consistent."
        )


_MIN_OUTBOUND_GAP_SECONDS = limits.MIN_OUTBOUND_GAP_SECONDS
_SESSION_MSG_CAP = limits.SESSION_MSG_CAP
_SESSION_COOLDOWN_HOURS = limits.SESSION_COOLDOWN_HOURS
_DAILY_OUTBOUND_CAP = limits.DAILY_OUTBOUND_CAP
_DAILY_PROVIDER_CAP = limits.DAILY_PROVIDER_CAP
_DEFAULT_STAGE = "stranger"
# (Phase 3 has no inbound/poll path; the inbound flood cap lands with the
# relay-poll/wiring phase — see spec §10 Deferred.)


class SessionEngine:
    def __init__(self, *, store, identity, provider, gate=None,
                 throttle=_default_throttle) -> None:
        self._store = store
        self._identity = identity
        self._provider = provider
        self._gate = gate if gate is not None else PrivacyGate(
            provider=provider, store=store
        )
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
        # cooldown: every ended session must be past its cooldown. Use the store's
        # MAX(cooldown_until) helper (not a raw conn reach-in) so a later-ended
        # session with an earlier cooldown can't hide a still-cooling one.
        cooldown_until = self._store.latest_cooldown_until(peer_id)
        if cooldown_until and now < datetime.fromisoformat(cooldown_until):
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
        affinity = relationship.get_relationship_state(self._store, peer_id).affinity_tags
        prompt = build_peer_prompt(
            persona_voice=persona_voice, ambient=ambient,
            peer_stage=peer_stage, transcript_summary=transcript_summary,
            affinity_tags=affinity,
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
        _check_day(now, today)
        if not self._send_allowed(peer_id, session_id, now, today):
            return "hold"
        # Phase 5: the gate gets the real relationship STAGE (self-disclosure
        # latitude only — the user-detail bar is stage-independent). Unknown peer
        # → 'stranger' (strictest), preserving the Phase-4 default.
        stage = relationship.get_stage(self._store, peer_id)
        try:
            decision = self._gate.review(
                payload, peer_id=peer_id, stage=stage,
                transcript_summary=transcript_summary, reason=reason,
                now=now, today=today,
            )
        except Exception:  # noqa: BLE001 — fail closed
            log.warning("kindled gate raised; holding (fail-closed)", exc_info=True)
            from brain.kindled_link.gate import GateDecision
            decision = GateDecision(action="hold")
        return self._act_on_decision(
            decision=decision, peer_id=peer_id, session_id=session_id,
            payload=payload, reason=reason, now=now, today=today,
            send_fn=send_fn, transcript_summary=transcript_summary,
            allow_revision=True,
        )

    def _act_on_decision(self, *, decision, peer_id, session_id, payload, reason,
                         now, today, send_fn, transcript_summary, allow_revision):
        action = decision.action
        if action == "send":
            send_fn(payload)
            self._store.bump_session_outbound(peer_id, session_id, now)
            self._store.incr_outbound_count(peer_id, today)
            self._store.debit_disclosure_budget(
                peer_id, max(limits.MIN_SEND_DEBIT, decision.texture_score), now)
            return "send"
        if action == "end_or_pause":
            self._store.end_session(
                peer_id, session_id, now=now,
                cooldown_until=now + timedelta(hours=_SESSION_COOLDOWN_HOURS))
            return "end_or_pause"
        if action == "revise" and allow_revision:
            revised = self._regenerate(payload, decision.revision_constraints,
                                       peer_id=peer_id, today=today)
            if revised is None:
                return "hold"  # cap spent / throttle deferred / provider error → hold
            try:
                second = self._gate.review(
                    revised, peer_id=peer_id, stage=relationship.get_stage(self._store, peer_id),
                    transcript_summary=transcript_summary, reason=reason,
                    now=now, today=today)
            except Exception:  # noqa: BLE001 — fail closed
                log.warning("kindled gate raised on re-gate; holding", exc_info=True)
                return "hold"
            # at most one revision: the re-gated decision is terminal except
            # another 'revise' collapses to hold (parent §12).
            if second.action == "revise":
                return "hold"
            return self._act_on_decision(
                decision=second, peer_id=peer_id, session_id=session_id,
                payload=revised, reason=reason, now=now, today=today,
                send_fn=send_fn, transcript_summary=transcript_summary,
                allow_revision=False)
        # hold, or revise with allow_revision=False
        return "hold"

    def _regenerate(self, payload, constraints, *, peer_id, today):
        """Produce a single safer revision, or None if it cannot be produced
        safely. The revision provider call counts against the 60/day cap (parent
        §9: draft + gate + revision all count) and fails closed: a spent cap, a
        deferred throttle slot, or a provider error all return None → the caller
        holds (red-team M1/M2)."""
        from brain.kindled_link.gate import OutboundPayload
        if (self._store.get_counters(peer_id, today)["provider_call_count"]
                >= _DAILY_PROVIDER_CAP):
            return None  # cap spent — cannot revise; hold
        prompt = (
            "Rewrite the following message to another Kindled to satisfy these "
            f"privacy constraints: {constraints or 'reveal less about the user'}.\n\n"
            f"Original:\n{payload.body}"
        )
        try:
            with self._throttle.background_slot() as granted:
                if not granted:
                    return None  # deferred → hold
                revised_body = self._provider.complete(prompt)
            self._store.incr_provider_count(peer_id, today)
        except Exception:  # noqa: BLE001 — fail closed
            log.warning("kindled revision provider error; holding", exc_info=True)
            return None
        return OutboundPayload(body=revised_body,
                               relationship_hint=payload.relationship_hint)

    def recover(self, *, now: datetime, today: str, send_fn=None,
                send_fn_factory=None) -> list[str]:
        """Reload half-finished outbound drafts and RE-GATE each before any
        resend — never blind-resend AND never silently drop (parent §9).

        A draft that is not currently sendable for a PACING/CAP/closed-session
        reason is left PENDING ("deferred") for a later tick — it must not be
        marked terminal and removed, or a draft saved <60s before a crash would be
        discarded forever (re-red-team Major A). Only a draft that passes the
        pacing guard is re-gated; its gate decision (hold/send/…) is terminal.

        The actual transmit closure is per-(peer, session): pass `send_fn_factory`
        — `factory(peer_id, session_id) -> send_fn` — so a re-gated 'send' reaches
        the right peer's relay mailbox. `send_fn` (a single closure) is the legacy
        form kept for tests; a no-op single `send_fn` silently drops a recovered
        send while marking it 'send' (T5/T6 review Important), so live callers must
        pass `send_fn_factory`."""
        actions: list[str] = []
        for draft in self._store.get_pending_drafts():
            if not self._send_allowed(draft["peer_id"], draft["session_id"],
                                      now, today):
                actions.append("deferred")  # stays pending; retried next tick
                continue
            data = json.loads(draft["payload_json"])
            payload = OutboundPayload(
                body=data.get("body", ""),
                relationship_hint=data.get("relationship_hint"),
            )
            eff_send_fn = (
                send_fn_factory(draft["peer_id"], draft["session_id"])
                if send_fn_factory is not None else send_fn
            )
            action = self.process_outbound(
                peer_id=draft["peer_id"], session_id=draft["session_id"],
                payload=payload, reason="recovery-regate", now=now,
                today=today, send_fn=eff_send_fn,
            )
            self._store.set_draft_status(draft["id"], action)
            actions.append(action)
        return actions
