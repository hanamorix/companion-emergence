"""Kindled-link supervisor tick (Phase 7a T6).

run_kindled_link_tick — the activation centrepiece:
  1. Cadence gate (persisted, 5-min interval).
  2. Recover half-finished drafts (re-gates, never blind-resends).
  3. Inbound poll+ingest — ALWAYS, even when disabled (D4).
  4. Autonomous outbound + autonomous start — ONLY if kindled_link_enabled (D1).
  5. Fault-isolated per-peer loop (one peer error cannot abort the tick).

All callables take explicit ``now`` — no internal clock.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from brain.kindled_link.cadence import save_tick_cadence, tick_is_due
from brain.kindled_link.gate import OutboundPayload
from brain.kindled_link.session import open_session
from brain.kindled_link.session_engine import SessionEngine
from brain.kindled_link.transport import poll_and_ingest, send_message

log = logging.getLogger(__name__)


def run_kindled_link_tick(
    persona_dir,
    *,
    store,
    identity,
    relay_client,
    provider,
    config,
    now: datetime,
    gate=None,
    throttle=None,
) -> dict:
    """Drive one kindled-link tick.

    Returns a summary dict:
      ``{"skipped": True}``                    — cadence gate not yet due
      ``{"peers_processed": n, "inbound": …,
          "recovered": bool, "degraded": […]}`` — full tick ran
    """
    persona_dir = Path(persona_dir)

    # 1. Cadence gate
    if not tick_is_due(persona_dir, now):
        return {"skipped": True}

    # 2. Recover half-finished outbound drafts (re-gates each; never blind-resend)
    engine_kwargs: dict = {"store": store, "identity": identity, "provider": provider}
    if gate is not None:
        engine_kwargs["gate"] = gate
    if throttle is not None:
        engine_kwargs["throttle"] = throttle
    engine = SessionEngine(**engine_kwargs)

    today = now.strftime("%Y-%m-%d")

    def _send_closure_for(peer_id: str, session_id: str):
        """Return a send_fn bound to this peer/session."""
        def _send(payload: OutboundPayload) -> None:
            send_message(
                store, identity, relay_client,
                peer_id=peer_id,
                session_id=session_id,
                payload={"text": payload.body},
                now=now,
                persona_dir=persona_dir,
            )
        return _send

    # Build a generic send closure for recover (it reads peer/session from the draft)
    def _recovery_send(payload: OutboundPayload) -> None:
        # recover() drives its own per-draft send_fn; this outer closure is the
        # fallback for the engine.recover() call signature.  The engine's recover
        # method calls process_outbound internally with draft-specific peer/session,
        # so we only need to supply *a* send_fn at this level.
        pass  # engine.recover calls process_outbound which passes its own send_fn

    recovered_actions = engine.recover(now=now, today=today, send_fn=_recovery_send)
    any_recovered = any(a not in ("deferred",) for a in recovered_actions)

    # 3. Inbound — ALWAYS (even when disabled — D4)
    inbound_summary = poll_and_ingest(
        store, identity, relay_client,
        persona_dir=persona_dir,
        now=now,
    )

    degraded: list[str] = []
    peers_processed = 0

    # 4. Autonomous outbound + start — ONLY if enabled (D1)
    if config.kindled_link_enabled:
        for peer_id in store.list_paired_peers():
            peer = store.get_peer(peer_id)
            if peer is None:
                continue
            cs = peer.get("consent_state", "")
            if cs in ("paused", "revoked", "blocked"):
                continue

            try:
                _tick_peer(
                    store=store,
                    identity=identity,
                    relay_client=relay_client,
                    engine=engine,
                    peer_id=peer_id,
                    now=now,
                    today=today,
                    persona_dir=persona_dir,
                )
                peers_processed += 1
            except Exception:  # noqa: BLE001 — fault-isolated per-peer
                log.warning("kindled tick: peer %s error", peer_id, exc_info=True)
                degraded.append(peer_id)

    # 5. Persist cadence timestamp
    save_tick_cadence(persona_dir, now)

    return {
        "peers_processed": peers_processed,
        "inbound": inbound_summary,
        "recovered": any_recovered,
        "degraded": degraded,
    }


def _tick_peer(
    *,
    store,
    identity,
    relay_client,
    engine: SessionEngine,
    peer_id: str,
    now: datetime,
    today: str,
    persona_dir: Path,
) -> None:
    """Drive one tick cycle for a single paired peer.

    Checks for an active session with a fresh inbound → gated response.
    Falls back to autonomous START if no active session and can_start_session.
    """
    active_sess = store.get_active_session(peer_id)

    if active_sess is not None:
        session_id = active_sess["session_id"]
        # Check for fresh inbound (any inbound transcript entry we haven't replied to)
        transcript = store.recent_transcript(peer_id)
        has_fresh_inbound = any(
            row["direction"] == "inbound" for row in transcript
        )

        if has_fresh_inbound:
            transcript_summary = " | ".join(
                row["text"] for row in reversed(transcript)
            )
            send_fn = _make_send_fn(
                store=store, identity=identity, relay_client=relay_client,
                peer_id=peer_id, session_id=session_id,
                now=now, persona_dir=persona_dir,
            )
            # generate_draft reads relationship stage + ambient
            draft = engine.generate_draft(
                peer_id=peer_id,
                session_id=session_id,
                persona_voice="",
                ambient="",
                peer_stage=store.get_relationship_row(peer_id)["stage"]
                if store.get_relationship_row(peer_id) else "stranger",
                transcript_summary=transcript_summary,
                today=today,
            )
            if draft is not None:
                payload = OutboundPayload(body=draft)
                engine.process_outbound(
                    peer_id=peer_id,
                    session_id=session_id,
                    payload=payload,
                    reason="peer-response",
                    now=now,
                    today=today,
                    send_fn=send_fn,
                    transcript_summary=transcript_summary,
                )
        return

    # No active session — try autonomous start
    if engine.can_start_session(peer_id, now):
        try:
            leg1 = open_session(store, identity, peer_id=peer_id, now=now)
            relay_client.push(leg1)
        except Exception:  # noqa: BLE001 — fault-isolated
            log.warning("kindled tick: open_session failed for %s", peer_id, exc_info=True)


def _make_send_fn(*, store, identity, relay_client, peer_id, session_id, now, persona_dir):
    """Return a send_fn(OutboundPayload) closure bound to this peer/session."""
    def send_fn(payload: OutboundPayload) -> None:
        send_message(
            store, identity, relay_client,
            peer_id=peer_id,
            session_id=session_id,
            payload={"text": payload.body},
            now=now,
            persona_dir=persona_dir,
        )
    return send_fn
