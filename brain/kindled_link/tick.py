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

import json
import logging
from datetime import datetime
from pathlib import Path

from brain.kindled_link.cadence import save_tick_cadence, tick_is_due
from brain.kindled_link.gate import OutboundPayload
from brain.kindled_link.relationship import (
    reflection_is_due,
    run_relationship_reflection,
    save_reflection_cadence,
)
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

    # Single-clock: `today` is derived ONCE from `now` and threaded everywhere
    # (recover, _tick_peer, generate_draft, process_outbound).  No callee
    # recomputes its own `today`; _check_day enforces consistency at the send
    # boundary (T8 Part C — confirmed, no behaviour change needed).
    today = now.strftime("%Y-%m-%d")

    def _recovery_send_factory(peer_id: str, session_id: str):
        """Per-draft transmit closure for engine.recover — binds the draft's own
        peer/session so a re-gated 'send' reaches the right relay mailbox (T5/T6
        review: a single no-op send_fn silently dropped + mis-marked recovered
        sends)."""
        return _make_send_fn(
            store=store, identity=identity, relay_client=relay_client,
            peer_id=peer_id, session_id=session_id, now=now, persona_dir=persona_dir,
        )

    # 3. Inbound — ALWAYS (even when disabled — D4)
    inbound_summary = poll_and_ingest(
        store, identity, relay_client,
        persona_dir=persona_dir,
        now=now,
    )

    degraded: list[str] = []
    peers_processed = 0
    any_recovered = False

    # 4. Recover + autonomous outbound + start — ONLY if enabled (D1). Recovery
    # re-gates half-finished drafts and may transmit, so it is gated too: when
    # disabled, pending drafts stay deferred (not sent), recovered on a later
    # enabled tick.
    if config.kindled_link_enabled:
        recovered_actions = engine.recover(
            now=now, today=today, send_fn_factory=_recovery_send_factory
        )
        any_recovered = any(a not in ("deferred",) for a in recovered_actions)
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
                    provider=provider,
                    peer_id=peer_id,
                    now=now,
                    today=today,
                    persona_dir=persona_dir,
                )
                peers_processed += 1
            except Exception:  # noqa: BLE001 — fault-isolated per-peer
                log.warning("kindled tick: peer %s error", peer_id, exc_info=True)
                degraded.append(peer_id)

    # Recovery indicator (G2): when a half-finished draft was recovered, drop a
    # flag the panel reads (server.py /kindled-link/status). Stage-6 review: the
    # reader existed but nothing wrote it. Fail-soft — a missing flag only hides
    # the banner, never breaks the tick.
    if any_recovered:
        try:
            (persona_dir / "kindled_link").mkdir(parents=True, exist_ok=True)
            (persona_dir / "kindled_link" / "recovered.flag").write_text(
                now.isoformat(), encoding="utf-8")
        except Exception:  # noqa: BLE001 — fail-soft observability flag
            log.warning("kindled tick: could not write recovered.flag", exc_info=True)

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
    provider,
    peer_id: str,
    now: datetime,
    today: str,
    persona_dir: Path,
) -> None:
    """Drive one tick cycle for a single paired peer.

    Checks for an active session with a fresh inbound → gated response.
    Falls back to autonomous START if no active session and can_start_session.
    Fires run_relationship_reflection when its cadence is due (provider call,
    gated by config.kindled_link_enabled — already true at this call site).
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
            rel_row = store.get_relationship_row(peer_id)
            draft = engine.generate_draft(
                peer_id=peer_id,
                session_id=session_id,
                persona_voice="",
                ambient="",
                peer_stage=rel_row["stage"] if rel_row else "stranger",
                transcript_summary=transcript_summary,
                today=today,
            )
            if draft is not None:
                payload = OutboundPayload(body=draft)
                # Persist the draft as 'pending' BEFORE gating so (a) a crash
                # mid-gate leaves a recoverable row (recover reads 'pending'),
                # and (b) the gate outcome is recorded — a 'hold' row feeds the
                # holds panel (views.holds_status queries 'hold') and the H4
                # gradual-regression signal (_count_recent_holds). Stage-6 review:
                # the live path previously never persisted a draft, leaving
                # recover/holds/H4 structurally empty.
                draft_id = store.save_draft(
                    peer_id=peer_id, session_id=session_id,
                    payload_json=json.dumps({"body": draft}), now=now,
                )
                action = engine.process_outbound(
                    peer_id=peer_id,
                    session_id=session_id,
                    payload=payload,
                    reason="peer-response",
                    now=now,
                    today=today,
                    send_fn=send_fn,
                    transcript_summary=transcript_summary,
                )
                store.set_draft_status(draft_id, action)
        return

    # No active session — try autonomous start
    if engine.can_start_session(peer_id, now):
        try:
            leg1 = open_session(store, identity, peer_id=peer_id, now=now)
            relay_client.push(leg1)
        except Exception:  # noqa: BLE001 — fault-isolated
            log.warning("kindled tick: open_session failed for %s", peer_id, exc_info=True)

    # Relationship reflection — run when cadence is due (provider call; caller
    # already confirmed config.kindled_link_enabled before entering _tick_peer).
    if reflection_is_due(persona_dir, now):
        try:
            transcript_rows = store.recent_transcript(peer_id)
            transcript_text = " | ".join(row["text"] for row in reversed(transcript_rows))
            hold_count = _count_recent_holds(store, peer_id)
            run_relationship_reflection(
                store=store,
                provider=provider,
                peer_id=peer_id,
                transcript=transcript_text,
                now=now,
                today=today,
                regression_signal={"hold_count": hold_count},
                persona_dir=persona_dir,
            )
            save_reflection_cadence(persona_dir, now)
        except Exception:  # noqa: BLE001 — fault-isolated; reflection is best-effort
            log.warning("kindled tick: reflection failed for %s", peer_id, exc_info=True)


def _count_recent_holds(store, peer_id: str) -> int:
    """Count 'hold'-status outbound drafts for a peer (fail-soft → 0).

    Uses a direct store query — get_pending_drafts only returns 'pending' rows,
    so we reach into the connection for the 'hold' status specifically.
    """
    try:
        rows = store._conn.execute(
            "SELECT COUNT(*) FROM outbound_drafts WHERE peer_id = ? AND status = 'hold'",
            (peer_id,),
        ).fetchone()
        return int(rows[0]) if rows else 0
    except Exception:  # noqa: BLE001 — fail-soft; 0 means no regression pressure
        return 0


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
