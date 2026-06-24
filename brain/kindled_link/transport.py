"""kindled-link/1 live transport seam.

Outbound:  send_message  — build_envelope → relay push
Inbound:   poll_and_ingest — fetch → group-by-sender → flood-cap →
           verify_and_open/decrypt → transcript → ack

No supervisor/tick wiring here. All callables accept explicit `now`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from brain.kindled_link.audit import log_transport
from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.limits import INBOUND_FLOOD_CAP
from brain.kindled_link.protocol import (
    build_envelope,
    verify_and_open,
)
from brain.kindled_link.relay_client import RelayClient, RelayUnavailableError
from brain.kindled_link.session import complete_session, on_session_open
from brain.kindled_link.store import KindledLinkStore

logger = logging.getLogger(__name__)

_MESSAGE_TTL = timedelta(days=7)

# A sentinel persona_dir for audit logging in contexts where no dir is
# explicitly provided.  Callers that carry a persona_dir should pass it via
# the keyword argument.
_NO_DIR = Path("/dev/null")


def send_message(
    store: KindledLinkStore,
    identity: KindledIdentity,
    relay_client: RelayClient,
    *,
    peer_id: str,
    session_id: str,
    payload: dict,
    now: datetime,
    persona_dir: Path = _NO_DIR,
    ttl: timedelta = _MESSAGE_TTL,
) -> bool:
    """Build, encrypt, sign, and push one message envelope.

    Returns True on success, False on any non-fatal failure (no session key,
    relay unavailable). The caller / tick layer handles retry cadence.
    """
    sk = store.get_session_key(peer_id, session_id)
    if sk is None:
        log_transport(
            persona_dir,
            event="push",
            peer_id=peer_id,
            session_id=session_id,
            relay_ok=False,
            now=now,
        )
        return False

    seq = store.next_outbound_sequence(peer_id, session_id)
    peer = store.get_peer(peer_id)
    mailbox = peer["relay_mailbox"] if peer else ""

    env = build_envelope(
        payload=payload,
        sender=identity,
        recipient_key_id=peer_id,
        relay_mailbox=mailbox,
        session_id=session_id,
        sequence=seq,
        role=sk["my_role"],
        session_key=sk["session_key"],
        now=now,
        ttl=ttl,
    )

    try:
        relay_client.push(env)
    except RelayUnavailableError:
        log_transport(
            persona_dir,
            event="relay_unavailable",
            peer_id=peer_id,
            session_id=session_id,
            relay_ok=False,
            now=now,
        )
        return False

    log_transport(
        persona_dir,
        event="push",
        peer_id=peer_id,
        session_id=session_id,
        seq=seq,
        relay_ok=True,
        now=now,
    )
    return True


def poll_and_ingest(
    store: KindledLinkStore,
    identity: KindledIdentity,
    relay_client: RelayClient,
    *,
    now: datetime,
    persona_dir: Path = _NO_DIR,
) -> dict:
    """Fetch the full mailbox, process each envelope, ack the processed ones.

    Returns a summary dict with per-peer counts and a ``degraded`` set of
    peer_ids that exceeded the flood cap.
    """
    # #48 C-12b: a relay rejection (the new 413/429/404 abuse caps) raises
    # RelayUnavailableError. Guard it here so the whole tick doesn't abort —
    # degrade to an empty poll this round (retried next tick).
    try:
        fetched = relay_client.fetch()
    except RelayUnavailableError:
        log_transport(persona_dir, event="poll_relay_unavailable", count=0, now=now)
        return {"accepted": {}, "degraded": [], "relay_error": True}
    log_transport(persona_dir, event="poll", count=len(fetched), now=now)

    # Group by sender_key_id — flood cap is per peer-group.
    groups: dict[str, list[dict]] = {}
    for item in fetched:
        env = item["envelope"]
        sender = env.get("sender_key_id", "")
        groups.setdefault(sender, []).append(item)

    to_ack: list[str] = []
    degraded: set[str] = set()
    accepted_counts: dict[str, int] = {}

    for sender_id, items in groups.items():
        if len(items) > INBOUND_FLOOD_CAP:
            excess = len(items) - INBOUND_FLOOD_CAP
            log_transport(
                persona_dir,
                event="flood_clamped",
                peer_id=sender_id,
                count=excess,
                now=now,
            )
            degraded.add(sender_id)
            # Process only the first INBOUND_FLOOD_CAP; leave the rest un-acked.
            items = items[:INBOUND_FLOOD_CAP]

        # Within a peer-group, handle session_open (handshake) envelopes before
        # message envelopes (stable — arrival order preserved within each class).
        # Defensive: a message can only decrypt once its session key exists, so a
        # session_open arriving in the same poll must be processed first or the
        # message would hit the no-session drop. Protocol forward-dependency makes
        # this near-unreachable, but the reorder removes the message-loss window
        # entirely (T3/T4 review Q4).
        items = sorted(items, key=lambda it: 0 if "session_open" in it["envelope"] else 1)

        for item in items:
            env_id = item["id"]
            env = item["envelope"]
            session_id = env.get("session_id", "")

            # --- session_open handshake envelopes (signed, not encrypted) ---
            if "session_open" in env:
                pending = store.get_pending_handshake(sender_id, session_id)
                if pending is None:
                    # We have no pending handshake → we are the RESPONDER (leg 2)
                    reply = on_session_open(
                        store, identity, envelope=env, now=now
                    )
                    if reply is not None:
                        try:
                            relay_client.push(reply)
                        except RelayUnavailableError:
                            log_transport(
                                persona_dir,
                                event="relay_unavailable",
                                peer_id=sender_id,
                                session_id=session_id,
                                now=now,
                            )
                else:
                    # We have a pending handshake → we are the INITIATOR (leg 3)
                    complete_session(store, identity, reply_envelope=env, now=now)
                to_ack.append(env_id)
                continue

            # --- regular message envelope ---
            sk = store.get_session_key(sender_id, session_id)
            if sk is None:
                log_transport(
                    persona_dir,
                    event="inbound_no_session",
                    peer_id=sender_id,
                    session_id=session_id,
                    now=now,
                )
                to_ack.append(env_id)  # terminal — would re-poll forever
                continue

            peer = store.get_peer(sender_id)
            sender_pub = bytes.fromhex(peer["identity_pub"]) if peer else b""
            hw = store.get_seq_high_water(sender_id, session_id)

            payload, reason = verify_and_open(
                env,
                recipient=identity,
                sender_pub=sender_pub,
                session_key=sk["session_key"],
                sender_role=sk["peer_role"],
                seq_high_water=hw,
                now=now,
            )

            if reason is not None:
                log_transport(
                    persona_dir,
                    event="inbound_rejected",
                    peer_id=sender_id,
                    session_id=session_id,
                    reject_reason=reason.name,
                    now=now,
                )
                to_ack.append(env_id)  # terminal — replay/expired/tampered
                continue

            seq = int(env["sequence"])
            text = payload.get("text", "")
            store.append_transcript(
                peer_id=sender_id,
                session_id=session_id,
                seq=seq,
                direction="inbound",
                text=text,
                now=now,
                provenance="peer",
            )
            store.set_seq_high_water(sender_id, session_id, seq)
            log_transport(
                persona_dir,
                event="inbound_accepted",
                peer_id=sender_id,
                session_id=session_id,
                seq=seq,
                now=now,
            )
            accepted_counts[sender_id] = accepted_counts.get(sender_id, 0) + 1
            to_ack.append(env_id)

    if to_ack:
        relay_client.ack(to_ack)

    return {
        "accepted": accepted_counts,
        "degraded": list(degraded),
    }
