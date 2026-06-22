"""kindled-link/1 three-leg mutual X25519 session handshake (Phase 7a T2.5).

Leg 1 — open_session   (INITIATOR): mints session_id, generates ephemeral,
                                     persists pending_handshakes, returns leg-1 envelope.
Leg 2 — on_session_open (RESPONDER): verifies leg-1, derives session key, persists it,
                                      returns leg-2 reply envelope.
Leg 3 — complete_session (INITIATOR): verifies leg-2, derives matching session key,
                                       persists it, clears pending_handshakes.

CRITICAL INVARIANT: both sides call derive_session_key with the *initiator's*
bootstrap_nonce as the HKDF salt, so they produce the same 32-byte key.

No transport/relay calls here — legs return envelopes; the tick/transport task pushes them.
All functions take an explicit `now` datetime (no internal clock).
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.protocol import (
    ROLE_INITIATOR,
    ROLE_RESPONDER,
    build_session_open,
    derive_session_key,
    generate_ephemeral,
    parse_session_open,
)
from brain.kindled_link.store import KindledLinkStore

# Default TTL for session_open envelopes (7 days, matching the spec examples).
_SESSION_OPEN_TTL = timedelta(days=7)


def open_session(
    store: KindledLinkStore,
    identity: KindledIdentity,
    *,
    peer_id: str,
    now: datetime,
    ttl: timedelta = _SESSION_OPEN_TTL,
) -> dict:
    """INITIATOR — leg 1.

    Mint a fresh session_id, generate an ephemeral keypair, persist a
    pending_handshakes row, and return the signed leg-1 session_open envelope.
    The clobber guard ensures we never overwrite an already-established session key.
    """
    session_id = "ks_" + secrets.token_hex(8)

    # Clobber guard: refuse if a session_keys row already exists.
    # (With a fresh random id this should never trigger, but the guard must exist.)
    if store.get_session_key(peer_id, session_id) is not None:
        raise ValueError(
            f"session_keys row already exists for ({peer_id!r}, {session_id!r})"
        )

    eph = generate_ephemeral()
    eph_pub_raw = eph.public_key().public_bytes_raw()
    bootstrap_nonce = secrets.token_bytes(16)

    store.save_pending_handshake(
        peer_id=peer_id,
        session_id=session_id,
        my_eph_priv_raw=eph.private_bytes_raw(),
        bootstrap_nonce=bootstrap_nonce,
        my_role=ROLE_INITIATOR,
        now=now,
    )

    peer = store.get_peer(peer_id)
    relay_mailbox = peer["relay_mailbox"] if peer else ""
    sender_mailbox = store.get_or_create_local_mailbox()

    return build_session_open(
        sender=identity,
        recipient_key_id=peer_id,
        relay_mailbox=relay_mailbox,
        session_id=session_id,
        ephemeral_pub=eph_pub_raw,
        bootstrap_nonce=bootstrap_nonce,
        sender_mailbox=sender_mailbox,
        now=now,
        ttl=ttl,
    )


def on_session_open(
    store: KindledLinkStore,
    identity: KindledIdentity,
    *,
    envelope: dict,
    now: datetime,
    ttl: timedelta = _SESSION_OPEN_TTL,
) -> dict | None:
    """RESPONDER — leg 2.

    Verify the initiator's leg-1 envelope, derive the session key, persist it,
    create the sessions row, and return the leg-2 reply envelope.
    Returns None on any reject (unknown peer, bad signature, already established).
    """
    initiator_peer_id = envelope.get("sender_key_id")
    session_id = envelope.get("session_id")

    # Look up initiator's public key from our peer store.
    peer = store.get_peer(initiator_peer_id) if initiator_peer_id else None
    if peer is None:
        return None  # Unknown sender — drop

    sender_pub = bytes.fromhex(peer["identity_pub"])
    parsed, reason = parse_session_open(envelope, sender_pub=sender_pub, now=now)
    if reason is not None:
        return None  # Reject (bad sig / expired / malformed)

    # Clobber guard: if already established, return None (idempotent ignore).
    if store.get_session_key(initiator_peer_id, session_id) is not None:
        return None

    # Learn the initiator's mailbox from the signed envelope.
    initiator_mailbox = parsed.get("sender_mailbox") or ""
    if initiator_mailbox:
        store.upsert_peer(
            peer_id=initiator_peer_id,
            identity_pub_hex=peer["identity_pub"],
            fingerprint=peer["fingerprint"],
            consent_state=peer["consent_state"],
            relay_url=peer["relay_url"],
            relay_mailbox=initiator_mailbox,
            now=now,
        )

    # Generate our own ephemeral and derive the session key.
    my_eph = generate_ephemeral()
    initiator_eph_pub = bytes.fromhex(parsed["ephemeral_pub"])
    # The salt MUST be the initiator's bootstrap_nonce (both sides use the same one).
    bootstrap_nonce = bytes.fromhex(parsed["bootstrap_nonce"])

    session_key = derive_session_key(
        my_eph,
        initiator_eph_pub,
        sender_fp=identity.key_id,
        recipient_fp=initiator_peer_id,
        session_id=session_id,
        bootstrap_nonce=bootstrap_nonce,
    )

    store.save_session_key(
        peer_id=initiator_peer_id,
        session_id=session_id,
        session_key=session_key,
        my_role=ROLE_RESPONDER,
        peer_role=ROLE_INITIATOR,
        now=now,
    )
    store.create_session(initiator_peer_id, session_id, now)

    # Build and return the leg-2 reply carrying our ephemeral + the SAME bootstrap_nonce.
    sender_mailbox = store.get_or_create_local_mailbox()
    reply_mailbox = initiator_mailbox or (store.get_peer(initiator_peer_id) or {}).get("relay_mailbox", "")

    return build_session_open(
        sender=identity,
        recipient_key_id=initiator_peer_id,
        relay_mailbox=reply_mailbox,
        session_id=session_id,
        ephemeral_pub=my_eph.public_key().public_bytes_raw(),
        bootstrap_nonce=bootstrap_nonce,  # SAME initiator bootstrap_nonce — critical
        sender_mailbox=sender_mailbox,
        now=now,
        ttl=ttl,
    )


def complete_session(
    store: KindledLinkStore,
    identity: KindledIdentity,
    *,
    reply_envelope: dict,
    now: datetime,
) -> None:
    """INITIATOR — leg 3.

    Verify the responder's leg-2 reply, derive the matching session key using the
    stored pending ephemeral private key + the original bootstrap_nonce, persist it,
    create the sessions row, and clear the pending_handshakes row.
    """
    responder_peer_id = reply_envelope.get("sender_key_id")
    session_id = reply_envelope.get("session_id")

    peer = store.get_peer(responder_peer_id) if responder_peer_id else None
    if peer is None:
        return  # Unknown peer — nothing to complete

    sender_pub = bytes.fromhex(peer["identity_pub"])
    parsed, reason = parse_session_open(reply_envelope, sender_pub=sender_pub, now=now)
    if reason is not None:
        return  # Bad reply — ignore

    pending = store.get_pending_handshake(responder_peer_id, session_id)
    if pending is None:
        return  # No pending handshake — nothing to complete

    # Initiator-side clobber guard (mirrors on_session_open): if a key is already
    # established for this session, a replayed/interrupted leg-3 must drop without
    # re-deriving or re-creating the sessions row (T2.5 review).
    if store.get_session_key(responder_peer_id, session_id) is not None:
        return

    my_eph_priv = X25519PrivateKey.from_private_bytes(pending["my_eph_priv_raw"])
    responder_eph_pub = bytes.fromhex(parsed["ephemeral_pub"])
    # Use the INITIATOR's original bootstrap_nonce (stored in pending row).
    bootstrap_nonce = pending["bootstrap_nonce"]

    session_key = derive_session_key(
        my_eph_priv,
        responder_eph_pub,
        sender_fp=identity.key_id,
        recipient_fp=responder_peer_id,
        session_id=session_id,
        bootstrap_nonce=bootstrap_nonce,
    )

    store.save_session_key(
        peer_id=responder_peer_id,
        session_id=session_id,
        session_key=session_key,
        my_role=ROLE_INITIATOR,
        peer_role=ROLE_RESPONDER,
        now=now,
    )
    store.create_session(responder_peer_id, session_id, now)
    store.clear_pending_handshake(responder_peer_id, session_id)
