"""Invite-only pairing (design §6). Phase 1: signed single-use invites,
fingerprint verification, and consent transitions, all local (no relay). The
X25519 one-time pairing key + the relay handshake are Phase 2 — deferred."""
from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

from brain.kindled_link.codec import canonical_json
from brain.kindled_link.identity import (
    KindledIdentity,
    fingerprint,
    fingerprint_phrase,
    verify,
)
from brain.kindled_link.store import KindledLinkStore

PROTOCOL = "kindled-link/1"
_INVITE_TTL = timedelta(days=7)


class InviteError(ValueError):
    """A malformed, expired, mis-signed, or already-consumed invite."""


def create_invite(
    idn: KindledIdentity, *, relay_url: str, now: datetime | None = None
) -> dict:
    now = now or datetime.now(UTC)
    body = {
        "protocol": PROTOCOL,
        "invite_id": "inv_" + secrets.token_hex(8),
        "relay_url": relay_url,
        "identity_pub": idn.public_bytes.hex(),
        "fingerprint": idn.key_id,
        "expires_at": (now + _INVITE_TTL).isoformat(),
    }
    return {"body": body, "signature": idn.sign(canonical_json(body)).hex()}


def import_invite(
    invite: dict, *, store: KindledLinkStore, now: datetime | None = None
) -> dict:
    now = now or datetime.now(UTC)
    body = invite["body"]
    sig = bytes.fromhex(invite["signature"])
    pub = bytes.fromhex(body["identity_pub"])

    if body.get("protocol") != PROTOCOL:
        raise InviteError("protocol_mismatch")
    if not verify(pub, sig, canonical_json(body)):
        raise InviteError("bad_signature")
    if fingerprint(pub) != body["fingerprint"]:
        raise InviteError("fingerprint_mismatch")
    if datetime.fromisoformat(body["expires_at"]) < now:
        raise InviteError("expired")
    if store.is_invite_consumed(body["invite_id"]):
        raise InviteError("invite_consumed")

    store.mark_invite_consumed(body["invite_id"], now)
    store.upsert_peer(
        peer_id=body["fingerprint"], identity_pub_hex=pub.hex(),
        fingerprint=body["fingerprint"], consent_state="pending_local",
        relay_url=body["relay_url"], now=now,
    )
    return {
        "peer_id": body["fingerprint"],
        "fingerprint_phrase": fingerprint_phrase(pub),
    }


def confirm_local_fingerprint(
    store: KindledLinkStore, peer_id: str, *, now: datetime | None = None
) -> None:
    """The local user verified the fingerprint phrase: pending_local -> pending_remote."""
    store.set_consent(peer_id, "pending_remote", now or datetime.now(UTC))


def mark_remote_paired(
    store: KindledLinkStore, peer_id: str, *, now: datetime | None = None
) -> None:
    """The remote side confirmed too: pending_remote -> paired (durable consent)."""
    store.set_consent(peer_id, "paired", now or datetime.now(UTC))
