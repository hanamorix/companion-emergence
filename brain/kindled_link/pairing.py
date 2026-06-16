"""Invite-only pairing (design §6). Phase 1: signed single-use invites,
fingerprint verification, and consent transitions, all local (no relay). The
X25519 one-time pairing key + the relay handshake are Phase 2 — deferred."""
from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

from brain.kindled_link.codec import canonical_json
from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.store import KindledLinkStore  # noqa: F401  (used in T8/T9)

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
