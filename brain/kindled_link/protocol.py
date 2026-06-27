"""kindled-link/1 wire protocol — canonical signing input, Ed25519 signatures,
X25519+HKDF session keys, ChaCha20-Poly1305 with a sequence-derived nonce, and
the receiver reject rules. Pure crypto + dict assembly; no I/O. See
docs/superpowers/specs/2026-06-15-kindled-to-kindled-protocol.md."""
from __future__ import annotations

import enum
import json
from datetime import UTC, datetime, timedelta

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from brain.kindled_link.codec import canonical_json
from brain.kindled_link.identity import KindledIdentity, fingerprint, verify

SUPPORTED_PROTOCOL = "kindled-link/1"

# Outer-envelope fields excluded from the AAD (bound everything else).
_AAD_EXCLUDE = ("ciphertext", "signature")
# Outer-envelope field excluded from the signing input.
_SIG_EXCLUDE = ("signature",)

_SKEW = timedelta(minutes=5)


def _without(outer: dict, keys) -> dict:
    return {k: v for k, v in outer.items() if k not in keys}


def aad_bytes(outer: dict) -> bytes:
    """Canonical AAD: the outer envelope minus ciphertext + signature."""
    return canonical_json(_without(outer, _AAD_EXCLUDE))


def sig_input_bytes(outer: dict) -> bytes:
    """Canonical signing input: the outer envelope minus the signature field
    (includes ciphertext)."""
    return canonical_json(_without(outer, _SIG_EXCLUDE))


def sign_envelope(outer: dict, idn: KindledIdentity) -> str:
    """Ed25519-sign sig_input_bytes(outer); return lowercase hex."""
    return idn.sign(sig_input_bytes(outer)).hex()


def verify_envelope_signature(outer: dict, sender_pub: bytes) -> bool:
    """Verify outer['signature'] (hex) over sig_input_bytes(outer). Fail-soft."""
    sig_hex = outer.get("signature")
    if not isinstance(sig_hex, str):
        return False
    try:
        sig = bytes.fromhex(sig_hex)
    except ValueError:
        return False
    return verify(sender_pub, sig, sig_input_bytes(outer))


def generate_ephemeral() -> X25519PrivateKey:
    """A fresh per-session X25519 ephemeral key pair."""
    return X25519PrivateKey.generate()


def derive_session_key(
    my_eph_priv: X25519PrivateKey,
    peer_eph_pub: bytes,
    *,
    sender_fp: str,
    recipient_fp: str,
    session_id: str,
    bootstrap_nonce: bytes,
) -> bytes:
    """HKDF-SHA256 over the ECDH shared secret (protocol §5). The info string
    sorts the two fingerprints, so both sides derive the identical key
    regardless of who initiated."""
    shared = my_eph_priv.exchange(X25519PublicKey.from_public_bytes(peer_eph_pub))
    fps = sorted([sender_fp, recipient_fp])
    info = b"kindled-link/1|" + fps[0].encode() + b"|" + fps[1].encode() + b"|" + session_id.encode()
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=bootstrap_nonce, info=info).derive(shared)


# role bytes (protocol §6)
ROLE_INITIATOR = 0
ROLE_RESPONDER = 1


def aead_nonce(role: int, sequence: int) -> bytes:
    """96-bit nonce = role_byte(1) ‖ sequence_be(11). Uniqueness is structural
    (distinct senders use distinct role bytes; a sender never repeats a
    sequence under one session key). NEVER use a random nonce here."""
    if role not in (ROLE_INITIATOR, ROLE_RESPONDER):
        raise ValueError(f"invalid role: {role}")
    if sequence < 0 or sequence >= (1 << 88):
        raise ValueError(f"sequence out of range: {sequence}")
    return bytes([role]) + sequence.to_bytes(11, "big")


def encrypt_payload(payload: dict, *, session_key: bytes, role: int, sequence: int, aad: bytes) -> str:
    """ChaCha20-Poly1305 over canonical_json(payload); return ciphertext+tag hex."""
    nonce = aead_nonce(role, sequence)
    ct = ChaCha20Poly1305(session_key).encrypt(nonce, canonical_json(payload), aad)
    return ct.hex()


def decrypt_payload(ciphertext_hex: str, *, session_key: bytes, role: int, sequence: int, aad: bytes) -> dict:
    """Decrypt + tag-verify; parse the canonical JSON payload. Raises on failure
    (caller maps to RejectReason.AEAD_FAILURE)."""
    nonce = aead_nonce(role, sequence)
    pt = ChaCha20Poly1305(session_key).decrypt(nonce, bytes.fromhex(ciphertext_hex), aad)
    return json.loads(pt.decode("utf-8"))


class RejectReason(enum.Enum):
    PROTOCOL_MISMATCH = "protocol_mismatch"
    BAD_SIGNATURE = "bad_signature"
    WRONG_RECIPIENT = "wrong_recipient"
    AEAD_FAILURE = "aead_failure"
    REPLAY = "replay"
    EXPIRED = "expired"


def build_envelope(
    *,
    payload: dict,
    sender: KindledIdentity,
    recipient_key_id: str,
    relay_mailbox: str,
    session_id: str,
    sequence: int,
    role: int,
    session_key: bytes,
    now: datetime,
    ttl: timedelta,
) -> dict:
    """Assemble a signed, encrypted outer envelope (protocol §3)."""
    outer = {
        "protocol": SUPPORTED_PROTOCOL,
        "relay_mailbox": relay_mailbox,
        "sender_key_id": sender.key_id,
        "recipient_key_id": recipient_key_id,
        "session_id": session_id,
        "sequence": sequence,
        "created_at": _iso(now),
        "expires_at": _iso(now + ttl),
    }
    aad = aad_bytes(outer)
    outer["ciphertext"] = encrypt_payload(
        payload, session_key=session_key, role=role, sequence=sequence, aad=aad
    )
    outer["signature"] = sign_envelope(outer, sender)
    return outer


def verify_and_open(
    envelope: dict,
    *,
    recipient: KindledIdentity,
    sender_pub: bytes,
    session_key: bytes,
    sender_role: int,
    seq_high_water: int,
    now: datetime,
) -> tuple[dict | None, RejectReason | None]:
    """Apply the receiver reject rules (protocol §8) in order. Returns
    (payload, None) on success or (None, RejectReason) on rejection. The
    transcript-hash check (rule 7) is advisory and intentionally NOT applied
    here."""
    if envelope.get("protocol") != SUPPORTED_PROTOCOL:
        return None, RejectReason.PROTOCOL_MISMATCH
    if not verify_envelope_signature(envelope, sender_pub):
        return None, RejectReason.BAD_SIGNATURE
    if envelope.get("recipient_key_id") != recipient.key_id:
        return None, RejectReason.WRONG_RECIPIENT
    try:
        sequence = int(envelope["sequence"])
    except (KeyError, TypeError, ValueError):
        return None, RejectReason.AEAD_FAILURE
    if sequence <= seq_high_water:
        return None, RejectReason.REPLAY
    if _expired(envelope, now):
        return None, RejectReason.EXPIRED
    try:
        aad = aad_bytes(envelope)
        payload = decrypt_payload(
            envelope["ciphertext"], session_key=session_key,
            role=sender_role, sequence=sequence, aad=aad,
        )
    except Exception:
        return None, RejectReason.AEAD_FAILURE
    return payload, None


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _expired(envelope: dict, now: datetime) -> bool:
    try:
        exp = datetime.strptime(envelope["expires_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except (KeyError, ValueError, TypeError):
        return True
    return now > exp + _SKEW


def build_session_open(
    *,
    sender: KindledIdentity,
    recipient_key_id: str,
    relay_mailbox: str,
    session_id: str,
    ephemeral_pub: bytes,
    bootstrap_nonce: bytes,
    sender_mailbox: str,
    now: datetime,
    ttl: timedelta,
) -> dict:
    """A signed (NOT encrypted — no session key yet) handshake envelope carrying
    the sender's ephemeral X25519 public key + bootstrap nonce (protocol §4).
    `sender_mailbox` is the sender's own relay mailbox, signed in the outer so the
    recipient can address its reply leg back (decoupled-mailbox scheme, Phase 7a)."""
    outer = {
        "protocol": SUPPORTED_PROTOCOL,
        "relay_mailbox": relay_mailbox,
        "sender_key_id": sender.key_id,
        "sender_mailbox": sender_mailbox,
        "recipient_key_id": recipient_key_id,
        "session_id": session_id,
        "sequence": 0,
        "created_at": _iso(now),
        "expires_at": _iso(now + ttl),
        "session_open": {
            "ephemeral_pub": ephemeral_pub.hex(),
            "bootstrap_nonce": bootstrap_nonce.hex(),
        },
    }
    outer["signature"] = sign_envelope(outer, sender)
    return outer


def parse_session_open(
    envelope: dict, *, sender_pub: bytes, now: datetime
) -> tuple[dict | None, RejectReason | None]:
    """Verify a session_open envelope's signature + protocol + expiry. Returns
    ({ephemeral_pub, bootstrap_nonce} hex, None) or (None, RejectReason)."""
    if envelope.get("protocol") != SUPPORTED_PROTOCOL:
        return None, RejectReason.PROTOCOL_MISMATCH
    if not verify_envelope_signature(envelope, sender_pub):
        return None, RejectReason.BAD_SIGNATURE
    if _expired(envelope, now):
        return None, RejectReason.EXPIRED
    body = envelope.get("session_open")
    if not isinstance(body, dict):
        return None, RejectReason.AEAD_FAILURE
    # Surface the signed outer sender_mailbox alongside the handshake body so the
    # responder can address its reply leg (decoupled-mailbox scheme, Phase 7a).
    return {**body, "sender_mailbox": envelope.get("sender_mailbox")}, None

_KEY_ROTATION_MSG_TYPE = "key_rotation_notice"


def build_key_rotation_notice(
    *,
    old_sender: KindledIdentity,
    new_identity_pub: bytes,
    new_key_id: str,
    relay_mailbox: str,
    recipient_key_id: str,
    now: datetime,
    ttl: timedelta,
) -> dict:
    """Build a signed (NOT encrypted) key-rotation notice envelope.

    Signed by the old key so the recipient can verify it with the stored
    old public key. No session key needed — mirrors session_open."""
    outer = {
        "protocol": SUPPORTED_PROTOCOL,
        "message_type": _KEY_ROTATION_MSG_TYPE,
        "relay_mailbox": relay_mailbox,
        "sender_key_id": old_sender.key_id,
        "recipient_key_id": recipient_key_id,
        "created_at": _iso(now),
        "expires_at": _iso(now + ttl),
        "key_rotation": {
            "new_identity_pub": new_identity_pub.hex(),
            "new_key_id": new_key_id,
        },
    }
    outer["signature"] = sign_envelope(outer, old_sender)
    return outer


def parse_key_rotation_notice(
    envelope: dict,
    *,
    sender_old_pub: bytes,
    now: datetime,
) -> tuple[dict | None, RejectReason | None]:
    """Verify a key_rotation_notice envelope. Returns
    ({"new_identity_pub": hex, "new_key_id": str}, None) on success
    or (None, RejectReason) on rejection."""
    if envelope.get("protocol") != SUPPORTED_PROTOCOL:
        return None, RejectReason.PROTOCOL_MISMATCH
    if envelope.get("message_type") != _KEY_ROTATION_MSG_TYPE:
        return None, RejectReason.PROTOCOL_MISMATCH
    if not verify_envelope_signature(envelope, sender_old_pub):
        return None, RejectReason.BAD_SIGNATURE
    if _expired(envelope, now):
        return None, RejectReason.EXPIRED
    body = envelope.get("key_rotation")
    if not isinstance(body, dict):
        return None, RejectReason.AEAD_FAILURE
    new_pub_hex = body.get("new_identity_pub")
    new_key_id = body.get("new_key_id")
    if not isinstance(new_pub_hex, str) or not isinstance(new_key_id, str):
        return None, RejectReason.AEAD_FAILURE
    try:
        new_pub_bytes = bytes.fromhex(new_pub_hex)
    except ValueError:
        return None, RejectReason.AEAD_FAILURE
    if fingerprint(new_pub_bytes) != new_key_id:
        return None, RejectReason.AEAD_FAILURE
    return {"new_identity_pub": new_pub_hex, "new_key_id": new_key_id}, None
