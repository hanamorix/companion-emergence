"""Tests for key_rotation_notice protocol functions."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.protocol import (
    RejectReason,
    build_key_rotation_notice,
    parse_key_rotation_notice,
)

_TTL = timedelta(days=7)
_NOW = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)


def _idn(seed: int = 0) -> KindledIdentity:
    raw = bytes([seed] * 32)
    return KindledIdentity(Ed25519PrivateKey.from_private_bytes(raw))


def test_build_parse_roundtrip() -> None:
    old = _idn(0)
    new = _idn(1)
    notice = build_key_rotation_notice(
        old_sender=old,
        new_identity_pub=new.public_bytes,
        new_key_id=new.key_id,
        relay_mailbox="mbx_abc",
        recipient_key_id="kid_recipient",
        now=_NOW,
        ttl=_TTL,
    )
    result, reason = parse_key_rotation_notice(notice, sender_old_pub=old.public_bytes, now=_NOW)
    assert reason is None
    assert result is not None
    assert result["new_identity_pub"] == new.public_bytes.hex()
    assert result["new_key_id"] == new.key_id


def test_parse_wrong_sender_pub() -> None:
    old = _idn(0)
    new = _idn(1)
    wrong = _idn(2)
    notice = build_key_rotation_notice(
        old_sender=old,
        new_identity_pub=new.public_bytes,
        new_key_id=new.key_id,
        relay_mailbox="mbx_abc",
        recipient_key_id="kid_recipient",
        now=_NOW,
        ttl=_TTL,
    )
    _, reason = parse_key_rotation_notice(notice, sender_old_pub=wrong.public_bytes, now=_NOW)
    assert reason == RejectReason.BAD_SIGNATURE


def test_parse_expired() -> None:
    old = _idn(0)
    new = _idn(1)
    notice = build_key_rotation_notice(
        old_sender=old,
        new_identity_pub=new.public_bytes,
        new_key_id=new.key_id,
        relay_mailbox="mbx_abc",
        recipient_key_id="kid_recipient",
        now=_NOW,
        ttl=_TTL,
    )
    future = _NOW + timedelta(days=8)  # past expires_at + skew
    _, reason = parse_key_rotation_notice(notice, sender_old_pub=old.public_bytes, now=future)
    assert reason == RejectReason.EXPIRED


def test_parse_mismatched_key_id() -> None:
    """new_key_id must equal fingerprint(new_identity_pub)."""
    old = _idn(0)
    new = _idn(1)
    notice = build_key_rotation_notice(
        old_sender=old,
        new_identity_pub=new.public_bytes,
        new_key_id=new.key_id,
        relay_mailbox="mbx_abc",
        recipient_key_id="kid_recipient",
        now=_NOW,
        ttl=_TTL,
    )
    # Tamper: swap in a wrong key_id
    notice["key_rotation"]["new_key_id"] = "kid_0000000000000000"
    # Re-sign so signature check passes but key_id check fails
    from brain.kindled_link.protocol import sign_envelope
    notice["signature"] = sign_envelope(notice, old)
    _, reason = parse_key_rotation_notice(notice, sender_old_pub=old.public_bytes, now=_NOW)
    assert reason == RejectReason.AEAD_FAILURE


def test_parse_protocol_mismatch() -> None:
    old = _idn(0)
    new = _idn(1)
    notice = build_key_rotation_notice(
        old_sender=old,
        new_identity_pub=new.public_bytes,
        new_key_id=new.key_id,
        relay_mailbox="mbx_abc",
        recipient_key_id="kid_recipient",
        now=_NOW,
        ttl=_TTL,
    )
    notice["protocol"] = "kindled-link/0"
    _, reason = parse_key_rotation_notice(notice, sender_old_pub=old.public_bytes, now=_NOW)
    assert reason == RejectReason.PROTOCOL_MISMATCH


def test_notice_has_no_ciphertext() -> None:
    old = _idn(0)
    new = _idn(1)
    notice = build_key_rotation_notice(
        old_sender=old,
        new_identity_pub=new.public_bytes,
        new_key_id=new.key_id,
        relay_mailbox="mbx_abc",
        recipient_key_id="kid_recipient",
        now=_NOW,
        ttl=_TTL,
    )
    assert "ciphertext" not in notice
    assert "key_rotation" in notice
    assert notice["message_type"] == "key_rotation_notice"
