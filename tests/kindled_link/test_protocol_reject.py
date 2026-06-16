"""Receiver reject rules (protocol §8): signature, recipient, AEAD tamper,
replay, expiry, protocol mismatch."""
from datetime import UTC, datetime, timedelta

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.protocol import (
    ROLE_INITIATOR,
    RejectReason,
    build_envelope,
    sign_envelope,
    verify_and_open,
)

_NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)


def _pair():
    a = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    b = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))))
    return a, b


def _session_key():
    return bytes(range(32))  # any 32 bytes; sender + receiver share it


def _make(a, b, *, sequence=1, session_key=None, session_id="ks_1"):
    return build_envelope(
        payload={"payload_type": "message", "body": "hi"},
        sender=a, recipient_key_id=b.key_id, relay_mailbox="mbx_b",
        session_id=session_id, sequence=sequence, role=ROLE_INITIATOR,
        session_key=session_key or _session_key(), now=_NOW, ttl=timedelta(days=7),
    )


def test_valid_envelope_opens():
    a, b = _pair()
    sk = _session_key()
    env = _make(a, b, session_key=sk)
    payload, reason = verify_and_open(
        env, recipient=b, sender_pub=a.public_bytes, session_key=sk,
        sender_role=ROLE_INITIATOR, seq_high_water=0, now=_NOW,
    )
    assert reason is None and payload["body"] == "hi"


def test_bad_signature_rejected():
    a, b = _pair()
    sk = _session_key()
    env = _make(a, b, session_key=sk)
    env["signature"] = "00" * 64
    _, reason = verify_and_open(env, recipient=b, sender_pub=a.public_bytes,
                                session_key=sk, sender_role=ROLE_INITIATOR,
                                seq_high_water=0, now=_NOW)
    assert reason == RejectReason.BAD_SIGNATURE


def test_wrong_recipient_rejected():
    a, b = _pair()
    c = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(64, 96))))
    sk = _session_key()
    env = _make(a, b, session_key=sk)
    _, reason = verify_and_open(env, recipient=c, sender_pub=a.public_bytes,
                                session_key=sk, sender_role=ROLE_INITIATOR,
                                seq_high_water=0, now=_NOW)
    assert reason == RejectReason.WRONG_RECIPIENT


def test_ciphertext_tamper_rejected():
    a, b = _pair()
    sk = _session_key()
    env = _make(a, b, session_key=sk)
    # flip a byte of ciphertext, then RE-SIGN so the signature passes — the AEAD
    # tag must still reject.
    ct = bytearray(bytes.fromhex(env["ciphertext"]))
    ct[0] ^= 1
    env["ciphertext"] = ct.hex()
    env["signature"] = sign_envelope({k: v for k, v in env.items() if k != "signature"}, a)
    _, reason = verify_and_open(env, recipient=b, sender_pub=a.public_bytes,
                                session_key=sk, sender_role=ROLE_INITIATOR,
                                seq_high_water=0, now=_NOW)
    assert reason == RejectReason.AEAD_FAILURE


def test_replay_below_high_water_rejected():
    a, b = _pair()
    sk = _session_key()
    env = _make(a, b, sequence=5, session_key=sk)
    _, reason = verify_and_open(env, recipient=b, sender_pub=a.public_bytes,
                                session_key=sk, sender_role=ROLE_INITIATOR,
                                seq_high_water=5, now=_NOW)  # already seen seq 5
    assert reason == RejectReason.REPLAY


def test_expired_envelope_rejected():
    a, b = _pair()
    sk = _session_key()
    env = _make(a, b, session_key=sk)
    later = _NOW + timedelta(days=8)  # past expiry + skew
    _, reason = verify_and_open(env, recipient=b, sender_pub=a.public_bytes,
                                session_key=sk, sender_role=ROLE_INITIATOR,
                                seq_high_water=0, now=later)
    assert reason == RejectReason.EXPIRED


def test_protocol_mismatch_rejected():
    a, b = _pair()
    sk = _session_key()
    env = _make(a, b, session_key=sk)
    env["protocol"] = "kindled-link/2"
    env["signature"] = sign_envelope({k: v for k, v in env.items() if k != "signature"}, a)
    _, reason = verify_and_open(env, recipient=b, sender_pub=a.public_bytes,
                                session_key=sk, sender_role=ROLE_INITIATOR,
                                seq_high_water=0, now=_NOW)
    assert reason == RejectReason.PROTOCOL_MISMATCH


def test_session_open_roundtrip():
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    from brain.kindled_link.protocol import build_session_open, parse_session_open

    a, b = _pair()
    eph = X25519PrivateKey.from_private_bytes(bytes(range(64, 96)))
    bootstrap = bytes(range(200, 216))
    env = build_session_open(
        sender=a, recipient_key_id=b.key_id, relay_mailbox="mbx_b",
        session_id="ks_1", ephemeral_pub=eph.public_key().public_bytes_raw(),
        bootstrap_nonce=bootstrap, now=_NOW, ttl=timedelta(days=7),
    )
    parsed, reason = parse_session_open(env, sender_pub=a.public_bytes, now=_NOW)
    assert reason is None
    assert parsed["ephemeral_pub"] == eph.public_key().public_bytes_raw().hex()
    assert parsed["bootstrap_nonce"] == bootstrap.hex()
    # tamper the ephemeral → signature breaks
    env["session_open"]["ephemeral_pub"] = "00" * 32
    _, reason2 = parse_session_open(env, sender_pub=a.public_bytes, now=_NOW)
    assert reason2 == RejectReason.BAD_SIGNATURE
