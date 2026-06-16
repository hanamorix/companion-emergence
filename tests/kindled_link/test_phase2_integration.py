"""End-to-end: two personas correspond through an in-process dev relay, with
adversarial replay / relay-compromise cases.

NOTE: Uses starlette.testclient.TestClient (a sync httpx.Client subclass) instead
of httpx.ASGITransport, which is async-only in httpx 0.28 and raises
AttributeError with the sync httpx.Client.  Both RelayClient instances share the
SAME TestClient so they share the in-memory relay store.
"""
from datetime import UTC, datetime, timedelta

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from starlette.testclient import TestClient

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.protocol import (
    ROLE_INITIATOR,
    build_envelope,
    build_session_open,
    derive_session_key,
    generate_ephemeral,
    parse_session_open,
    verify_and_open,
)
from brain.kindled_link.relay_client import RelayClient
from brain.kindled_link.store import KindledLinkStore
from relay.dev_relay import create_app

_NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)


def _both_clients(idn_a, idn_b):
    """Shared TestClient so both RelayClients see the same in-memory store."""
    http = TestClient(create_app(require_auth=True), base_url="http://relay.test")
    a = RelayClient(http, identity=idn_a, mailbox_id="mbx_a")
    b = RelayClient(http, identity=idn_b, mailbox_id="mbx_b")
    a.register()
    b.register()
    return a, b


def _handshake(idn_a, idn_b):
    """A initiates; both derive the same session key. Returns session_key bytes."""
    eph_a, eph_b = generate_ephemeral(), generate_ephemeral()
    bootstrap = bytes(range(16))
    so = build_session_open(
        sender=idn_a, recipient_key_id=idn_b.key_id, relay_mailbox="mbx_b",
        session_id="ks_1", ephemeral_pub=eph_a.public_key().public_bytes_raw(),
        bootstrap_nonce=bootstrap, now=_NOW, ttl=timedelta(days=7),
    )
    parsed, reason = parse_session_open(so, sender_pub=idn_a.public_bytes, now=_NOW)
    assert reason is None
    key_b = derive_session_key(
        eph_b, bytes.fromhex(parsed["ephemeral_pub"]),
        sender_fp=idn_b.key_id, recipient_fp=idn_a.key_id,
        session_id="ks_1", bootstrap_nonce=bootstrap,
    )
    key_a = derive_session_key(
        eph_a, eph_b.public_key().public_bytes_raw(),
        sender_fp=idn_a.key_id, recipient_fp=idn_b.key_id,
        session_id="ks_1", bootstrap_nonce=bootstrap,
    )
    assert key_a == key_b
    return key_a


def test_full_message_loop_through_relay(tmp_path):
    a = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    b = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))))
    rc_a, rc_b = _both_clients(a, b)
    sk = _handshake(a, b)
    store_b = KindledLinkStore(tmp_path / "b.db")

    env = build_envelope(
        payload={"payload_type": "message", "body": "hello B"},
        sender=a, recipient_key_id=b.key_id, relay_mailbox="mbx_b",
        session_id="ks_1", sequence=1, role=ROLE_INITIATOR,
        session_key=sk, now=_NOW, ttl=timedelta(days=7),
    )
    rc_a.push(env)

    fetched = rc_b.fetch()
    assert len(fetched) == 1
    # Strip the relay-internal "id" tracking field before protocol verification.
    # The relay merges {"id": ..., **envelope} for ack purposes; "id" is not
    # part of the signed envelope and must not appear in sig_input_bytes.
    env_id = fetched[0]["id"]
    wire_env = {k: v for k, v in fetched[0].items() if k != "id"}
    hw = store_b.get_seq_high_water(a.key_id, "ks_1")
    payload, reason = verify_and_open(
        wire_env, recipient=b, sender_pub=a.public_bytes,
        session_key=sk, sender_role=ROLE_INITIATOR,
        seq_high_water=hw, now=_NOW,
    )
    assert reason is None and payload["body"] == "hello B"
    store_b.set_seq_high_water(a.key_id, "ks_1", 1)
    rc_b.ack([env_id])

    # Replay simulation: relay re-delivers the same envelope → rejected by HWM.
    rc_a.push(env)
    again_raw = rc_b.fetch()[0]
    again = {k: v for k, v in again_raw.items() if k != "id"}
    _, reason2 = verify_and_open(
        again, recipient=b, sender_pub=a.public_bytes, session_key=sk,
        sender_role=ROLE_INITIATOR,
        seq_high_water=store_b.get_seq_high_water(a.key_id, "ks_1"),
        now=_NOW,
    )
    assert reason2.value == "replay"


def test_relay_cannot_decrypt_or_forge(tmp_path):
    a = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    b = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))))
    sk = _handshake(a, b)
    env = build_envelope(
        payload={"payload_type": "message", "body": "secret"},
        sender=a, recipient_key_id=b.key_id, relay_mailbox="mbx_b",
        session_id="ks_1", sequence=1, role=ROLE_INITIATOR,
        session_key=sk, now=_NOW, ttl=timedelta(days=7),
    )
    # The relay only ever holds env (ciphertext hex + metadata). It has no
    # session key, so it cannot recover "secret".
    assert "secret" not in env["ciphertext"]
    # A relay forgery: flip the ciphertext and try to pass it off. Without the
    # identity private key it cannot re-sign, so the signature check fails.
    forged = dict(env)
    ct = bytearray(bytes.fromhex(forged["ciphertext"]))
    ct[0] ^= 0xFF
    forged["ciphertext"] = ct.hex()
    _, reason = verify_and_open(
        forged, recipient=b, sender_pub=a.public_bytes, session_key=sk,
        sender_role=ROLE_INITIATOR, seq_high_water=0, now=_NOW,
    )
    assert reason.value == "bad_signature"  # signature no longer matches the mutated ciphertext
