"""T3 — outbound send_message round-trip (B1).

Builds a real envelope, pushes it to an in-process dev_relay via a real
RelayClient, then fetches + verify_and_open with the RECIPIENT identity.
"""
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from starlette.testclient import TestClient

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.protocol import ROLE_INITIATOR, ROLE_RESPONDER, verify_and_open
from brain.kindled_link.relay_client import RelayClient
from brain.kindled_link.store import KindledLinkStore
from brain.kindled_link.transport import send_message
from relay.dev_relay import create_app

_NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)


def _identities():
    idn_a = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    idn_b = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))))
    return idn_a, idn_b


def _relay_clients(idn_a, idn_b):
    app = create_app(require_auth=True)
    http = TestClient(app, base_url="http://relay.test")
    rc_a = RelayClient(http, identity=idn_a, mailbox_id="mbx_a")
    rc_b = RelayClient(http, identity=idn_b, mailbox_id="mbx_b")
    rc_a.register()
    rc_b.register()
    return rc_a, rc_b


def _setup_stores(tmp_path, idn_a, idn_b, session_key: bytes, session_id: str):
    store_a = KindledLinkStore(tmp_path / "a.db")
    store_b = KindledLinkStore(tmp_path / "b.db")
    store_a.upsert_peer(
        peer_id=idn_b.key_id,
        identity_pub_hex=idn_b.public_bytes.hex(),
        fingerprint=idn_b.key_id,
        consent_state="paired",
        relay_url="https://relay.test",
        relay_mailbox="mbx_b",
        now=_NOW,
    )
    store_b.upsert_peer(
        peer_id=idn_a.key_id,
        identity_pub_hex=idn_a.public_bytes.hex(),
        fingerprint=idn_a.key_id,
        consent_state="paired",
        relay_url="https://relay.test",
        relay_mailbox="mbx_a",
        now=_NOW,
    )
    store_a.save_session_key(
        peer_id=idn_b.key_id,
        session_id=session_id,
        session_key=session_key,
        my_role=ROLE_INITIATOR,
        peer_role=ROLE_RESPONDER,
        now=_NOW,
    )
    return store_a, store_b


def test_send_message_round_trip(tmp_path):
    """send_message → relay → fetch → verify_and_open recovers original payload."""
    idn_a, idn_b = _identities()
    rc_a, rc_b = _relay_clients(idn_a, idn_b)

    session_key = b"\xAA" * 32
    session_id = "ks_test1"
    store_a, _store_b = _setup_stores(tmp_path, idn_a, idn_b, session_key, session_id)

    payload = {"text": "hello from A", "ts": "2026-06-21T12:00:00Z"}

    result = send_message(
        store_a,
        idn_a,
        rc_a,
        peer_id=idn_b.key_id,
        session_id=session_id,
        payload=payload,
        now=_NOW,
    )
    assert result is True

    envelopes = rc_b.fetch()
    assert len(envelopes) == 1
    env = envelopes[0]["envelope"]

    decrypted, reason = verify_and_open(
        env,
        recipient=idn_b,
        sender_pub=idn_a.public_bytes,
        session_key=session_key,
        sender_role=ROLE_INITIATOR,
        seq_high_water=0,
        now=_NOW,
    )
    assert reason is None, f"verify_and_open should succeed, got {reason}"
    assert decrypted == payload


def test_send_message_no_session_key_returns_false(tmp_path):
    """send_message returns False when there is no session_keys row."""
    idn_a, idn_b = _identities()
    rc_a, _rc_b = _relay_clients(idn_a, idn_b)

    store_a = KindledLinkStore(tmp_path / "a.db")
    store_a.upsert_peer(
        peer_id=idn_b.key_id,
        identity_pub_hex=idn_b.public_bytes.hex(),
        fingerprint=idn_b.key_id,
        consent_state="paired",
        relay_url="https://relay.test",
        relay_mailbox="mbx_b",
        now=_NOW,
    )
    result = send_message(
        store_a,
        idn_a,
        rc_a,
        peer_id=idn_b.key_id,
        session_id="ks_no_key",
        payload={"text": "should not send"},
        now=_NOW,
    )
    assert result is False
