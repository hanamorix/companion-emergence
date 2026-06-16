"""Dev relay: store-and-forward by mailbox. Ciphertext is opaque; the relay
never decrypts."""
from starlette.testclient import TestClient

from relay.dev_relay import create_app


def _client():
    return TestClient(create_app())


def _envelope(mailbox="mbx_b", seq=1):
    return {"protocol": "kindled-link/1", "relay_mailbox": mailbox,
            "sender_key_id": "kid_a", "recipient_key_id": "kid_b",
            "session_id": "ks_1", "sequence": seq, "created_at": "x",
            "expires_at": "y", "ciphertext": "deadbeef", "signature": "00"}


def test_push_then_fetch_returns_envelope():
    c = _client()
    assert c.post("/envelope", json=_envelope()).status_code == 200
    r = c.post("/mailbox/fetch", json={"mailbox_id": "mbx_b"})
    assert r.status_code == 200
    envs = r.json()["envelopes"]
    assert len(envs) == 1 and envs[0]["ciphertext"] == "deadbeef"


def test_ack_removes_envelope():
    c = _client()
    c.post("/envelope", json=_envelope())
    env_id = c.post("/mailbox/fetch", json={"mailbox_id": "mbx_b"}).json()["envelopes"][0]["id"]
    assert c.post("/mailbox/ack", json={"mailbox_id": "mbx_b", "envelope_ids": [env_id]}).status_code == 200
    assert c.post("/mailbox/fetch", json={"mailbox_id": "mbx_b"}).json()["envelopes"] == []
