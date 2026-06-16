"""Outbound-only relay client, exercised against the in-process dev relay.

NOTE: starlette.testclient.TestClient IS an httpx.Client subclass (inherits it
directly), so it can be passed wherever httpx.Client is required. This lets us
exercise RelayClient against the in-process FastAPI relay without opening any
real sockets and without requiring the async-only httpx.ASGITransport.
"""
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from starlette.testclient import TestClient

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.relay_client import RelayClient
from relay.dev_relay import create_app


def _client(idn):
    # TestClient subclasses httpx.Client — passes the type check in RelayClient.
    http = TestClient(create_app(require_auth=True), base_url="http://relay.test")
    return RelayClient(http, identity=idn, mailbox_id="mbx_me")


def _envelope():
    return {"protocol": "kindled-link/1", "relay_mailbox": "mbx_me",
            "sender_key_id": "kid_x", "recipient_key_id": "kid_me", "session_id": "ks_1",
            "sequence": 1, "created_at": "x", "expires_at": "y",
            "ciphertext": "deadbeef", "signature": "00"}


def test_register_push_fetch_ack_roundtrip():
    idn = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    rc = _client(idn)
    rc.register()
    rc.push(_envelope())                     # anyone can push to mbx_me
    envs = rc.fetch()                         # owner-authenticated fetch
    assert len(envs) == 1 and envs[0]["ciphertext"] == "deadbeef"
    rc.ack([envs[0]["id"]])
    assert rc.fetch() == []


def test_client_exposes_no_inbound_listener():
    # The client is a pure httpx caller — it must not open any server/socket.
    import inspect

    import brain.kindled_link.relay_client as mod

    src = inspect.getsource(mod)
    for forbidden in ("uvicorn", "FastAPI", "socket.bind", ".serve(", "listen("):
        assert forbidden not in src, f"relay client must be outbound-only ({forbidden})"
