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


def _sign_challenge(idn, mailbox_id, nonce_hex):
    from brain.kindled_link.codec import canonical_json
    body = {"purpose": "kindled-relay-auth/1", "mailbox": mailbox_id, "nonce": nonce_hex}
    return idn.sign(canonical_json(body)).hex()


def test_authenticated_fetch_requires_valid_signature():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from brain.kindled_link.identity import KindledIdentity

    owner = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    c = TestClient(create_app(require_auth=True))
    # register the mailbox to the owner's key
    c.post("/mailbox/register", json={"mailbox_id": "mbx_b", "identity_pub": owner.public_bytes.hex()})
    c.post("/envelope", json=_envelope())

    # no signature → 401
    assert c.post("/mailbox/fetch", json={"mailbox_id": "mbx_b"}).status_code == 401

    # valid challenge + signature → 200
    nonce = c.post("/mailbox/challenge", json={"mailbox_id": "mbx_b"}).json()["nonce"]
    sig = _sign_challenge(owner, "mbx_b", nonce)
    r = c.post("/mailbox/fetch", json={"mailbox_id": "mbx_b", "nonce": nonce,
                                       "signature": sig, "identity_pub": owner.public_bytes.hex()})
    assert r.status_code == 200 and len(r.json()["envelopes"]) == 1


def test_leaked_mailbox_id_without_key_cannot_fetch():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from brain.kindled_link.identity import KindledIdentity

    owner = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    attacker = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))))
    c = TestClient(create_app(require_auth=True))
    c.post("/mailbox/register", json={"mailbox_id": "mbx_b", "identity_pub": owner.public_bytes.hex()})
    c.post("/envelope", json=_envelope())
    nonce = c.post("/mailbox/challenge", json={"mailbox_id": "mbx_b"}).json()["nonce"]
    # attacker signs with their OWN key → rejected (key != registered owner)
    sig = _sign_challenge(attacker, "mbx_b", nonce)
    r = c.post("/mailbox/fetch", json={"mailbox_id": "mbx_b", "nonce": nonce,
                                       "signature": sig, "identity_pub": attacker.public_bytes.hex()})
    assert r.status_code == 401
