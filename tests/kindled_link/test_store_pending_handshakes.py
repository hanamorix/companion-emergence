"""Test 1 — pending_handshakes save/get/clear round-trips (T2.5)."""
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from brain.kindled_link.protocol import ROLE_INITIATOR
from brain.kindled_link.store import KindledLinkStore


def _store(tmp_path):
    return KindledLinkStore(tmp_path / "kl.db")


def test_save_get_pending_handshake_round_trips(tmp_path):
    """save_pending_handshake persists all fields; get returns them correctly,
    including rebuilding the X25519 private key from its raw bytes."""
    s = _store(tmp_path)
    eph = X25519PrivateKey.generate()
    eph_priv_raw = eph.private_bytes_raw()
    nonce = b"\xab" * 16
    s.save_pending_handshake(
        peer_id="kid_p",
        session_id="ks_1",
        my_eph_priv_raw=eph_priv_raw,
        bootstrap_nonce=nonce,
        my_role=ROLE_INITIATOR,
    )
    row = s.get_pending_handshake("kid_p", "ks_1")
    assert row is not None
    assert row["my_role"] == ROLE_INITIATOR
    assert row["bootstrap_nonce"] == nonce
    # Rebuild the X25519 private key from the stored bytes
    rebuilt = X25519PrivateKey.from_private_bytes(row["my_eph_priv_raw"])
    # Both keys produce the same public key bytes
    assert rebuilt.public_key().public_bytes_raw() == eph.public_key().public_bytes_raw()


def test_get_missing_pending_handshake_returns_none(tmp_path):
    s = _store(tmp_path)
    assert s.get_pending_handshake("kid_missing", "ks_x") is None


def test_clear_pending_handshake_removes_row(tmp_path):
    s = _store(tmp_path)
    eph_priv_raw = X25519PrivateKey.generate().private_bytes_raw()
    s.save_pending_handshake(
        peer_id="kid_p", session_id="ks_1",
        my_eph_priv_raw=eph_priv_raw, bootstrap_nonce=b"\x00" * 16,
        my_role=ROLE_INITIATOR,
    )
    assert s.get_pending_handshake("kid_p", "ks_1") is not None
    s.clear_pending_handshake("kid_p", "ks_1")
    assert s.get_pending_handshake("kid_p", "ks_1") is None
