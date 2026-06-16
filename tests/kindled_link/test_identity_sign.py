from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from brain.kindled_link.identity import KindledIdentity, verify


def test_sign_is_deterministic(tmp_path) -> None:
    idn = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    assert idn.sign(b"hello") == idn.sign(b"hello")  # Ed25519 (RFC 8032)


def test_verify_roundtrip(tmp_path) -> None:
    idn = KindledIdentity.load_or_create(tmp_path)
    sig = idn.sign(b"a message")
    assert verify(idn.public_bytes, sig, b"a message") is True


def test_verify_rejects_tamper(tmp_path) -> None:
    idn = KindledIdentity.load_or_create(tmp_path)
    sig = idn.sign(b"a message")
    assert verify(idn.public_bytes, sig, b"a MESSAGE") is False
    flipped = bytes([sig[0] ^ 1]) + sig[1:]
    assert verify(idn.public_bytes, flipped, b"a message") is False
