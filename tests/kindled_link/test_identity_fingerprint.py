from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from brain.kindled_link.identity import fingerprint, fingerprint_phrase

# Protocol doc §10 KAT: identity A priv = bytes(0..31)
_IDA_PUB_HEX = "03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8"
_KIDA = "kid_56475aa75463474c"


def _ida_pub() -> bytes:
    priv = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    return priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def test_pub_matches_kat() -> None:
    assert _ida_pub().hex() == _IDA_PUB_HEX


def test_fingerprint_matches_kat() -> None:
    assert fingerprint(_ida_pub()) == _KIDA


def test_fingerprint_phrase_is_deterministic_grouped_hex() -> None:
    phrase = fingerprint_phrase(_ida_pub())
    assert phrase == fingerprint_phrase(_ida_pub())  # deterministic
    groups = phrase.split(" ")
    assert len(groups) == 8 and all(len(g) == 4 for g in groups)
