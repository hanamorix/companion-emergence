"""Kindled-link local identity: an Ed25519 keypair, its fingerprint, and a
human-verifiable phrase. Identity keys ONLY sign; they never encrypt bodies
(crypto-decision doc §2). Private key persisted under the persona dir, 0600."""
from __future__ import annotations

import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)


def _raw_pub(pub: Ed25519PublicKey) -> bytes:
    return pub.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def fingerprint(pub_raw: bytes) -> str:
    """kid_ = "kid_" + sha256(pubkey)[:16 hex] (protocol §2)."""
    return "kid_" + hashlib.sha256(pub_raw).hexdigest()[:16]


def fingerprint_phrase(pub_raw: bytes) -> str:
    """Human-verifiable phrase: first 16 bytes of the sha256 as eight 4-hex
    groups. (Word-list / emoji rendering is a UI concern — deferred.)"""
    h = hashlib.sha256(pub_raw).hexdigest()
    return " ".join(h[i : i + 4] for i in range(0, 32, 4))


def verify(pub_raw: bytes, signature: bytes, data: bytes) -> bool:
    """True iff `signature` is a valid Ed25519 signature over `data`. Fail-soft."""
    try:
        Ed25519PublicKey.from_public_bytes(pub_raw).verify(signature, data)
        return True
    except Exception:
        return False
