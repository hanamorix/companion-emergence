"""Kindled-link local identity: an Ed25519 keypair, its fingerprint, and a
human-verifiable phrase. Identity keys ONLY sign; they never encrypt bodies
(crypto-decision doc §2). Private key persisted under the persona dir, 0600."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
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


_KEY_DIRNAME = "kindled_link"
_KEY_FILENAME = "identity_ed25519.key"


class KindledIdentity:
    """This Kindled's long-term Ed25519 identity. `load_or_create` is idempotent:
    it loads the persisted key if present, else generates + persists one (0600)."""

    def __init__(self, priv: Ed25519PrivateKey) -> None:
        self._priv = priv
        self.public_bytes = _raw_pub(priv.public_key())
        self.key_id = fingerprint(self.public_bytes)

    @classmethod
    def load_or_create(cls, persona_dir: Path) -> KindledIdentity:
        key_path = persona_dir / _KEY_DIRNAME / _KEY_FILENAME
        if key_path.exists():
            return cls(Ed25519PrivateKey.from_private_bytes(key_path.read_bytes()))
        key_path.parent.mkdir(parents=True, exist_ok=True)
        priv = Ed25519PrivateKey.generate()
        raw = priv.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        # O_EXCL + 0600 so the key is never created world-readable even briefly.
        fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, raw)
        finally:
            os.close(fd)
        try:
            os.chmod(key_path, 0o600)  # best-effort; no-op semantics on Windows
        except OSError:
            pass
        return cls(priv)

    def sign(self, data: bytes) -> bytes:
        return self._priv.sign(data)
