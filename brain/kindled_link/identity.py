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

from brain.utils.atomic_write import save_bytes_atomic
from brain.utils.file_lock import file_lock


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
        # Two callers can make the identity for the first time together (the bridge start and the supervisor both do). The
        # check-and-create runs under the key's lock, and the loser loads the winner's key instead of returning the keypair it
        # generated, which would not be the one on disk (#483).
        with file_lock(key_path):
            if key_path.exists():
                return cls(Ed25519PrivateKey.from_private_bytes(key_path.read_bytes()))
            return cls._create_locked(key_path)

    @classmethod
    def _create_locked(cls, key_path: Path) -> KindledIdentity:
        priv = Ed25519PrivateKey.generate()
        raw = priv.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        # The shared atomic bytes writer (#44, #465): a unique temp created 0600 in the key's directory (never
        # world-readable, even briefly), fsynced, then one retried atomic replace. A mid-write failure (e.g. a full
        # disc) removes the temp and leaves key_path absent, never a truncated key that crashes the next load via
        # from_private_bytes(b'').
        save_bytes_atomic(key_path, raw, mode=0o600)
        try:
            os.chmod(key_path, 0o600)  # best-effort; no-op semantics on Windows
        except OSError:
            pass
        return cls(priv)

    def sign(self, data: bytes) -> bytes:
        return self._priv.sign(data)

    @classmethod
    def rotate(cls, persona_dir: Path) -> tuple[KindledIdentity, KindledIdentity]:
        """Generate a new keypair, write it atomically. Returns (new, old).

        The caller MUST use `old` to sign rotation notices before discarding it.
        The key file is overwritten through the same shared atomic bytes writer
        as load_or_create (#44). `old` remains valid in memory
        throughout the caller's lifetime — only the on-disk file changes."""
        old = cls.load_or_create(persona_dir)  # before the lock: file_lock is not re-entrant
        key_path = persona_dir / _KEY_DIRNAME / _KEY_FILENAME
        with file_lock(key_path):
            return cls._rotate_locked(key_path, old)

    @classmethod
    def _rotate_locked(cls, key_path: Path, old: KindledIdentity) -> tuple[KindledIdentity, KindledIdentity]:
        priv = Ed25519PrivateKey.generate()
        raw = priv.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        save_bytes_atomic(key_path, raw, mode=0o600)
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass
        return cls(priv), old
