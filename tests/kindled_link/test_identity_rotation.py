"""Tests for KindledIdentity.rotate."""
from __future__ import annotations

import pytest

from brain.kindled_link.identity import KindledIdentity, verify

_KEY_REL = "kindled_link/identity_ed25519.key"


def test_rotate_returns_different_key(tmp_path) -> None:
    original = KindledIdentity.load_or_create(tmp_path)
    new, old = KindledIdentity.rotate(tmp_path)
    assert new.key_id != original.key_id
    assert old.key_id == original.key_id


def test_rotate_old_key_still_signs(tmp_path) -> None:
    """old returned by rotate() must still be able to sign (private key in memory)."""
    KindledIdentity.load_or_create(tmp_path)
    new, old = KindledIdentity.rotate(tmp_path)
    sig = old.sign(b"test data")
    assert verify(old.public_bytes, sig, b"test data") is True


def test_rotate_new_key_loaded_by_load_or_create(tmp_path) -> None:
    """After rotate(), load_or_create returns the new key."""
    KindledIdentity.load_or_create(tmp_path)
    new, _ = KindledIdentity.rotate(tmp_path)
    reloaded = KindledIdentity.load_or_create(tmp_path)
    assert reloaded.key_id == new.key_id
    assert reloaded.public_bytes == new.public_bytes


def test_rotate_key_file_is_atomic(tmp_path, monkeypatch) -> None:
    """A mid-write failure leaves the OLD key in place, not a truncated file."""

    import brain.kindled_link.identity as idmod

    KindledIdentity.load_or_create(tmp_path)
    original_key_id = KindledIdentity.load_or_create(tmp_path).key_id

    def fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(idmod.os, "replace", fail_replace)
    with pytest.raises(OSError):
        KindledIdentity.rotate(tmp_path)

    recovered = KindledIdentity.load_or_create(tmp_path)
    assert recovered.key_id == original_key_id
