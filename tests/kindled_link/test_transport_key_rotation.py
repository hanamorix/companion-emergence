"""Tests for key_rotation_notice receive path in transport.poll_and_ingest."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.protocol import build_key_rotation_notice, sign_envelope
from brain.kindled_link.store import KindledLinkStore, kindled_db_path
from brain.kindled_link.transport import poll_and_ingest

_NOW = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)
_TTL = timedelta(days=7)


def _idn(seed: int) -> KindledIdentity:
    return KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes([seed] * 32)))


@pytest.fixture()
def store(tmp_path: Path) -> KindledLinkStore:
    db = kindled_db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    s = KindledLinkStore(db)
    s.upsert_peer(
        peer_id="kid_old",
        identity_pub_hex=_idn(0).public_bytes.hex(),
        fingerprint="kid_old",
        consent_state="paired",
        relay_url="http://relay",
        now=_NOW,
    )
    yield s
    s.close()


def _make_relay(envelopes: list[dict]) -> MagicMock:
    rc = MagicMock()
    rc.fetch.return_value = [{"id": f"e{i}", "envelope": env} for i, env in enumerate(envelopes)]
    rc.ack = MagicMock()
    return rc


def test_rotation_notice_updates_peer_identity(store: KindledLinkStore, tmp_path: Path) -> None:
    old = _idn(0)
    new = _idn(1)
    notice = build_key_rotation_notice(
        old_sender=old,
        new_identity_pub=new.public_bytes,
        new_key_id=new.key_id,
        relay_mailbox="mbx_local",
        recipient_key_id="kid_local",
        now=_NOW,
        ttl=_TTL,
    )
    # sender_key_id in the notice is old.key_id == "kid_old" (matches the stored peer)
    notice["sender_key_id"] = "kid_old"
    notice["signature"] = sign_envelope(notice, old)

    rc = _make_relay([notice])
    local_idn = _idn(99)
    poll_and_ingest(store, local_idn, rc, now=_NOW, persona_dir=tmp_path)

    # #6: rotation rekeys peer_id to the new key id; the peer resolves by new,
    # not the old id.
    assert store.get_peer("kid_old") is None
    peer = store.get_peer(new.key_id)
    assert peer["identity_pub"] == new.public_bytes.hex()
    assert peer["previous_identity_pub"] == old.public_bytes.hex()


def test_rotation_notice_idempotent(store: KindledLinkStore, tmp_path: Path) -> None:
    """Receiving the same rotation notice twice must not error."""
    old = _idn(0)
    new = _idn(1)
    notice = build_key_rotation_notice(
        old_sender=old,
        new_identity_pub=new.public_bytes,
        new_key_id=new.key_id,
        relay_mailbox="mbx_local",
        recipient_key_id="kid_local",
        now=_NOW,
        ttl=_TTL,
    )
    notice["sender_key_id"] = "kid_old"
    notice["signature"] = sign_envelope(notice, old)

    rc = _make_relay([notice])
    local_idn = _idn(99)
    poll_and_ingest(store, local_idn, rc, now=_NOW, persona_dir=tmp_path)
    # Second delivery — peer is now rekeyed to new.key_id (#6); the re-delivered
    # notice (sender_key_id=kid_old) finds no peer and is silently dropped. No
    # error, no double-apply.
    poll_and_ingest(store, local_idn, rc, now=_NOW, persona_dir=tmp_path)
    assert store.get_peer("kid_old") is None
    peer = store.get_peer(new.key_id)
    assert peer["identity_pub"] == new.public_bytes.hex()


def test_rotation_notice_rollback_rejected(store: KindledLinkStore, tmp_path: Path) -> None:
    """A rotation notice from an unknown sender_key_id does not apply."""
    old = _idn(0)
    new = _idn(1)
    # First rotate: old -> new
    store.update_peer_identity("kid_old", new.public_bytes.hex(), new.key_id, _NOW)
    # Rollback attempt: attacker uses new key, claims new_identity_pub=old
    rollback_notice = build_key_rotation_notice(
        old_sender=new,
        new_identity_pub=old.public_bytes,
        new_key_id=old.key_id,
        relay_mailbox="mbx_local",
        recipient_key_id="kid_local",
        now=_NOW,
        ttl=_TTL,
    )
    # sender_key_id = new.key_id; after #6 the peer is rekeyed to new.key_id, so
    # the notice FINDS the peer and hits the rollback guard (new_identity_pub ==
    # previous_identity_pub → rejected).
    rollback_notice["sender_key_id"] = new.key_id
    rc = _make_relay([rollback_notice])
    local_idn = _idn(99)
    poll_and_ingest(store, local_idn, rc, now=_NOW, persona_dir=tmp_path)
    peer = store.get_peer(new.key_id)
    # Still has the new key; rollback did not apply
    assert peer["identity_pub"] == new.public_bytes.hex()
