"""Tests for key-rotation store methods."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.kindled_link.store import KindledLinkStore, kindled_db_path


@pytest.fixture()
def store(tmp_path: Path) -> KindledLinkStore:
    db = kindled_db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    s = KindledLinkStore(db)
    yield s
    s.close()


def _add_peer(store: KindledLinkStore, peer_id: str = "peer_1",
              consent: str = "paired", pub: str = "aa" * 32) -> None:
    now = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)
    store.upsert_peer(
        peer_id=peer_id,
        identity_pub_hex=pub,
        fingerprint=peer_id,
        consent_state=consent,
        relay_url="http://relay",
        now=now,
    )


def test_queue_and_pop(store: KindledLinkStore) -> None:
    _add_peer(store)
    now = datetime(2026, 6, 27, tzinfo=UTC)
    env = json.dumps({"protocol": "kindled-link/1", "message_type": "key_rotation_notice"})
    store.queue_rotation_notice("peer_1", env, now)
    rows = store.pop_pending_rotation_notices()
    assert len(rows) == 1
    assert rows[0]["peer_id"] == "peer_1"
    assert rows[0]["envelope_json"] == env


def test_queue_upserts_on_second_rotation(store: KindledLinkStore) -> None:
    _add_peer(store)
    now = datetime(2026, 6, 27, tzinfo=UTC)
    store.queue_rotation_notice("peer_1", "first", now)
    store.queue_rotation_notice("peer_1", "second", now)
    rows = store.pop_pending_rotation_notices()
    assert len(rows) == 1
    assert rows[0]["envelope_json"] == "second"


def test_clear_rotation_notice(store: KindledLinkStore) -> None:
    _add_peer(store)
    now = datetime(2026, 6, 27, tzinfo=UTC)
    store.queue_rotation_notice("peer_1", "env", now)
    store.clear_rotation_notice("peer_1")
    assert store.pop_pending_rotation_notices() == []


def test_update_peer_identity_sets_previous(store: KindledLinkStore) -> None:
    old_pub = "aa" * 32
    new_pub = "bb" * 32
    _add_peer(store, pub=old_pub)
    now = datetime(2026, 6, 27, tzinfo=UTC)
    store.update_peer_identity("peer_1", new_pub, "kid_new", now)
    # peer_id (== fingerprint) is rekeyed on rotation (#6): resolvable by the
    # NEW key id; the old id no longer resolves.
    assert store.get_peer("peer_1") is None
    peer = store.get_peer("kid_new")
    assert peer["identity_pub"] == new_pub
    assert peer["fingerprint"] == "kid_new"
    assert peer["previous_identity_pub"] == old_pub


def test_list_paired_peers_filters_consent(store: KindledLinkStore) -> None:
    _add_peer(store, peer_id="p1", consent="paired")
    _add_peer(store, peer_id="p2", consent="paused")
    _add_peer(store, peer_id="p3", consent="paired")
    paired = store.list_paired_peers()
    ids = {p["peer_id"] for p in paired}
    assert ids == {"p1", "p3"}
