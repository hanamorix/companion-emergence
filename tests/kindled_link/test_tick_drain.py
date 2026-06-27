"""Tests for _drain_pending_rotation_notices in tick.py."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.kindled_link.relay_client import RelayUnavailableError
from brain.kindled_link.store import KindledLinkStore, kindled_db_path
from brain.kindled_link.tick import _drain_pending_rotation_notices

_NOW = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path: Path) -> KindledLinkStore:
    db = kindled_db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    s = KindledLinkStore(db)
    s.upsert_peer(
        peer_id="peer_1",
        identity_pub_hex="aa" * 32,
        fingerprint="peer_1",
        consent_state="paired",
        relay_url="http://relay",
        now=_NOW,
    )
    yield s
    s.close()


def _fresh_envelope(now: datetime = _NOW, expires_days: int = 7) -> str:
    exp = now + timedelta(days=expires_days)
    return json.dumps({
        "protocol": "kindled-link/1",
        "message_type": "key_rotation_notice",
        "expires_at": exp.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })


def test_drain_sends_queued_notice(store: KindledLinkStore, tmp_path: Path) -> None:
    store.queue_rotation_notice("peer_1", _fresh_envelope(), _NOW)
    rc = MagicMock()
    _drain_pending_rotation_notices(store, rc, tmp_path, _NOW)
    rc.push.assert_called_once()
    # After successful send, notice removed from queue
    assert store.pop_pending_rotation_notices() == []


def test_drain_keeps_notice_on_relay_failure(store: KindledLinkStore, tmp_path: Path) -> None:
    store.queue_rotation_notice("peer_1", _fresh_envelope(), _NOW)
    rc = MagicMock()
    rc.push.side_effect = RelayUnavailableError("down")
    _drain_pending_rotation_notices(store, rc, tmp_path, _NOW)
    # Notice still in queue
    assert len(store.pop_pending_rotation_notices()) == 1


def test_drain_drops_expired_notice(store: KindledLinkStore, tmp_path: Path) -> None:
    store.queue_rotation_notice("peer_1", _fresh_envelope(expires_days=-1), _NOW)
    rc = MagicMock()
    _drain_pending_rotation_notices(store, rc, tmp_path, _NOW)
    rc.push.assert_not_called()
    # Dropped from queue
    assert store.pop_pending_rotation_notices() == []
