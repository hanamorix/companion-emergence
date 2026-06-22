"""Test 2 — session_keys save/get round-trips (T2.5)."""
from datetime import UTC, datetime

from brain.kindled_link.protocol import ROLE_INITIATOR, ROLE_RESPONDER
from brain.kindled_link.store import KindledLinkStore

_NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)


def _store(tmp_path):
    return KindledLinkStore(tmp_path / "kl.db")


def test_save_get_session_key_round_trips(tmp_path):
    """save_session_key persists the key bytes + roles; get_session_key returns them."""
    s = _store(tmp_path)
    sk = b"\x42" * 32
    s.save_session_key(
        peer_id="kid_p", session_id="ks_1",
        session_key=sk, my_role=ROLE_INITIATOR, peer_role=ROLE_RESPONDER,
        now=_NOW,
    )
    row = s.get_session_key("kid_p", "ks_1")
    assert row is not None
    assert row["session_key"] == sk
    assert row["my_role"] == ROLE_INITIATOR
    assert row["peer_role"] == ROLE_RESPONDER


def test_get_session_key_absent_returns_none(tmp_path):
    s = _store(tmp_path)
    assert s.get_session_key("kid_missing", "ks_x") is None
