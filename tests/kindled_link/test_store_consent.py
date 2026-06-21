from datetime import UTC, datetime

import pytest

from brain.kindled_link.store import ConsentTransitionError, KindledLinkStore


def _peer():
    s = KindledLinkStore(":memory:")
    now = datetime(2026, 6, 15, tzinfo=UTC)
    s.upsert_peer(peer_id="p", identity_pub_hex="00" * 32, fingerprint="kid_x",
                  consent_state="pending_local", relay_url=None, now=now)
    return s, now


def test_legal_path_to_paired() -> None:
    s, now = _peer()
    s.set_consent("p", "pending_remote", now)
    s.set_consent("p", "paired", now)
    assert s.get_peer("p")["consent_state"] == "paired"


def test_paired_can_pause_and_resume() -> None:
    s, now = _peer()
    s.set_consent("p", "pending_remote", now)
    s.set_consent("p", "paired", now)
    s.set_consent("p", "paused", now)
    s.set_consent("p", "paired", now)
    assert s.get_peer("p")["consent_state"] == "paired"


def test_blocked_is_terminal() -> None:
    s, now = _peer()
    s.set_consent("p", "blocked", now)
    with pytest.raises(ConsentTransitionError):
        s.set_consent("p", "paired", now)


def test_revoked_only_escalates_to_blocked() -> None:
    s, now = _peer()
    s.set_consent("p", "revoked", now)
    with pytest.raises(ConsentTransitionError):
        s.set_consent("p", "paired", now)
    s.set_consent("p", "blocked", now)  # allowed
    assert s.get_peer("p")["consent_state"] == "blocked"


def test_invite_consumed_ledger() -> None:
    s, now = _peer()
    assert s.is_invite_consumed("inv_1") is False
    s.mark_invite_consumed("inv_1", now)
    assert s.is_invite_consumed("inv_1") is True
    with pytest.raises(ConsentTransitionError):
        s.mark_invite_consumed("inv_1", now)  # single-use: re-consume rejected


def test_close_is_idempotent_enough() -> None:
    """close() must exist and not raise on first call."""
    s = KindledLinkStore(":memory:")
    s.close()  # must not raise
