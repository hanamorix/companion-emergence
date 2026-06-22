from datetime import UTC, datetime

from brain.kindled_link.store import KindledLinkStore


def _store():
    return KindledLinkStore(":memory:")


def test_upsert_then_get() -> None:
    s = _store()
    now = datetime(2026, 6, 15, tzinfo=UTC)
    s.upsert_peer(
        peer_id="kid_aaa", identity_pub_hex="ab" * 32, fingerprint="kid_aaa",
        consent_state="pending_local", relay_url="https://r", now=now,
    )
    p = s.get_peer("kid_aaa")
    assert p["consent_state"] == "pending_local"
    assert p["identity_pub"] == "ab" * 32
    assert p["relay_url"] == "https://r"


def test_get_missing_returns_none() -> None:
    assert _store().get_peer("nope") is None


def test_list_paired_peers_returns_only_paired() -> None:
    """list_paired_peers returns peer_ids of paired peers, not others."""
    s = _store()
    now = datetime(2026, 6, 21, tzinfo=UTC)
    s.upsert_peer(peer_id="kid_paired", identity_pub_hex="aa" * 32,
                  fingerprint="kid_paired", consent_state="paired",
                  relay_url=None, now=now)
    s.upsert_peer(peer_id="kid_pending", identity_pub_hex="bb" * 32,
                  fingerprint="kid_pending", consent_state="pending_local",
                  relay_url=None, now=now)
    result = s.list_paired_peers()
    assert result == ["kid_paired"]
