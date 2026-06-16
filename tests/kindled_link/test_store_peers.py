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
