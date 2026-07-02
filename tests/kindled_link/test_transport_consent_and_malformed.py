"""Bug #7 (consent gate) + #8 (malformed relay item crash) in poll_and_ingest."""
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.store import KindledLinkStore
from brain.kindled_link.transport import poll_and_ingest

_NOW = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)


class _FakeRelay:
    def __init__(self, items):
        self._items = items
        self.acked = []

    def fetch(self):
        return self._items

    def ack(self, ids):
        self.acked.extend(ids)

    def push(self, env):  # unused here
        pass


def _idn():
    return KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))))


def test_malformed_relay_item_does_not_crash_the_tick(tmp_path):
    """#8: a hostile/broken relay returning items missing 'envelope' or 'id'
    must not crash the whole inbound tick (unguarded item[...] → KeyError)."""
    store = KindledLinkStore(tmp_path / "s.db")
    idn = _idn()
    relay = _FakeRelay([
        {"no_envelope": True},                       # missing 'envelope'
        {"envelope": {"sender_key_id": "kid_x"}},    # missing 'id'
        "not even a dict",                           # wrong type
    ])
    # Must return a summary, not raise.
    summary = poll_and_ingest(store, idn, relay, now=_NOW)
    assert isinstance(summary, dict)
    assert summary.get("accepted") == {}


def test_blocked_peer_inbound_is_not_processed(tmp_path, monkeypatch):
    """#7: blocked/revoked/paused peers must not have ANY inbound processed —
    the handshake path is never entered; items are dropped (acked)."""
    import brain.kindled_link.transport as tp

    def _boom(*a, **k):
        raise AssertionError("blocked peer's inbound was processed")

    monkeypatch.setattr(tp, "on_session_open", _boom)
    store = KindledLinkStore(tmp_path / "s.db")
    idn = _idn()
    for state in ("blocked", "revoked", "paused"):
        kid = f"kid_{state}"
        store.upsert_peer(
            peer_id=kid, identity_pub_hex="aa" * 32, fingerprint=kid,
            consent_state=state, relay_url="https://r", now=_NOW,
        )
        relay = _FakeRelay([
            {"id": f"e_{state}", "envelope": {"sender_key_id": kid, "session_id": "s1",
                                              "session_open": {"x": 1}}},
        ])
        summary = poll_and_ingest(store, idn, relay, now=_NOW)
        assert summary.get("accepted", {}).get(kid) is None
        assert f"e_{state}" in relay.acked, f"{state} item not dropped"


def test_paired_peer_session_open_is_processed(tmp_path, monkeypatch):
    """Control for #7: a PAIRED peer's inbound handshake IS entered (the gate
    only blocks blocked/revoked/paused)."""
    import brain.kindled_link.transport as tp

    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return None

    monkeypatch.setattr(tp, "on_session_open", _spy)
    store = KindledLinkStore(tmp_path / "s.db")
    idn = _idn()
    store.upsert_peer(
        peer_id="kid_ok", identity_pub_hex="aa" * 32, fingerprint="kid_ok",
        consent_state="paired", relay_url="https://r", now=_NOW,
    )
    relay = _FakeRelay([
        {"id": "e1", "envelope": {"sender_key_id": "kid_ok", "session_id": "s1",
                                   "session_open": {"x": 1}}},
    ])
    poll_and_ingest(store, idn, relay, now=_NOW)
    assert called["n"] == 1
