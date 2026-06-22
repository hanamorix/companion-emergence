"""Test 3 — full 3-leg X25519 session handshake (T2.5).

Two identities, two in-memory stores that have each other paired via upsert_peer
with mailboxes. After the full handshake: both get_session_key returns equal bytes,
both have an open sessions row.
"""
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.session import complete_session, on_session_open, open_session
from brain.kindled_link.store import KindledLinkStore

_NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)


def _make_pair(tmp_path):
    """Return (idn_a, store_a, idn_b, store_b) with each paired to the other."""
    idn_a = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    idn_b = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))))

    store_a = KindledLinkStore(tmp_path / "a.db")
    store_b = KindledLinkStore(tmp_path / "b.db")

    # A knows B
    store_a.upsert_peer(
        peer_id=idn_b.key_id, identity_pub_hex=idn_b.public_bytes.hex(),
        fingerprint=idn_b.key_id, consent_state="paired",
        relay_url="https://relay.test", relay_mailbox="mbx_b", now=_NOW,
    )
    # B knows A
    store_b.upsert_peer(
        peer_id=idn_a.key_id, identity_pub_hex=idn_a.public_bytes.hex(),
        fingerprint=idn_a.key_id, consent_state="paired",
        relay_url="https://relay.test", relay_mailbox="mbx_a", now=_NOW,
    )
    return idn_a, store_a, idn_b, store_b


def test_full_3_leg_handshake_both_keys_equal_and_sessions_open(tmp_path):
    """After open_session → on_session_open → complete_session both sides have
    the same session_key bytes AND an open sessions row."""
    idn_a, store_a, idn_b, store_b = _make_pair(tmp_path)

    # Leg 1: A initiates
    leg1 = open_session(store_a, idn_a, peer_id=idn_b.key_id, now=_NOW)
    session_id = leg1["session_id"]

    # Leg 2: B responds
    leg2 = on_session_open(store_b, idn_b, envelope=leg1, now=_NOW)
    assert leg2 is not None

    # Both sides' session key must be identical
    key_b = store_b.get_session_key(idn_a.key_id, session_id)
    assert key_b is not None

    # Leg 3: A completes
    complete_session(store_a, idn_a, reply_envelope=leg2, now=_NOW)

    key_a = store_a.get_session_key(idn_b.key_id, session_id)
    assert key_a is not None

    assert key_a["session_key"] == key_b["session_key"], "Both sides must derive the same key"

    # Both sides must have an open sessions row
    assert store_a.get_session(idn_b.key_id, session_id) is not None
    assert store_b.get_session(idn_a.key_id, session_id) is not None


def test_session_key_and_sessions_row_survive_restart(tmp_path):
    """After the full 3-leg handshake, both session_key and the sessions row
    survive a store close + reopen (B0-2: persistence / restart)."""
    idn_a, store_a, idn_b, store_b = _make_pair(tmp_path)

    leg1 = open_session(store_a, idn_a, peer_id=idn_b.key_id, now=_NOW)
    session_id = leg1["session_id"]
    leg2 = on_session_open(store_b, idn_b, envelope=leg1, now=_NOW)
    complete_session(store_a, idn_a, reply_envelope=leg2, now=_NOW)

    key_a_before = store_a.get_session_key(idn_b.key_id, session_id)
    store_a.close()
    store_b.close()

    # Re-open both stores from the on-disk DB files.
    store_a2 = KindledLinkStore(tmp_path / "a.db")
    store_b2 = KindledLinkStore(tmp_path / "b.db")

    key_a_after = store_a2.get_session_key(idn_b.key_id, session_id)
    key_b_after = store_b2.get_session_key(idn_a.key_id, session_id)

    assert key_a_after is not None and key_a_after["session_key"] == key_a_before["session_key"]
    assert key_b_after is not None
    assert key_a_after["session_key"] == key_b_after["session_key"]

    assert store_a2.get_session(idn_b.key_id, session_id) is not None
    assert store_b2.get_session(idn_a.key_id, session_id) is not None


def test_clobber_guard_on_session_open_does_not_overwrite(tmp_path):
    """A second on_session_open for a session that already has a session_keys row
    returns None and does NOT overwrite the established key (B0-5)."""
    idn_a, store_a, idn_b, store_b = _make_pair(tmp_path)

    leg1 = open_session(store_a, idn_a, peer_id=idn_b.key_id, now=_NOW)
    session_id = leg1["session_id"]
    leg2 = on_session_open(store_b, idn_b, envelope=leg1, now=_NOW)
    complete_session(store_a, idn_a, reply_envelope=leg2, now=_NOW)

    established_key = store_b.get_session_key(idn_a.key_id, session_id)["session_key"]

    # Re-deliver the same leg-1 envelope to the responder.
    result = on_session_open(store_b, idn_b, envelope=leg1, now=_NOW)
    assert result is None, "Clobber guard must return None on duplicate session_open"

    # Original key must be unchanged.
    after_key = store_b.get_session_key(idn_a.key_id, session_id)["session_key"]
    assert after_key == established_key


def test_complete_session_duplicate_leg3_does_not_raise(tmp_path):
    """An interrupted/replayed leg-3 (session_keys already established while the
    pending row still exists, e.g. a crash between save and clear) must drop
    gracefully — not raise sqlite IntegrityError on the sessions PK (initiator-side
    clobber guard, mirrors on_session_open)."""
    idn_a, store_a, idn_b, store_b = _make_pair(tmp_path)

    leg1 = open_session(store_a, idn_a, peer_id=idn_b.key_id, now=_NOW)
    session_id = leg1["session_id"]
    leg2 = on_session_open(store_b, idn_b, envelope=leg1, now=_NOW)

    # Simulate leg-3 having run far enough to persist the key + sessions row but
    # crash before clearing the pending row.
    complete_session(store_a, idn_a, reply_envelope=leg2, now=_NOW)
    established = store_a.get_session_key(idn_b.key_id, session_id)["session_key"]
    store_a.save_pending_handshake(  # re-create the not-cleared pending row
        peer_id=idn_b.key_id, session_id=session_id,
        my_eph_priv_raw=bytes(range(32)), bootstrap_nonce=bytes(range(16)),
        my_role=0, now=_NOW,
    )

    # Re-running leg-3 must NOT raise and must NOT change the key.
    complete_session(store_a, idn_a, reply_envelope=leg2, now=_NOW)
    assert store_a.get_session_key(idn_b.key_id, session_id)["session_key"] == established


def test_on_session_open_unknown_sender_returns_none(tmp_path):
    """on_session_open drops (returns None) when the sender has no peer row."""
    idn_a = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    idn_b = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))))

    store_a = KindledLinkStore(tmp_path / "a.db")
    store_b = KindledLinkStore(tmp_path / "b.db")

    # Give A a local mailbox so open_session can find it.
    store_a.upsert_peer(
        peer_id=idn_b.key_id, identity_pub_hex=idn_b.public_bytes.hex(),
        fingerprint=idn_b.key_id, consent_state="paired",
        relay_url="https://relay.test", relay_mailbox="mbx_b", now=_NOW,
    )
    # B has NO peer row for A — so on_session_open must drop.

    leg1 = open_session(store_a, idn_a, peer_id=idn_b.key_id, now=_NOW)
    result = on_session_open(store_b, idn_b, envelope=leg1, now=_NOW)
    assert result is None
