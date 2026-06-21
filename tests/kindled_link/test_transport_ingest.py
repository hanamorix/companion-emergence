"""T4 — inbound poll_and_ingest tests.

Tests are added one at a time per TDD discipline.
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from starlette.testclient import TestClient

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.limits import INBOUND_FLOOD_CAP
from brain.kindled_link.protocol import (
    ROLE_INITIATOR,
    ROLE_RESPONDER,
    build_envelope,
)
from brain.kindled_link.relay_client import RelayClient
from brain.kindled_link.store import KindledLinkStore
from brain.kindled_link.transport import poll_and_ingest, send_message
from relay.dev_relay import create_app

_NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
_TTL = timedelta(days=7)


def _idn_pair():
    idn_a = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    idn_b = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))))
    return idn_a, idn_b


def _make_relay_pair(idn_a, idn_b):
    """Shared in-process relay; returns (rc_a, rc_b)."""
    app = create_app(require_auth=True)
    http = TestClient(app, base_url="http://relay.test")
    rc_a = RelayClient(http, identity=idn_a, mailbox_id="mbx_a")
    rc_b = RelayClient(http, identity=idn_b, mailbox_id="mbx_b")
    rc_a.register()
    rc_b.register()
    return rc_a, rc_b


def _paired_stores(tmp_path, idn_a, idn_b, session_key: bytes, session_id: str):
    """Both stores paired + session_keys rows on BOTH sides."""
    sa = KindledLinkStore(tmp_path / "a.db")
    sb = KindledLinkStore(tmp_path / "b.db")
    sa.upsert_peer(
        peer_id=idn_b.key_id,
        identity_pub_hex=idn_b.public_bytes.hex(),
        fingerprint=idn_b.key_id,
        consent_state="paired",
        relay_url="https://relay.test",
        relay_mailbox="mbx_b",
        now=_NOW,
    )
    sb.upsert_peer(
        peer_id=idn_a.key_id,
        identity_pub_hex=idn_a.public_bytes.hex(),
        fingerprint=idn_a.key_id,
        consent_state="paired",
        relay_url="https://relay.test",
        relay_mailbox="mbx_a",
        now=_NOW,
    )
    # A is initiator, B is responder
    sa.save_session_key(
        peer_id=idn_b.key_id, session_id=session_id,
        session_key=session_key, my_role=ROLE_INITIATOR, peer_role=ROLE_RESPONDER,
        now=_NOW,
    )
    sb.save_session_key(
        peer_id=idn_a.key_id, session_id=session_id,
        session_key=session_key, my_role=ROLE_RESPONDER, peer_role=ROLE_INITIATOR,
        now=_NOW,
    )
    return sa, sb


# B0-2: replay rejection — sequence ≤ stored high-water → REPLAY, no second row

def test_replayed_message_rejected_no_second_transcript(tmp_path):
    """A replayed envelope (seq ≤ high-water) is rejected with REPLAY,
    no second transcript row is written, and the high-water survives reopen."""
    idn_a, idn_b = _idn_pair()
    rc_a, rc_b = _make_relay_pair(idn_a, idn_b)
    sk = b"\xBB" * 32
    sid = "ks_replay"
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b, sk, sid)

    # Send one real message from A to B
    send_message(sa, idn_a, rc_a, peer_id=idn_b.key_id, session_id=sid,
                 payload={"text": "first"}, now=_NOW)

    # B ingests — one transcript row
    summary = poll_and_ingest(sb, idn_b, rc_b, now=_NOW)
    assert summary["accepted"].get(idn_a.key_id, 0) == 1
    assert len(sb.recent_transcript(idn_a.key_id)) == 1

    # Build a replay envelope with the same sequence (seq=1) directly
    # and push it again (simulating an attacker replaying the envelope).
    replay_env = build_envelope(
        payload={"text": "replay"},
        sender=idn_a,
        recipient_key_id=idn_b.key_id,
        relay_mailbox="mbx_b",
        session_id=sid,
        sequence=1,  # same seq as the first message
        role=ROLE_INITIATOR,
        session_key=sk,
        now=_NOW,
        ttl=_TTL,
    )
    rc_a.push(replay_env)

    # B ingests again — replay must be rejected, no new transcript row
    summary2 = poll_and_ingest(sb, idn_b, rc_b, now=_NOW)
    assert summary2["accepted"].get(idn_a.key_id, 0) == 0
    assert len(sb.recent_transcript(idn_a.key_id)) == 1  # still one row

    # High-water persists through a store reopen
    sb.close()
    sb2 = KindledLinkStore(tmp_path / "b.db")
    assert sb2.get_seq_high_water(idn_a.key_id, sid) == 1


# B3: tampered envelope → correct RejectReason, no transcript

def test_tampered_envelope_rejected_no_transcript(tmp_path):
    """An envelope with a bad signature is rejected, no transcript row written."""
    idn_a, idn_b = _idn_pair()
    rc_a, rc_b = _make_relay_pair(idn_a, idn_b)
    sk = b"\xCC" * 32
    sid = "ks_tamper"
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b, sk, sid)

    # Build a valid envelope then corrupt the signature
    env = build_envelope(
        payload={"text": "tamper me"},
        sender=idn_a,
        recipient_key_id=idn_b.key_id,
        relay_mailbox="mbx_b",
        session_id=sid,
        sequence=1,
        role=ROLE_INITIATOR,
        session_key=sk,
        now=_NOW,
        ttl=_TTL,
    )
    # Corrupt the signature (flip first byte)
    bad_sig = "ff" + env["signature"][2:]
    env["signature"] = bad_sig
    rc_a.push(env)

    summary = poll_and_ingest(sb, idn_b, rc_b, now=_NOW)
    assert summary["accepted"].get(idn_a.key_id, 0) == 0
    assert len(sb.recent_transcript(idn_a.key_id)) == 0


# C1: flood cap — INBOUND_FLOOD_CAP + N envelopes, only cap processed, N left un-acked

def test_flood_cap_limits_processed_envelopes(tmp_path):
    """A peer-group of INBOUND_FLOOD_CAP + 5 → exactly CAP processed, 5 left un-acked,
    peer marked degraded."""
    idn_a, idn_b = _idn_pair()
    rc_a, rc_b = _make_relay_pair(idn_a, idn_b)
    sk = b"\xDD" * 32
    sid = "ks_flood"
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b, sk, sid)

    excess = 5
    total = INBOUND_FLOOD_CAP + excess

    # Push total envelopes from A → B's mailbox, each with a unique seq
    for seq in range(1, total + 1):
        env = build_envelope(
            payload={"text": f"msg {seq}"},
            sender=idn_a,
            recipient_key_id=idn_b.key_id,
            relay_mailbox="mbx_b",
            session_id=sid,
            sequence=seq,
            role=ROLE_INITIATOR,
            session_key=sk,
            now=_NOW,
            ttl=_TTL,
        )
        rc_a.push(env)

    summary = poll_and_ingest(sb, idn_b, rc_b, now=_NOW)

    # Exactly CAP transcript rows accepted
    assert summary["accepted"].get(idn_a.key_id, 0) == INBOUND_FLOOD_CAP
    assert len(sb.recent_transcript(idn_a.key_id, limit=total + 1)) == INBOUND_FLOOD_CAP

    # Peer marked degraded
    assert idn_a.key_id in summary["degraded"]

    # 5 envelopes remain un-acked on the relay (B can still fetch them)
    remaining = rc_b.fetch()
    assert len(remaining) == excess


# C2: verify_and_open call count ≤ INBOUND_FLOOD_CAP in a flooded poll

def test_flood_cap_bounds_verify_and_open_call_count(tmp_path):
    """The number of verify_and_open/decrypt calls in a flooded poll is ≤ cap.
    This is the critical invariant: the flood cap bounds DECRYPT WORK, not just
    transcript rows."""
    idn_a, idn_b = _idn_pair()
    rc_a, rc_b = _make_relay_pair(idn_a, idn_b)
    sk = b"\xEE" * 32
    sid = "ks_flood_c2"
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b, sk, sid)

    excess = 3
    total = INBOUND_FLOOD_CAP + excess

    for seq in range(1, total + 1):
        env = build_envelope(
            payload={"text": f"msg {seq}"},
            sender=idn_a,
            recipient_key_id=idn_b.key_id,
            relay_mailbox="mbx_b",
            session_id=sid,
            sequence=seq,
            role=ROLE_INITIATOR,
            session_key=sk,
            now=_NOW,
            ttl=_TTL,
        )
        rc_a.push(env)

    call_count = []

    import brain.kindled_link.transport as transport_mod

    original_vao = transport_mod.verify_and_open

    def counting_vao(*args, **kwargs):
        call_count.append(1)
        return original_vao(*args, **kwargs)

    with patch.object(transport_mod, "verify_and_open", side_effect=counting_vao):
        poll_and_ingest(sb, idn_b, rc_b, now=_NOW)

    assert len(call_count) <= INBOUND_FLOOD_CAP, (
        f"verify_and_open called {len(call_count)} times — must be ≤ {INBOUND_FLOOD_CAP}"
    )


# B0-3: no session key → dropped, logged inbound_no_session, no transcript

def test_no_session_key_drops_and_acks(tmp_path):
    """An envelope for a session with no key is dropped, logged inbound_no_session,
    no transcript row written. The envelope IS acked (terminal — can never be
    processed, leaving it would re-poll forever)."""
    idn_a, idn_b = _idn_pair()
    rc_a, rc_b = _make_relay_pair(idn_a, idn_b)
    sk = b"\xFF" * 32
    sid = "ks_no_key_on_b"

    # Only A's store has the session key; B has no session key for this session.
    sa = KindledLinkStore(tmp_path / "a.db")
    sb = KindledLinkStore(tmp_path / "b.db")
    sa.upsert_peer(
        peer_id=idn_b.key_id,
        identity_pub_hex=idn_b.public_bytes.hex(),
        fingerprint=idn_b.key_id,
        consent_state="paired",
        relay_url="https://relay.test",
        relay_mailbox="mbx_b",
        now=_NOW,
    )
    sb.upsert_peer(
        peer_id=idn_a.key_id,
        identity_pub_hex=idn_a.public_bytes.hex(),
        fingerprint=idn_a.key_id,
        consent_state="paired",
        relay_url="https://relay.test",
        relay_mailbox="mbx_a",
        now=_NOW,
    )
    sa.save_session_key(
        peer_id=idn_b.key_id, session_id=sid,
        session_key=sk, my_role=ROLE_INITIATOR, peer_role=ROLE_RESPONDER,
        now=_NOW,
    )
    # B deliberately has NO session_keys row for this session.

    send_message(sa, idn_a, rc_a, peer_id=idn_b.key_id, session_id=sid,
                 payload={"text": "secret"}, now=_NOW)

    summary = poll_and_ingest(sb, idn_b, rc_b, now=_NOW)

    # No transcript row
    assert len(sb.recent_transcript(idn_a.key_id)) == 0
    # Nothing accepted
    assert summary["accepted"].get(idn_a.key_id, 0) == 0
    # The envelope was acked (mailbox is empty after poll)
    remaining = rc_b.fetch()
    assert len(remaining) == 0


# session_open routing: leg-1 inbound → on_session_open fires → reply pushed

def test_session_open_inbound_triggers_on_session_open_and_pushes_reply(tmp_path):
    """When poll_and_ingest sees a session_open envelope with no pending handshake,
    it calls on_session_open (responder leg-2) and pushes the reply to the relay."""
    from brain.kindled_link.session import open_session

    idn_a, idn_b = _idn_pair()
    rc_a, rc_b = _make_relay_pair(idn_a, idn_b)

    sa = KindledLinkStore(tmp_path / "a.db")
    sb = KindledLinkStore(tmp_path / "b.db")

    # Pre-seed the local mailbox IDs to match the relay client mailbox IDs.
    # get_or_create_local_mailbox generates a random id on first call; we must
    # ensure the stored mailbox matches what the RelayClient uses so the reply
    # envelope's relay_mailbox routes to the correct mailbox.
    sa._conn.execute(
        "INSERT INTO local_identity (key, value) VALUES ('relay_mailbox', 'mbx_a')"
    )
    sa._conn.commit()
    sb._conn.execute(
        "INSERT INTO local_identity (key, value) VALUES ('relay_mailbox', 'mbx_b')"
    )
    sb._conn.commit()

    # A knows B, B knows A (for on_session_open to find the peer)
    sa.upsert_peer(
        peer_id=idn_b.key_id,
        identity_pub_hex=idn_b.public_bytes.hex(),
        fingerprint=idn_b.key_id,
        consent_state="paired",
        relay_url="https://relay.test",
        relay_mailbox="mbx_b",
        now=_NOW,
    )
    sb.upsert_peer(
        peer_id=idn_a.key_id,
        identity_pub_hex=idn_a.public_bytes.hex(),
        fingerprint=idn_a.key_id,
        consent_state="paired",
        relay_url="https://relay.test",
        relay_mailbox="mbx_a",
        now=_NOW,
    )

    # A generates leg-1 and pushes it to B's mailbox
    leg1 = open_session(sa, idn_a, peer_id=idn_b.key_id, now=_NOW)
    session_id = leg1["session_id"]
    rc_a.push(leg1)

    # B's poll_and_ingest should trigger on_session_open and push the reply
    summary = poll_and_ingest(sb, idn_b, rc_b, now=_NOW)

    # B must now have a session_keys row (responder established the key)
    sk_b = sb.get_session_key(idn_a.key_id, session_id)
    assert sk_b is not None, "B should have a session_keys row after processing leg-1"

    # A's mailbox must have the leg-2 reply pushed by poll_and_ingest
    a_inbox = rc_a.fetch()
    assert len(a_inbox) == 1, "A's mailbox should contain the leg-2 reply"
    reply_env = a_inbox[0]["envelope"]
    assert "session_open" in reply_env, "Reply must be a session_open envelope"
    assert reply_env["session_id"] == session_id
