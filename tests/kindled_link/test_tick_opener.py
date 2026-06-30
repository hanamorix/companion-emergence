"""Kindled-link no-flow fix — autonomous content opener tests.

Added one at a time per TDD discipline. See spec:
  docs/superpowers/specs/2026-06-29-kindled-no-flow-opener-design.md
"""
from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import MagicMock

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from starlette.testclient import TestClient

from brain.kindled_link.gate import GateDecision
from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.protocol import ROLE_INITIATOR, ROLE_RESPONDER
from brain.kindled_link.relay_client import RelayClient
from brain.kindled_link.session import open_session
from brain.kindled_link.store import KindledLinkStore
from brain.kindled_link.tick import run_kindled_link_tick
from brain.kindled_link.transport import poll_and_ingest
from brain.persona_config import PersonaConfig
from relay.dev_relay import create_app

_NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)
_TODAY = _NOW.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_identities():
    idn_a = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    idn_b = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))))
    return idn_a, idn_b


def _make_relay(idn_a, idn_b):
    app = create_app(require_auth=True)
    http = TestClient(app, base_url="http://relay.test")
    rc_a = RelayClient(http, identity=idn_a, mailbox_id="mbx_a")
    rc_b = RelayClient(http, identity=idn_b, mailbox_id="mbx_b")
    rc_a.register()
    rc_b.register()
    return rc_a, rc_b


def _paired_stores(tmp_path, idn_a, idn_b, suffix_a="a.db", suffix_b="b.db"):
    sa = KindledLinkStore(tmp_path / suffix_a)
    sb = KindledLinkStore(tmp_path / suffix_b)
    sa._conn.execute(
        "INSERT INTO local_identity (key, value) VALUES ('relay_mailbox', 'mbx_a')"
    )
    sa._conn.commit()
    sb._conn.execute(
        "INSERT INTO local_identity (key, value) VALUES ('relay_mailbox', 'mbx_b')"
    )
    sb._conn.commit()
    sa.upsert_peer(
        peer_id=idn_b.key_id, identity_pub_hex=idn_b.public_bytes.hex(),
        fingerprint=idn_b.key_id, consent_state="paired",
        relay_url="https://relay.test", relay_mailbox="mbx_b", now=_NOW,
    )
    sb.upsert_peer(
        peer_id=idn_a.key_id, identity_pub_hex=idn_a.public_bytes.hex(),
        fingerprint=idn_a.key_id, consent_state="paired",
        relay_url="https://relay.test", relay_mailbox="mbx_a", now=_NOW,
    )
    return sa, sb


def _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, *, tmp_path):
    leg1 = open_session(sa, idn_a, peer_id=idn_b.key_id, now=_NOW)
    session_id = leg1["session_id"]
    rc_a.push(leg1)
    poll_and_ingest(sb, idn_b, rc_b, now=_NOW, persona_dir=tmp_path / "b")
    poll_and_ingest(sa, idn_a, rc_a, now=_NOW, persona_dir=tmp_path / "a")
    return session_id


def _spy_throttle():
    calls = []

    @contextlib.contextmanager
    def background_slot(**_kwargs):
        calls.append("acquire")
        yield True

    mock = MagicMock()
    mock.background_slot = background_slot
    mock.should_yield = MagicMock(return_value=False)
    mock._calls = calls
    return mock


def _spy_gate(action="send"):
    calls = []

    class SpyGate:
        def review(self, payload, *, peer_id, stage, transcript_summary, reason, now, today):
            calls.append({"peer_id": peer_id, "action": action, "reason": reason})
            return GateDecision(action=action)

    g = SpyGate()
    g._calls = calls
    return g


def _spy_provider(draft="Hello from the other side."):
    calls = []

    class SpyProvider:
        def complete(self, prompt):
            calls.append(prompt)
            return draft

    p = SpyProvider()
    p._calls = calls
    return p


def _enabled_config():
    return PersonaConfig(kindled_link_enabled=True)


def _disabled_config():
    return PersonaConfig(kindled_link_enabled=False)


# ---------------------------------------------------------------------------
# Test 1: initiator opens first content when no inbound and no prior outbound
# ---------------------------------------------------------------------------

def test_initiator_opens_first_content(tmp_path):
    """After the handshake, if A is the initiator and has no inbound and no
    prior outbound, a tick for A must compose and send a content opener
    → an outbound draft with status='send' exists for this session."""
    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    session_id = _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    # Verify A is the initiator
    sk = sa.get_session_key(idn_b.key_id, session_id)
    assert sk is not None
    assert sk["my_role"] == ROLE_INITIATOR

    # No inbound, no prior outbound
    assert sa.recent_transcript(idn_b.key_id) == []

    persona_dir_a = tmp_path / "a"
    persona_dir_a.mkdir(parents=True, exist_ok=True)

    run_kindled_link_tick(
        persona_dir_a,
        store=sa,
        identity=idn_a,
        relay_client=rc_a,
        provider=_spy_provider(),
        gate=_spy_gate(action="send"),
        throttle=_spy_throttle(),
        config=_enabled_config(),
        now=_NOW,
    )

    drafts = sa._conn.execute(
        "SELECT * FROM outbound_drafts WHERE peer_id = ? AND session_id = ?",
        (idn_b.key_id, session_id),
    ).fetchall()
    assert len(drafts) >= 1, "Initiator must save a draft for the opener"
    statuses = [dict(d)["status"] for d in drafts]
    assert "send" in statuses, (
        f"Expected a 'send' draft (gate=send, throttle grants), got statuses: {statuses}"
    )


# ---------------------------------------------------------------------------
# Test 2: responder does NOT compose an opener
# ---------------------------------------------------------------------------

def test_responder_does_not_open(tmp_path):
    """B is the responder. With no inbound and no prior outbound, a tick for B
    must NOT call generate_draft and must leave no outbound drafts."""
    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    session_id = _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    # Verify B is the responder
    sk = sb.get_session_key(idn_a.key_id, session_id)
    assert sk is not None
    assert sk["my_role"] == ROLE_RESPONDER

    assert sb.recent_transcript(idn_a.key_id) == []

    persona_dir_b = tmp_path / "b"
    persona_dir_b.mkdir(parents=True, exist_ok=True)

    provider = _spy_provider()
    run_kindled_link_tick(
        persona_dir_b,
        store=sb,
        identity=idn_b,
        relay_client=rc_b,
        provider=provider,
        gate=_spy_gate(action="send"),
        throttle=_spy_throttle(),
        config=_enabled_config(),
        now=_NOW,
    )

    assert provider._calls == [], (
        "Responder must NOT call generate_draft when there is no inbound"
    )
    drafts = sb._conn.execute(
        "SELECT * FROM outbound_drafts WHERE peer_id = ? AND session_id = ?",
        (idn_a.key_id, session_id),
    ).fetchall()
    assert drafts == [], (
        f"Responder must not save opener drafts; got: {[dict(d) for d in drafts]}"
    )


# ---------------------------------------------------------------------------
# Test 3: single-opener idempotent — second tick with no new inbound skips
# ---------------------------------------------------------------------------

def test_opener_is_single(tmp_path):
    """After the initiator sends one opener, a second tick (still no inbound)
    must NOT compose a second draft — the outbound transcript row gates it."""
    from datetime import timedelta

    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    persona_dir_a = tmp_path / "a"
    persona_dir_a.mkdir(parents=True, exist_ok=True)

    provider = _spy_provider()
    gate = _spy_gate(action="send")
    throttle = _spy_throttle()

    # First tick — opener fires
    run_kindled_link_tick(
        persona_dir_a,
        store=sa,
        identity=idn_a,
        relay_client=rc_a,
        provider=provider,
        gate=gate,
        throttle=throttle,
        config=_enabled_config(),
        now=_NOW,
    )

    calls_after_first = len(provider._calls)
    assert calls_after_first >= 1, "First tick must call generate_draft for opener"

    # Second tick — no inbound, opener already sent (outbound transcript row exists)
    now2 = _NOW + timedelta(minutes=10)
    run_kindled_link_tick(
        persona_dir_a,
        store=sa,
        identity=idn_a,
        relay_client=rc_a,
        provider=provider,
        gate=gate,
        throttle=throttle,
        config=_enabled_config(),
        now=now2,
    )

    extra = len(provider._calls) - calls_after_first
    assert extra == 0, (
        f"Second tick must NOT compose another opener; generate_draft called {extra} extra time(s)"
    )


# ---------------------------------------------------------------------------
# Test 4: opener held when throttle denies background slot
# ---------------------------------------------------------------------------

def _spy_throttle_no_slots():
    """A throttle that never grants background slots."""
    @contextlib.contextmanager
    def background_slot(**_kwargs):
        yield False

    mock = MagicMock()
    mock.background_slot = background_slot
    mock.should_yield = MagicMock(return_value=False)
    return mock


def test_opener_respects_caps(tmp_path):
    """When the throttle denies the background slot, generate_draft returns None
    → no outbound transcript row and no 'send' draft (opener deferred, not sent)."""
    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    persona_dir_a = tmp_path / "a"
    persona_dir_a.mkdir(parents=True, exist_ok=True)

    run_kindled_link_tick(
        persona_dir_a,
        store=sa,
        identity=idn_a,
        relay_client=rc_a,
        provider=_spy_provider(),
        gate=_spy_gate(action="send"),
        throttle=_spy_throttle_no_slots(),
        config=_enabled_config(),
        now=_NOW,
    )

    # generate_draft returned None → no draft row saved (None short-circuits save_draft)
    drafts = sa._conn.execute(
        "SELECT * FROM outbound_drafts WHERE peer_id = ?",
        (idn_b.key_id,),
    ).fetchall()
    send_drafts = [d for d in drafts if dict(d)["status"] == "send"]
    assert send_drafts == [], (
        f"When throttle denies, no 'send' draft should exist; got: {[dict(d) for d in drafts]}"
    )


# ---------------------------------------------------------------------------
# Test 5: opener suppressed when disabled (inside enabled gate)
# ---------------------------------------------------------------------------

def test_opener_suppressed_when_disabled(tmp_path):
    """With kindled_link_enabled=False the opener branch is never reached;
    generate_draft is not called, no draft is saved."""
    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    persona_dir_a = tmp_path / "a"
    persona_dir_a.mkdir(parents=True, exist_ok=True)

    provider = _spy_provider()
    run_kindled_link_tick(
        persona_dir_a,
        store=sa,
        identity=idn_a,
        relay_client=rc_a,
        provider=provider,
        gate=_spy_gate(action="send"),
        throttle=_spy_throttle(),
        config=_disabled_config(),
        now=_NOW,
    )

    assert provider._calls == [], (
        "generate_draft must NOT be called when kindled_link_enabled=False"
    )
    drafts = sa._conn.execute(
        "SELECT * FROM outbound_drafts WHERE peer_id = ?", (idn_b.key_id,)
    ).fetchall()
    assert drafts == [], (
        f"No opener drafts when disabled; got: {[dict(d) for d in drafts]}"
    )


# ---------------------------------------------------------------------------
# Test 6: two-persona through-test — content appears on both sides
# ---------------------------------------------------------------------------

def test_two_personas_exchange_content(tmp_path):
    """Causal-toggle regression test for the no-flow root.

    Before fix: handshake completes, both sides silent forever.
    After fix: A (initiator) opens on first tick → B polls, receives it,
    replies → A polls B's reply. Content rows appear on both sides.

    Uses real in-process relay; SpyProvider returns plain strings; gate=send.
    """
    from datetime import timedelta

    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    persona_dir_a = tmp_path / "a"
    persona_dir_a.mkdir(parents=True, exist_ok=True)
    persona_dir_b = tmp_path / "b"
    persona_dir_b.mkdir(parents=True, exist_ok=True)

    session_id = _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    # Sanity: A is initiator, B is responder
    assert sa.get_session_key(idn_b.key_id, session_id)["my_role"] == ROLE_INITIATOR
    assert sb.get_session_key(idn_a.key_id, session_id)["my_role"] == ROLE_RESPONDER

    gate_send = _spy_gate(action="send")
    throttle = _spy_throttle()
    provider_a = _spy_provider(draft="Hello from A!")
    provider_b = _spy_provider(draft="Hello from B!")

    # 4 rounds: A tick (opens/replies) → B tick (polls inbound + replies) → A polls B's reply
    for i in range(4):
        tick_time = _NOW + timedelta(minutes=i * 6)  # exceed 5-min cadence gap

        run_kindled_link_tick(
            persona_dir_a,
            store=sa,
            identity=idn_a,
            relay_client=rc_a,
            provider=provider_a,
            gate=gate_send,
            throttle=throttle,
            config=_enabled_config(),
            now=tick_time,
        )

        # B's tick: run_kindled_link_tick does poll_and_ingest inside, so B
        # receives A's message and replies in the same tick.
        run_kindled_link_tick(
            persona_dir_b,
            store=sb,
            identity=idn_b,
            relay_client=rc_b,
            provider=provider_b,
            gate=gate_send,
            throttle=throttle,
            config=_enabled_config(),
            now=tick_time,
        )

        # A polls to receive B's reply
        poll_and_ingest(sa, idn_a, rc_a, now=tick_time, persona_dir=persona_dir_a)

    # A must have sent at least one opener (outbound draft with status='send')
    a_sent_drafts = sa._conn.execute(
        "SELECT * FROM outbound_drafts WHERE peer_id = ? AND status = 'send'",
        (idn_b.key_id,),
    ).fetchall()
    assert len(a_sent_drafts) >= 1, (
        f"A (initiator) must have at least 1 sent draft; all drafts: "
        f"{[dict(d) for d in sa._conn.execute('SELECT * FROM outbound_drafts WHERE peer_id=?', (idn_b.key_id,)).fetchall()]}"
    )

    # B must have received at least one inbound message from A
    b_inbound = [r for r in sb.recent_transcript(idn_a.key_id, limit=20)
                 if r["direction"] == "inbound"]
    assert len(b_inbound) >= 1, (
        f"B must have at least 1 inbound row from A; transcript: "
        f"{sb.recent_transcript(idn_a.key_id, limit=20)}"
    )

    # B must have sent at least one reply (outbound draft on B's store)
    b_sent_drafts = sb._conn.execute(
        "SELECT * FROM outbound_drafts WHERE peer_id = ? AND status = 'send'",
        (idn_a.key_id,),
    ).fetchall()
    assert len(b_sent_drafts) >= 1, (
        f"B (responder) must have replied at least once; drafts: "
        f"{[dict(d) for d in sb._conn.execute('SELECT * FROM outbound_drafts WHERE peer_id=?', (idn_a.key_id,)).fetchall()]}"
    )
