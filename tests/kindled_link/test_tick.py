"""T6 — run_kindled_link_tick: §14 through-path + D1/D4 off-by-default tests.

One test at a time (tdd-guard). Uses real identities, real in-process dev relay,
real 3-leg handshake — no live LLM. Spy gate / spy provider / spy throttle.
"""
from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from starlette.testclient import TestClient

from brain.kindled_link import relationship as _rel
from brain.kindled_link import transport as _transport
from brain.kindled_link.gate import GateDecision
from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.relay_client import RelayClient
from brain.kindled_link.session import open_session
from brain.kindled_link.store import KindledLinkStore
from brain.kindled_link.tick import run_kindled_link_tick
from brain.kindled_link.transport import poll_and_ingest, send_message
from brain.persona_config import PersonaConfig
from relay.dev_relay import create_app

_NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
_TODAY = _NOW.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Shared fixtures
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
    # Pre-seed local mailbox IDs so open_session's sender_mailbox matches the
    # RelayClient mailbox registered with the relay (same pattern as
    # test_transport_ingest.py:318-329 — without this the leg-2 reply envelope
    # routes to a random mbx_xxx that nobody polls, so the 3-leg handshake
    # never completes and send_message returns False silently).
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


def _spy_throttle():
    """A throttle spy that always grants background slots (fail-open)."""
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


def _spy_gate(action="hold"):
    """A gate spy that records review() calls and returns a fixed decision."""
    calls = []

    class SpyGate:
        def review(self, payload, *, peer_id, stage, transcript_summary, reason, now, today):
            calls.append({"peer_id": peer_id, "stage": stage, "action": action})
            return GateDecision(action=action)

    g = SpyGate()
    g._calls = calls
    return g


def _spy_provider(draft="Hello from A"):
    """A provider spy that records complete() calls."""
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


def _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, *, tmp_path):
    """Run the full 3-leg handshake via the relay. Returns session_id."""
    leg1 = open_session(sa, idn_a, peer_id=idn_b.key_id, now=_NOW)
    session_id = leg1["session_id"]
    rc_a.push(leg1)
    poll_and_ingest(sb, idn_b, rc_b, now=_NOW, persona_dir=tmp_path / "b")
    poll_and_ingest(sa, idn_a, rc_a, now=_NOW, persona_dir=tmp_path / "a")
    return session_id


# ---------------------------------------------------------------------------
# §14 A1 — inbound transcript row written after run_kindled_link_tick
# ---------------------------------------------------------------------------

def test_through_path_a1_inbound_transcript_written(tmp_path):
    """§14 A1: after A sends a real encrypted message, one run_kindled_link_tick
    for B writes exactly one transcript row for that peer/session.
    Addresses are taken from the persisted handshake, not literals.
    """
    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    session_id = _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    # A sends a real encrypted message to B
    send_message(sa, idn_a, rc_a,
                 peer_id=idn_b.key_id, session_id=session_id,
                 payload={"text": "hello peer"}, now=_NOW)

    persona_dir_b = tmp_path / "b"
    persona_dir_b.mkdir(parents=True, exist_ok=True)

    run_kindled_link_tick(
        persona_dir_b,
        store=sb,
        identity=idn_b,
        relay_client=rc_b,
        provider=_spy_provider(),
        config=_enabled_config(),
        now=_NOW,
    )

    # A1: exactly one transcript row
    rows = sb.recent_transcript(idn_a.key_id)
    assert len(rows) == 1, f"Expected 1 transcript row, got {len(rows)}"
    assert rows[0]["text"] == "hello peer"
    assert rows[0]["session_id"] == session_id


# ---------------------------------------------------------------------------
# §14 A2 — gate consulted AND relationship stage read during the tick
# ---------------------------------------------------------------------------

def test_through_path_a2_gate_and_relationship_consulted(tmp_path):
    """§14 A2: the tick calls gate.review() AND reads relationship stage
    (get_stage / get_relationship_row).  A 'hold' outcome is fine — the
    assertion is that the gate ran and the relationship was read, not that
    a send happened.  Spy gate injected via the tick's optional gate= param.
    """
    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    session_id = _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    send_message(sa, idn_a, rc_a,
                 peer_id=idn_b.key_id, session_id=session_id,
                 payload={"text": "hi there"}, now=_NOW)

    spy_gate = _spy_gate(action="hold")

    get_stage_calls = []
    original_get_stage = _rel.get_stage

    def _spy_get_stage(store, peer_id):
        get_stage_calls.append(peer_id)
        return original_get_stage(store, peer_id)

    persona_dir_b = tmp_path / "b"
    persona_dir_b.mkdir(parents=True, exist_ok=True)

    with patch("brain.kindled_link.session_engine.relationship.get_stage", _spy_get_stage):
        run_kindled_link_tick(
            persona_dir_b,
            store=sb,
            identity=idn_b,
            relay_client=rc_b,
            provider=_spy_provider(),
            gate=spy_gate,
            config=_enabled_config(),
            now=_NOW,
        )

    # A2a: gate.review was called at least once (hold outcome is acceptable)
    assert len(spy_gate._calls) >= 1, "gate.review must be called during the tick"
    # A2b: relationship stage was read (get_stage called at least once)
    assert len(get_stage_calls) >= 1, "get_stage must be read during the tick"


# ---------------------------------------------------------------------------
# §14 E4 — provider draft call takes a cli_throttle background slot
# ---------------------------------------------------------------------------

def test_through_path_e4_draft_takes_throttle_background_slot(tmp_path):
    """§14 E4: when the tick drives a draft (generate_draft), the provider call
    acquires a cli_throttle background slot.  Verified via a spy throttle
    injected through the tick's throttle= parameter.
    """
    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    session_id = _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    send_message(sa, idn_a, rc_a,
                 peer_id=idn_b.key_id, session_id=session_id,
                 payload={"text": "hello throttle"}, now=_NOW)

    spy_throttle = _spy_throttle()

    persona_dir_b = tmp_path / "b"
    persona_dir_b.mkdir(parents=True, exist_ok=True)

    run_kindled_link_tick(
        persona_dir_b,
        store=sb,
        identity=idn_b,
        relay_client=rc_b,
        provider=_spy_provider(draft="throttled draft"),
        gate=_spy_gate(action="hold"),
        throttle=spy_throttle,
        config=_enabled_config(),
        now=_NOW,
    )

    # E4: the throttle background_slot was acquired at least once (for the draft)
    assert len(spy_throttle._calls) >= 1, (
        "generate_draft must acquire a cli_throttle background slot"
    )


# ---------------------------------------------------------------------------
# D1 — disabled flag: no outbound send, no autonomous start
# ---------------------------------------------------------------------------

def test_d1_disabled_no_outbound_no_start(tmp_path):
    """D1: with kindled_link_enabled=False the tick performs NO outbound send and
    NO autonomous start for a paired peer, even if there is an active session with
    an inbound message and caps are available.  send_message is spied to confirm
    it is NOT called for outbound paths.
    """
    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    session_id = _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    # A sends a message so B has an inbound to potentially respond to
    send_message(sa, idn_a, rc_a,
                 peer_id=idn_b.key_id, session_id=session_id,
                 payload={"text": "trigger me"}, now=_NOW)

    send_calls = []
    original_send = _transport.send_message

    def _spy_send(store, identity, relay_client, *, peer_id, session_id, payload, now,
                  persona_dir=None, **kw):
        send_calls.append({"peer_id": peer_id, "payload": payload})
        return original_send(store, identity, relay_client, peer_id=peer_id,
                             session_id=session_id, payload=payload, now=now,
                             persona_dir=persona_dir or tmp_path / "b", **kw)

    persona_dir_b = tmp_path / "b"
    persona_dir_b.mkdir(parents=True, exist_ok=True)

    with patch("brain.kindled_link.tick.send_message", _spy_send):
        run_kindled_link_tick(
            persona_dir_b,
            store=sb,
            identity=idn_b,
            relay_client=rc_b,
            provider=_spy_provider(),
            gate=_spy_gate(action="hold"),
            config=_disabled_config(),
            now=_NOW,
        )

    # D1: no outbound send_message calls were made
    assert send_calls == [], (
        f"D1 violation: send_message called {len(send_calls)} time(s) when disabled"
    )


# ---------------------------------------------------------------------------
# D4 — disabled flag: inbound messages are STILL ingested
# ---------------------------------------------------------------------------

def test_d4_disabled_inbound_still_ingested(tmp_path):
    """D4: with kindled_link_enabled=False, receiving + recording an inbound
    message is NOT gated.  A transcript row must be written even when the
    autonomous outbound path is disabled.
    """
    idn_a, idn_b = _make_identities()
    rc_a, rc_b = _make_relay(idn_a, idn_b)
    sa, sb = _paired_stores(tmp_path, idn_a, idn_b)

    session_id = _run_handshake(sa, idn_a, sb, idn_b, rc_a, rc_b, tmp_path=tmp_path)

    # A sends a message to B's relay mailbox
    send_message(sa, idn_a, rc_a,
                 peer_id=idn_b.key_id, session_id=session_id,
                 payload={"text": "still receive me"}, now=_NOW)

    persona_dir_b = tmp_path / "b"
    persona_dir_b.mkdir(parents=True, exist_ok=True)

    # Run tick with kindled_link DISABLED
    run_kindled_link_tick(
        persona_dir_b,
        store=sb,
        identity=idn_b,
        relay_client=rc_b,
        provider=_spy_provider(),
        config=_disabled_config(),
        now=_NOW,
    )

    # D4: the inbound message was still written to the transcript
    rows = sb.recent_transcript(idn_a.key_id)
    assert len(rows) == 1, (
        f"D4 violation: expected 1 transcript row when disabled, got {len(rows)}"
    )
    assert rows[0]["text"] == "still receive me"
