from datetime import UTC, datetime

from brain.kindled_link.gate import DenyAllGate, GateDecision, OutboundPayload
from brain.kindled_link.session_engine import SessionEngine
from brain.kindled_link.store import KindledLinkStore


class _StubProvider:
    def complete(self, prompt): return "x"


class _SendSpy:
    def __init__(self): self.sent = []
    def __call__(self, payload): self.sent.append(payload)


class _SendGate:
    def review(self, payload, **kw): return GateDecision(action="send")


class _BoomGate:
    def review(self, payload, **kw): raise RuntimeError("gate exploded")


class _CapturingGate:
    """Returns send, but records the payload it was handed (inv. 127 check)."""
    def __init__(self): self.seen = []
    def review(self, payload, **kw):
        self.seen.append(payload)
        return GateDecision(action="send")


def _eng(tmp_path, gate):
    store = KindledLinkStore(tmp_path / "k.db")
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.create_session("kid_a", "s1", now)
    return SessionEngine(store=store, identity=None, provider=_StubProvider(),
                         gate=gate), store, now


def test_denyall_gate_sends_nothing(tmp_path):
    spy = _SendSpy()
    eng, store, now = _eng(tmp_path, DenyAllGate())
    action = eng.process_outbound(
        peer_id="kid_a", session_id="s1", payload=OutboundPayload(body="hi"),
        reason="autonomous-start", now=now, today="2026-06-20", send_fn=spy)
    assert action == "hold"
    assert spy.sent == []  # CRITERION 2: zero envelopes leave


def test_send_decision_calls_send_and_records(tmp_path):
    spy = _SendSpy()
    eng, store, now = _eng(tmp_path, _SendGate())
    action = eng.process_outbound(
        peer_id="kid_a", session_id="s1", payload=OutboundPayload(body="hi"),
        reason="autonomous-start", now=now, today="2026-06-20", send_fn=spy)
    assert action == "send"
    assert len(spy.sent) == 1
    assert store.get_session("kid_a", "s1")["msg_count"] == 1
    assert store.get_counters("kid_a", "2026-06-20")["outbound_count"] == 1


def test_gate_exception_fails_closed_to_hold(tmp_path):
    spy = _SendSpy()
    eng, store, now = _eng(tmp_path, _BoomGate())
    action = eng.process_outbound(
        peer_id="kid_a", session_id="s1", payload=OutboundPayload(body="hi"),
        reason="autonomous-start", now=now, today="2026-06-20", send_fn=spy)
    assert action == "hold"
    assert spy.sent == []


def test_exhausted_session_cap_refuses_send_even_on_send_decision(tmp_path):
    # red-team Major #2: a 'send' decision must NOT send when the session cap is
    # spent — the predicate is wired into the send path, not just unit-tested.
    spy = _SendSpy()
    eng, store, now = _eng(tmp_path, _SendGate())
    for _ in range(24):  # spend the 24-message session cap
        store.bump_session_outbound("kid_a", "s1", now)
    action = eng.process_outbound(
        peer_id="kid_a", session_id="s1", payload=OutboundPayload(body="hi"),
        reason="autonomous-start", now=now, today="2026-06-20", send_fn=spy)
    assert action == "hold"
    assert spy.sent == []  # refused despite the gate saying send


def test_60s_gap_refuses_send_even_on_send_decision(tmp_path):
    spy = _SendSpy()
    eng, store, now = _eng(tmp_path, _SendGate())
    store.bump_session_outbound("kid_a", "s1", now)  # last outbound = now
    action = eng.process_outbound(
        peer_id="kid_a", session_id="s1", payload=OutboundPayload(body="hi"),
        reason="autonomous-start", now=now, today="2026-06-20", send_fn=spy)
    assert action == "hold"
    assert spy.sent == []


def test_full_payload_incl_relationship_hint_reaches_gate(tmp_path):
    # inv. 127: nothing crosses ungated — the hint must reach the gate with the body.
    gate = _CapturingGate()
    eng, store, now = _eng(tmp_path, gate)
    payload = OutboundPayload(body="hi", relationship_hint={"local_continuity_note": "x"})
    eng.process_outbound(peer_id="kid_a", session_id="s1", payload=payload,
                         reason="autonomous-start", now=now, today="2026-06-20",
                         send_fn=_SendSpy())
    assert gate.seen[0].relationship_hint == {"local_continuity_note": "x"}
