from datetime import UTC, datetime

from brain.kindled_link.gate import DenyAllGate, GateDecision, OutboundPayload


def test_denyall_gate_always_holds():
    gate = DenyAllGate()
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    decision = gate.review(
        OutboundPayload(body="hello peer", relationship_hint={"note": "x"}),
        peer_id="kid_abc",
        stage="stranger",
        transcript_summary="(none)",
        reason="autonomous-start",
        now=now, today="2026-06-20",
    )
    assert isinstance(decision, GateDecision)
    assert decision.action == "hold"


def test_gate_decision_actions_are_the_four_known():
    # documents the contract for Phase 4
    for action in ("send", "revise", "hold", "end_or_pause"):
        assert GateDecision(action=action).action == action


def test_gate_decision_carries_texture_score_default_zero():
    d = GateDecision(action="send")
    assert d.texture_score == 0.0
    d2 = GateDecision(action="send", texture_score=0.4)
    assert d2.texture_score == 0.4


def test_denyall_gate_accepts_now_today_and_still_holds():
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    decision = DenyAllGate().review(
        OutboundPayload(body="hi"), peer_id="kid_a", stage="stranger",
        transcript_summary="(none)", reason="x", now=now, today="2026-06-20",
    )
    assert decision.action == "hold"
    assert decision.texture_score == 0.0
