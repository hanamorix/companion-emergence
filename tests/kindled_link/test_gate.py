from brain.kindled_link.gate import DenyAllGate, GateDecision, OutboundPayload


def test_denyall_gate_always_holds():
    gate = DenyAllGate()
    decision = gate.review(
        OutboundPayload(body="hello peer", relationship_hint={"note": "x"}),
        peer_id="kid_abc",
        stage="stranger",
        transcript_summary="(none)",
        reason="autonomous-start",
    )
    assert isinstance(decision, GateDecision)
    assert decision.action == "hold"


def test_gate_decision_actions_are_the_four_known():
    # documents the contract for Phase 4
    for action in ("send", "revise", "hold", "end_or_pause"):
        assert GateDecision(action=action).action == action
