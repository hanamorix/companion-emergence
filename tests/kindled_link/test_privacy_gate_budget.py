import contextlib
from datetime import UTC, datetime

from brain.kindled_link.gate import GateDecision, OutboundPayload
from brain.kindled_link.privacy_gate import PrivacyGate, _apply_budget
from brain.kindled_link.store import KindledLinkStore


def test_apply_budget_downgrades_send_when_depleted():
    d = GateDecision(action="send", texture_score=0.3)
    out = _apply_budget(d, budget=0.1)  # below threshold 0.25
    assert out.action == "revise"


def test_apply_budget_leaves_send_when_ample():
    d = GateDecision(action="send", texture_score=0.3)
    out = _apply_budget(d, budget=0.9)
    assert out.action == "send"


def test_apply_budget_never_loosens():
    # a hold stays hold even with a full budget
    d = GateDecision(action="hold")
    assert _apply_budget(d, budget=1.0).action == "hold"


def test_review_downgrades_send_when_budget_depleted(tmp_path):
    class _P:
        def complete(self, prompt): return '{"decision":"send","texture_score":0.2}'

    class _GrantThrottle:
        @contextlib.contextmanager
        def background_slot(self, *, now=None):
            yield True
        def should_yield(self, *, now=None):
            return False

    store = KindledLinkStore(tmp_path / "k.db")
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.debit_disclosure_budget("kid_a", 1.0, now)  # → 0.0, depleted
    gate = PrivacyGate(provider=_P(), store=store, throttle=_GrantThrottle())
    d = gate.review(OutboundPayload(body="broad texture about my user"),
                    peer_id="kid_a", stage="stranger", transcript_summary="(none)",
                    reason="r", now=now, today="2026-06-20")
    assert d.action == "revise"  # send downgraded by depleted budget
