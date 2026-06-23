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
    # m10: a FULLY depleted budget (0.0 < BUDGET_DEPLETED_THRESHOLD) now hard-stops
    # to hold (was revise pre-m10). The tighten-band revise case is covered by
    # test_m10_apply_budget_revises_in_tighten_band.
    assert d.action == "hold"


# --- m10: budget depletion -> hold (guarded-change kindled-link-gate-m9-m10) ---

import pytest  # noqa: E402

from brain.kindled_link import limits  # noqa: E402


def test_m10_apply_budget_holds_when_depleted():
    # C-m10-1: below the depletion floor, a model 'send' becomes hold (not revise).
    d = GateDecision(action="send", texture_score=0.3)
    out = _apply_budget(d, budget=limits.BUDGET_DEPLETED_THRESHOLD - 0.001)
    assert out.action == "hold"


def test_m10_apply_budget_revises_in_tighten_band():
    # C-m10-2: between floor and tighten threshold, send -> revise (unchanged).
    d = GateDecision(action="send", texture_score=0.3)
    out = _apply_budget(d, budget=0.1)  # 0.02 <= 0.1 < 0.25
    assert out.action == "revise"


def test_m10_apply_budget_sends_when_ample():
    # C-m10-3: at/above the tighten threshold, send stays send.
    d = GateDecision(action="send", texture_score=0.3)
    assert _apply_budget(d, budget=0.9).action == "send"


def test_m10_floor_is_exclusive_below():
    # C-m10-5: exactly at the floor you can still afford the min debit -> revise.
    d = GateDecision(action="send", texture_score=0.3)
    out = _apply_budget(d, budget=limits.BUDGET_DEPLETED_THRESHOLD)
    assert out.action == "revise"


@pytest.mark.parametrize("action", ["hold", "revise", "end_or_pause"])
@pytest.mark.parametrize("budget", [0.0, 0.1, 0.9])
def test_m10_never_loosens_non_send(action, budget):
    # C-INV-1: _apply_budget never converts a non-send into send, at any budget.
    d = GateDecision(action=action, texture_score=0.3)
    assert _apply_budget(d, budget=budget).action == action


def test_m10_review_holds_when_budget_depleted(tmp_path):
    # C-m10-4: through PrivacyGate.review with a stored depleted budget read at the
    # SAME now (no refill) and a model 'send' verdict -> hold, not revise.
    class _P:
        def complete(self, prompt):
            return '{"decision":"send","texture_score":0.2}'

    class _GrantThrottle:
        @contextlib.contextmanager
        def background_slot(self, *, now=None):
            yield True

        def should_yield(self, *, now=None):
            return False

    store = KindledLinkStore(tmp_path / "k.db")
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.debit_disclosure_budget("kid_a", 1.0, now)  # -> 0.0, depleted
    gate = PrivacyGate(provider=_P(), store=store, throttle=_GrantThrottle())
    d = gate.review(OutboundPayload(body="broad texture about my user"),
                    peer_id="kid_a", stage="stranger", transcript_summary="(none)",
                    reason="r", now=now, today="2026-06-20")
    assert d.action == "hold"  # depleted budget hard-stops, no revised send
