"""Tests for Part B — atomic cap reserve wired into the session engine (T8).

The key invariant: the slot is reserved BEFORE the irreversible action
(provider call / send_fn).  These tests confirm that:
  1. generate_draft reserves the provider slot before calling provider.complete
  2. _regenerate does the same
  3. _act_on_decision reserves the outbound slot before calling send_fn
"""
import contextlib
from datetime import UTC, datetime

from brain.kindled_link.gate import GateDecision, OutboundPayload
from brain.kindled_link.session_engine import SessionEngine
from brain.kindled_link.store import KindledLinkStore


class _SendGate:
    def review(self, payload, **kw):
        return GateDecision(action="send", texture_score=0.1)


class _StubThrottle:
    @contextlib.contextmanager
    def background_slot(self, *, now=None):
        yield True

    def should_yield(self, *, now=None):
        return False


def _store_and_eng(tmp_path, provider, gate=None):
    store = KindledLinkStore(tmp_path / "k.db")
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.create_session("kid_a", "s1", now)
    eng = SessionEngine(
        store=store, identity=None, provider=provider,
        gate=gate or _SendGate(),
        throttle=_StubThrottle(),
    )
    return store, eng, now


def test_generate_draft_reserves_before_provider_call(tmp_path):
    """When the cap is exactly 1 away from full, generate_draft reserves the
    slot atomically BEFORE calling provider.complete, so a concurrent caller
    that tries to reserve while complete() is running gets False."""
    calls = []

    class _SpyProv:
        def complete(self, prompt):
            # At the moment of the call the slot must already be reserved
            calls.append(
                store.get_counters("kid_a", "2026-06-20")["provider_call_count"]
            )
            return "reply"

    store, eng, _ = _store_and_eng(tmp_path, _SpyProv())
    result = eng.generate_draft(
        peer_id="kid_a", session_id="s1",
        persona_voice="v", ambient="a", peer_stage="stranger",
        transcript_summary="t", today="2026-06-20",
    )
    assert result == "reply"
    # The slot was already incremented when provider.complete ran
    assert calls == [1]
    # Final count is still 1 (no double-increment)
    assert store.get_counters("kid_a", "2026-06-20")["provider_call_count"] == 1


def test_regenerate_reserves_before_provider_call(tmp_path):
    """_regenerate must also reserve the provider slot atomically BEFORE the
    revision provider.complete call."""
    calls_at_complete = []

    class _SpyRevProv:
        def complete(self, prompt):
            calls_at_complete.append(
                store.get_counters("kid_a", "2026-06-20")["provider_call_count"]
            )
            return "revised"

    class _ReviseThenSend:
        def __init__(self): self.n = 0
        def review(self, payload, **kw):
            self.n += 1
            if self.n == 1:
                return GateDecision(action="revise", revision_constraints="less")
            return GateDecision(action="send", texture_score=0.1)

    store, eng, now = _store_and_eng(tmp_path, _SpyRevProv(), gate=_ReviseThenSend())
    eng.process_outbound(
        peer_id="kid_a", session_id="s1",
        payload=OutboundPayload(body="original"),
        reason="r", now=now, today="2026-06-20",
        send_fn=lambda p: None,
    )
    # The slot was already incremented when provider.complete ran during revision
    assert calls_at_complete == [1]
    assert store.get_counters("kid_a", "2026-06-20")["provider_call_count"] == 1


def test_send_reserves_outbound_before_send_fn(tmp_path):
    """The outbound slot must be reserved atomically BEFORE send_fn is called,
    so a race cannot exceed the daily outbound cap."""
    counts_at_send = []

    def _spy_send(payload):
        counts_at_send.append(
            store.get_counters("kid_a", "2026-06-20")["outbound_count"]
        )

    store, eng, now = _store_and_eng(tmp_path, _StubProvider())
    eng.process_outbound(
        peer_id="kid_a", session_id="s1",
        payload=OutboundPayload(body="hi"),
        reason="r", now=now, today="2026-06-20",
        send_fn=_spy_send,
    )
    # The slot was already incremented when send_fn ran
    assert counts_at_send == [1]
    assert store.get_counters("kid_a", "2026-06-20")["outbound_count"] == 1


class _StubProvider:
    def complete(self, prompt):
        return "reply"
