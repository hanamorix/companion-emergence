"""The gate's provider-cap accounting is ATOMIC (try_reserve_provider) and
refunds the reserved slot when the LLM call never completes (throttle defer or
provider error) — so only completed calls net-count against DAILY_PROVIDER_CAP,
and two concurrent gate/reflection calls for one peer can't over-spend the cap
(the old check-then-incr read+write was racy). #7a/Phase-4 minor.
"""
import contextlib
from datetime import UTC, datetime

from brain.kindled_link.gate import OutboundPayload
from brain.kindled_link.privacy_gate import PrivacyGate
from brain.kindled_link.store import KindledLinkStore

_TODAY = "2026-06-20"
_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


class _SpyStore(KindledLinkStore):
    def __init__(self, path):
        super().__init__(path)
        self.calls = []

    def try_reserve_provider(self, *a, **k):
        self.calls.append("try_reserve_provider")
        return super().try_reserve_provider(*a, **k)

    def incr_provider_count(self, *a, **k):
        self.calls.append("incr_provider_count")
        return super().incr_provider_count(*a, **k)

    def release_provider_slot(self, *a, **k):
        self.calls.append("release_provider_slot")
        return super().release_provider_slot(*a, **k)


class _SendP:
    def complete(self, prompt):
        return '{"decision":"send","texture_score":0.1}'


class _GrantThrottle:
    @contextlib.contextmanager
    def background_slot(self, *, now=None):
        yield True

    def should_yield(self, *, now=None):
        return False


def _review(gate):
    return gate.review(
        OutboundPayload(body="hello peer"), peer_id="kid_a", stage="stranger",
        transcript_summary="(none)", reason="r", now=_NOW, today=_TODAY,
    )


def test_gate_uses_atomic_reserve_and_counts_one_on_success(tmp_path):
    spy = _SpyStore(tmp_path / "k.db")
    gate = PrivacyGate(provider=_SendP(), store=spy, throttle=_GrantThrottle())
    _review(gate)
    assert "try_reserve_provider" in spy.calls  # atomic path, not get+incr
    assert "incr_provider_count" not in spy.calls
    assert "release_provider_slot" not in spy.calls  # success → no refund
    assert spy.get_counters("kid_a", _TODAY)["provider_call_count"] == 1


class _DeferThrottle:
    @contextlib.contextmanager
    def background_slot(self, *, now=None):
        yield False  # slot denied → call never happens

    def should_yield(self, *, now=None):
        return False


def test_gate_releases_reserved_slot_on_throttle_defer(tmp_path):
    spy = _SpyStore(tmp_path / "k.db")
    gate = PrivacyGate(provider=_SendP(), store=spy, throttle=_DeferThrottle())
    d = _review(gate)
    assert d.action == "hold"
    assert "try_reserve_provider" in spy.calls
    assert "release_provider_slot" in spy.calls  # refunded — defer ≠ failure
    assert spy.get_counters("kid_a", _TODAY)["provider_call_count"] == 0


class _ErrP:
    def complete(self, prompt):
        raise RuntimeError("boom")


def test_gate_releases_reserved_slot_on_provider_error(tmp_path):
    spy = _SpyStore(tmp_path / "k.db")
    gate = PrivacyGate(provider=_ErrP(), store=spy, throttle=_GrantThrottle())
    d = _review(gate)
    assert d.action == "hold"  # fail-closed
    assert "release_provider_slot" in spy.calls  # call attempted then failed → refund
    assert spy.get_counters("kid_a", _TODAY)["provider_call_count"] == 0
