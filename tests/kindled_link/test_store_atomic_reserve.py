"""Tests for atomic cap-reserve methods (T8 Part A).

try_reserve_outbound / try_reserve_provider: single-statement
UPDATE … WHERE col < cap RETURNING col.  A race between two concurrent
writers can never exceed the cap because SQLite serialises writers and the
check+increment is one atomic statement.
"""
from brain.kindled_link.store import KindledLinkStore


def _store(tmp_path):
    return KindledLinkStore(tmp_path / "k.db")


def test_try_reserve_outbound_allows_up_to_cap(tmp_path):
    """Reserve slots 0→1 (True), 1→2 (True), then 2 blocks at cap=2 (False).
    The stored count must never exceed cap even after the False."""
    s = _store(tmp_path)
    today = "2026-06-20"
    assert s.try_reserve_outbound("kid_a", today, cap=2) is True
    assert s.try_reserve_outbound("kid_a", today, cap=2) is True
    assert s.try_reserve_outbound("kid_a", today, cap=2) is False
    # stored count is exactly cap — never exceeded
    assert s.get_counters("kid_a", today)["outbound_count"] == 2


def test_try_reserve_outbound_independent_per_peer(tmp_path):
    """Reserves for different peers are independent."""
    s = _store(tmp_path)
    today = "2026-06-20"
    s.try_reserve_outbound("kid_a", today, cap=1)
    assert s.try_reserve_outbound("kid_a", today, cap=1) is False
    # kid_b is untouched — can still reserve
    assert s.try_reserve_outbound("kid_b", today, cap=1) is True


def test_try_reserve_outbound_independent_per_day(tmp_path):
    """A new day resets the counter — reserve is allowed again."""
    s = _store(tmp_path)
    s.try_reserve_outbound("kid_a", "2026-06-20", cap=1)
    assert s.try_reserve_outbound("kid_a", "2026-06-20", cap=1) is False
    # new day: fresh slate
    assert s.try_reserve_outbound("kid_a", "2026-06-21", cap=1) is True


def test_try_reserve_provider_allows_up_to_cap(tmp_path):
    """Same semantics for the provider-call counter."""
    s = _store(tmp_path)
    today = "2026-06-20"
    assert s.try_reserve_provider("kid_a", today, cap=2) is True
    assert s.try_reserve_provider("kid_a", today, cap=2) is True
    assert s.try_reserve_provider("kid_a", today, cap=2) is False
    assert s.get_counters("kid_a", today)["provider_call_count"] == 2


def test_try_reserve_provider_independent_from_outbound(tmp_path):
    """Outbound and provider counters are separate columns; exhausting one
    must not affect the other."""
    s = _store(tmp_path)
    today = "2026-06-20"
    # exhaust provider cap=1
    s.try_reserve_provider("kid_a", today, cap=1)
    assert s.try_reserve_provider("kid_a", today, cap=1) is False
    # outbound counter is still at 0 — can still reserve
    assert s.try_reserve_outbound("kid_a", today, cap=1) is True


def test_release_provider_slot_decrements_floored_at_zero(tmp_path):
    """release_provider_slot refunds a reserved provider slot (so a gate/reflection
    call that reserves-then-defers/errors nets zero), floored at 0 — never goes
    negative even if released more than reserved."""
    s = _store(tmp_path)
    today = "2026-06-20"
    s.try_reserve_provider("kid_a", today, cap=5)
    s.try_reserve_provider("kid_a", today, cap=5)  # count == 2
    s.release_provider_slot("kid_a", today)
    assert s.get_counters("kid_a", today)["provider_call_count"] == 1
    s.release_provider_slot("kid_a", today)
    assert s.get_counters("kid_a", today)["provider_call_count"] == 0
    s.release_provider_slot("kid_a", today)  # floored — no negative count
    assert s.get_counters("kid_a", today)["provider_call_count"] == 0
