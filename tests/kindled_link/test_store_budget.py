from datetime import UTC, datetime, timedelta

from brain.kindled_link import limits
from brain.kindled_link.store import KindledLinkStore


def _store(tmp_path):
    return KindledLinkStore(tmp_path / "k.db")


def test_unknown_peer_starts_full(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    assert s.get_disclosure_budget("kid_a", now) == limits.BUDGET_MAX


def test_debit_reduces_budget(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    s.debit_disclosure_budget("kid_a", 0.4, now)
    assert abs(s.get_disclosure_budget("kid_a", now) - 0.6) < 1e-9


def test_debit_clamps_at_zero(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    s.debit_disclosure_budget("kid_a", 5.0, now)
    assert s.get_disclosure_budget("kid_a", now) == 0.0


def test_refill_over_wall_time(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    s.debit_disclosure_budget("kid_a", 1.0, now)  # → 0.0
    # after 1 day, refills by BUDGET_REFILL_PER_DAY (0.5)
    later = now + timedelta(days=1)
    assert abs(s.get_disclosure_budget("kid_a", later) - limits.BUDGET_REFILL_PER_DAY) < 1e-9
    # never exceeds MAX
    much_later = now + timedelta(days=10)
    assert s.get_disclosure_budget("kid_a", much_later) == limits.BUDGET_MAX


def test_debit_refills_to_now_before_subtracting(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    s.debit_disclosure_budget("kid_a", 1.0, now)  # → 0.0 at `now`
    later = now + timedelta(days=1)  # refilled to 0.5
    s.debit_disclosure_budget("kid_a", 0.2, later)  # 0.5 - 0.2 = 0.3
    assert abs(s.get_disclosure_budget("kid_a", later) - 0.3) < 1e-9
