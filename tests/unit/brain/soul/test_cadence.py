"""Tests for brain/soul/cadence.py — persisted self-pacing soul-review cadence."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.soul.cadence import compute_next_state

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_NORMAL = 6 * 3600.0


def test_compute_next_state_paces_by_outcome():
    # Clean drain (no backlog, no failures) -> normal 6h interval, failures reset.
    clean = compute_next_state(
        now=_NOW, model_failures=0, eligible_pending=0,
        normal_interval_s=_NORMAL, prev_failures=0,
    )
    assert clean.next_review_at == _NOW + timedelta(seconds=_NORMAL)
    assert clean.consecutive_failures == 0

    # Backlog remains, calls working -> 30-min catch-up to drain fast.
    backlog = compute_next_state(
        now=_NOW, model_failures=0, eligible_pending=12,
        normal_interval_s=_NORMAL, prev_failures=0,
    )
    assert backlog.next_review_at == _NOW + timedelta(minutes=30)
    assert backlog.consecutive_failures == 0

    # Model failures -> escalating backoff (30m * 2^(cf-1)), failures increment.
    f1 = compute_next_state(
        now=_NOW, model_failures=3, eligible_pending=5,
        normal_interval_s=_NORMAL, prev_failures=0,
    )
    assert f1.next_review_at == _NOW + timedelta(minutes=30)
    assert f1.consecutive_failures == 1

    f2 = compute_next_state(
        now=_NOW, model_failures=3, eligible_pending=5,
        normal_interval_s=_NORMAL, prev_failures=1,
    )
    assert f2.next_review_at == _NOW + timedelta(hours=1)
    assert f2.consecutive_failures == 2

    # Backoff is capped at the normal interval (won't exceed 6h).
    fcap = compute_next_state(
        now=_NOW, model_failures=1, eligible_pending=0,
        normal_interval_s=_NORMAL, prev_failures=10,
    )
    assert fcap.next_review_at == _NOW + timedelta(seconds=_NORMAL)
    assert fcap.consecutive_failures == 11
