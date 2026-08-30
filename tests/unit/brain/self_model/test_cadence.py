"""Tests for brain/self_model/cadence.py — wall-clock reflection cadence."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.self_model.cadence import (
    SelfModelCadenceState,
    compute_next_state,
    is_due,
    load,
    save,
)


def test_is_due_true_when_never_run(tmp_path):
    state = load(tmp_path)
    now = datetime.now(UTC)
    assert is_due(state, now=now)


def test_is_due_true_after_long_wall_clock_gap(tmp_path):
    """After a very long gap since last run, is_due returns True."""
    now = datetime.now(UTC)
    long_ago = now - timedelta(hours=24)
    state = SelfModelCadenceState(next_reflection_at=long_ago, consecutive_failures=0)
    assert is_due(state, now=now)


def test_fires_once_not_repeatedly(tmp_path):
    """After compute_next_state advances, is_due is False until next interval."""
    now = datetime.now(UTC)
    state = SelfModelCadenceState(next_reflection_at=None, consecutive_failures=0)
    # Was due
    assert is_due(state, now=now)
    # Advance after a clean tick
    next_state = compute_next_state(state, outcome="clean", now=now)
    # Should NOT be due immediately after
    assert not is_due(next_state, now=now)


def test_backoff_grows_on_repeated_failure():
    """Each failure doubles the backoff interval."""
    now = datetime.now(UTC)
    state0 = SelfModelCadenceState(next_reflection_at=None, consecutive_failures=0)
    s1 = compute_next_state(state0, outcome="failure", now=now)
    s2 = compute_next_state(s1, outcome="failure", now=now)
    # Each successive failure extends the next_reflection_at further out
    assert s1.next_reflection_at is not None
    assert s2.next_reflection_at is not None
    assert s2.next_reflection_at > s1.next_reflection_at
    assert s1.consecutive_failures == 1
    assert s2.consecutive_failures == 2


def test_clean_resets_failures_to_base():
    """A clean tick after failures resets consecutive_failures to 0 and uses base interval."""
    now = datetime.now(UTC)
    state_with_failures = SelfModelCadenceState(
        next_reflection_at=now - timedelta(hours=1),
        consecutive_failures=5,
    )
    clean_state = compute_next_state(state_with_failures, outcome="clean", now=now)
    assert clean_state.consecutive_failures == 0
    # Interval should be the base 6-hour interval
    from brain.self_model.cadence import _BASE_INTERVAL_S
    expected_at = now + timedelta(seconds=_BASE_INTERVAL_S)
    delta = abs((clean_state.next_reflection_at - expected_at).total_seconds())
    assert delta < 1.0


def test_backlog_uses_catchup_interval():
    """Backlog outcome uses the short catch-up interval."""
    from brain.self_model.cadence import _CATCHUP_INTERVAL_S
    now = datetime.now(UTC)
    state = SelfModelCadenceState(next_reflection_at=None, consecutive_failures=0)
    next_state = compute_next_state(state, outcome="backlog", now=now)
    assert next_state.consecutive_failures == 0
    expected_at = now + timedelta(seconds=_CATCHUP_INTERVAL_S)
    delta = abs((next_state.next_reflection_at - expected_at).total_seconds())
    assert delta < 1.0


def test_deferred_uses_short_retry_interval_and_leaves_failures_unchanged():
    """C8: a throttle-denial ("deferred") outcome sets a short retry interval
    (not the ~6h normal interval, and not the escalating failure backoff),
    and leaves consecutive_failures UNCHANGED — a deferral is not a provider
    failure, so it must neither reset nor increment the backoff counter
    (mirrors brain/maker/__init__.py's "no cooldown, no error" framing for
    the identical ThrottleDeferred situation)."""
    from brain.self_model.cadence import _DEFERRED_RETRY_INTERVAL_S

    now = datetime.now(UTC)
    state = SelfModelCadenceState(next_reflection_at=None, consecutive_failures=0)
    next_state = compute_next_state(state, outcome="deferred", now=now)
    assert next_state.consecutive_failures == 0
    expected_at = now + timedelta(seconds=_DEFERRED_RETRY_INTERVAL_S)
    delta = abs((next_state.next_reflection_at - expected_at).total_seconds())
    assert delta < 1.0


def test_deferred_does_not_reset_existing_failure_count():
    """The "leaves consecutive_failures unchanged" guarantee must hold on
    both sides — including when there's an existing nonzero count (a
    deferral must neither reset it back to 0, matching a clean tick's
    behavior, nor increment it, matching a failure's)."""
    now = datetime.now(UTC)
    state = SelfModelCadenceState(next_reflection_at=None, consecutive_failures=3)
    next_state = compute_next_state(state, outcome="deferred", now=now)
    assert next_state.consecutive_failures == 3


def test_round_trip_persist(tmp_path):
    """Cadence state survives save → load."""
    now = datetime.now(UTC)
    state = SelfModelCadenceState(
        next_reflection_at=now + timedelta(hours=6),
        consecutive_failures=2,
    )
    save(tmp_path, state)
    loaded = load(tmp_path)
    assert loaded.consecutive_failures == 2
    assert loaded.next_reflection_at is not None
    # Timestamps should agree to within a second (ISO round-trip)
    delta = abs((loaded.next_reflection_at - state.next_reflection_at).total_seconds())
    assert delta < 1.0
