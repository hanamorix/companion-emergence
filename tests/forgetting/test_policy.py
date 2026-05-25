# tests/forgetting/test_policy.py
from datetime import UTC, datetime, timedelta

from brain.forgetting.policy import (
    FADE_THRESHOLD,
    LOST_PASS_COUNT,
    LOST_THRESHOLD,
    RECENT_LIVED_HOURS,
    Transition,
    is_exempt,
    next_state,
)
from brain.memory.store import Memory


def _make_memory(*, id="mem_test", state="active", created_iso=None) -> Memory:
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    object.__setattr__(m, "id", id)
    object.__setattr__(m, "state", state)
    if created_iso:
        object.__setattr__(m, "created_at", datetime.fromisoformat(created_iso))
    return m


def test_constants_match_spec():
    assert FADE_THRESHOLD == 0.25
    assert LOST_THRESHOLD == 0.10
    assert LOST_PASS_COUNT == 2
    assert RECENT_LIVED_HOURS == 24.0


def test_active_to_fading_when_below_fade_threshold():
    m = _make_memory(state="active")
    t = next_state(m, salience=0.20, consecutive_low_passes=0)
    assert t == Transition.FADE


def test_active_stays_active_at_threshold():
    m = _make_memory(state="active")
    t = next_state(m, salience=0.25, consecutive_low_passes=0)
    assert t == Transition.NONE


def test_active_stays_active_above_threshold():
    m = _make_memory(state="active")
    t = next_state(m, salience=0.50, consecutive_low_passes=0)
    assert t == Transition.NONE


def test_fading_to_active_when_recalled():
    """recall is signalled by salience rising back above FADE_THRESHOLD."""
    m = _make_memory(state="fading")
    t = next_state(m, salience=0.30, consecutive_low_passes=0)
    assert t == Transition.UNFADE


def test_fading_to_lost_only_after_consecutive_passes_below_lost_threshold():
    m = _make_memory(state="fading")
    # First low pass: salience low but not yet enough consecutive
    t1 = next_state(m, salience=0.05, consecutive_low_passes=1)
    assert t1 == Transition.NONE
    # Second consecutive low pass: cross the threshold + count
    t2 = next_state(m, salience=0.05, consecutive_low_passes=LOST_PASS_COUNT)
    assert t2 == Transition.LOSE


def test_fading_stays_fading_when_salience_above_lost_threshold():
    m = _make_memory(state="fading")
    t = next_state(m, salience=0.15, consecutive_low_passes=5)
    assert t == Transition.NONE


def test_exempt_soul_crystallised_blocks_any_transition():
    m = _make_memory(state="active")
    assert is_exempt(
        m, soul_crystallised_ids={m.id}, under_review_ids=set(), now_lived_age_hours=100.0
    )


def test_exempt_under_review_blocks_any_transition():
    m = _make_memory(state="active")
    assert is_exempt(
        m, soul_crystallised_ids=set(), under_review_ids={m.id}, now_lived_age_hours=100.0
    )


def test_exempt_recent_buffer_blocks_any_transition():
    """A memory created within RECENT_LIVED_HOURS is exempt."""
    # Simulate created 6 hours ago wall-clock. With FeltTimeState
    # showing lived_age=12.0, that's roughly 12h ago lived-age.
    recent_created = (datetime.now(UTC) - timedelta(hours=6)).isoformat()
    m = _make_memory(state="active", created_iso=recent_created)
    # 12h lived-age is < RECENT_LIVED_HOURS=24
    assert is_exempt(
        m,
        soul_crystallised_ids=set(),
        under_review_ids=set(),
        now_lived_age_hours=12.0,
    )


def test_not_exempt_when_old():
    """A memory older than RECENT_LIVED_HOURS in lived-age is fair game."""
    old_created = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    m = _make_memory(state="active", created_iso=old_created)
    assert not is_exempt(
        m,
        soul_crystallised_ids=set(),
        under_review_ids=set(),
        now_lived_age_hours=500.0,
    )


# ---------------------------------------------------------------------------
# is_within_import_grace — migration settling window
# ---------------------------------------------------------------------------


def _mem(created):
    return Memory(id="m", content="c", memory_type="conversation", domain="us",
                  created_at=created)


def test_pre_migration_memory_within_grace_is_exempt():
    from brain.forgetting import policy
    mig = datetime(2026, 5, 1, tzinfo=UTC)
    mem = _mem(datetime(2026, 4, 1, tzinfo=UTC))
    assert policy.is_within_import_grace(
        mem, migrated_at_utc=mig, lived_age_hours_at_migration=100.0,
        current_lived_age_hours=150.0) is True


def test_grace_lapsed_not_exempt():
    from brain.forgetting import policy
    mig = datetime(2026, 5, 1, tzinfo=UTC)
    mem = _mem(datetime(2026, 4, 1, tzinfo=UTC))
    assert policy.is_within_import_grace(
        mem, migrated_at_utc=mig, lived_age_hours_at_migration=100.0,
        current_lived_age_hours=300.0) is False


def test_post_migration_memory_not_import_exempt():
    from brain.forgetting import policy
    mig = datetime(2026, 5, 1, tzinfo=UTC)
    mem = _mem(datetime(2026, 5, 2, tzinfo=UTC))
    assert policy.is_within_import_grace(
        mem, migrated_at_utc=mig, lived_age_hours_at_migration=0.0,
        current_lived_age_hours=1.0) is False


def test_no_manifest_means_no_grace():
    from brain.forgetting import policy
    mem = _mem(datetime(2026, 4, 1, tzinfo=UTC))
    assert policy.is_within_import_grace(
        mem, migrated_at_utc=None, lived_age_hours_at_migration=0.0,
        current_lived_age_hours=1.0) is False


def test_clock_skew_clamped():
    from brain.forgetting import policy
    mig = datetime(2026, 5, 1, tzinfo=UTC)
    mem = _mem(datetime(2026, 4, 1, tzinfo=UTC))
    assert policy.is_within_import_grace(
        mem, migrated_at_utc=mig, lived_age_hours_at_migration=100.0,
        current_lived_age_hours=50.0) is True
