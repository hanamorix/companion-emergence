"""Tests for pressure.py tick aggregation."""

from brain.felt_time.pressure import TickInput, apply_horizon_tick, apply_tick
from brain.felt_time.state import Anchor, HorizonBucket, PressureCounters


def test_apply_tick_increments_when_no_new_anchors():
    before = PressureCounters(heartbeats=5, chat_turns=2, reflex_firings=0, wall_clock_s=900.0)
    after = apply_tick(
        before,
        tick=TickInput(heartbeats=1, chat_turns=3, reflex_firings=2, wall_clock_s_delta=900.0),
        new_anchors=[],
    )
    assert after.heartbeats == 6
    assert after.chat_turns == 5
    assert after.reflex_firings == 2
    assert after.wall_clock_s == 1800.0


def test_apply_tick_resets_on_anchor():
    before = PressureCounters(heartbeats=42, chat_turns=10, reflex_firings=3, wall_clock_s=3600.0)
    new = Anchor(
        type="dream",
        ts="2026-05-17T22:00:00+00:00",
        label="the boat one",
        source_ref="dreams.log.jsonl:1",
    )
    after = apply_tick(
        before,
        tick=TickInput(heartbeats=1, chat_turns=0, reflex_firings=0, wall_clock_s_delta=900.0),
        new_anchors=[new],
    )
    # Reset to zero on anchor, then current tick is NOT counted (the anchor
    # IS this tick, so by definition nothing accumulated since it).
    assert after == PressureCounters()


def test_apply_tick_with_multiple_anchors_uses_latest():
    before = PressureCounters(heartbeats=42)
    older = Anchor(type="dream", ts="2026-05-17T20:00:00+00:00", label="a", source_ref="x:1")
    newer = Anchor(type="growth", ts="2026-05-17T21:00:00+00:00", label="b", source_ref="y:1")
    after = apply_tick(
        before,
        tick=TickInput(heartbeats=1, chat_turns=0, reflex_firings=0, wall_clock_s_delta=900.0),
        new_anchors=[newer, older],  # unordered input
    )
    assert after == PressureCounters()


def test_apply_tick_cold_start_zero():
    after = apply_tick(
        PressureCounters(),
        tick=TickInput(heartbeats=0, chat_turns=0, reflex_firings=0, wall_clock_s_delta=0.0),
        new_anchors=[],
    )
    assert after == PressureCounters()


def test_apply_horizon_tick_initialises_missing_buckets():
    result = apply_horizon_tick(
        {},
        tick=TickInput(heartbeats=1, chat_turns=2, reflex_firings=0, wall_clock_s_delta=60.0),
        now_ts="2026-06-08T10:00:00+00:00",
    )
    assert "week" in result
    assert "month" in result
    assert result["week"].counters.chat_turns == 2
    assert result["month"].counters.heartbeats == 1


def test_apply_horizon_tick_accumulates_without_rollover():
    start_ts = "2026-06-02T10:00:00+00:00"  # 6 days before now
    now_ts = "2026-06-08T10:00:00+00:00"
    buckets = {
        "week": HorizonBucket(
            counters=PressureCounters(chat_turns=5, heartbeats=10),
            prev_counters=PressureCounters(),
            period_start_ts=start_ts,
        ),
        "month": HorizonBucket(
            counters=PressureCounters(chat_turns=20),
            prev_counters=PressureCounters(),
            period_start_ts=start_ts,
        ),
    }
    result = apply_horizon_tick(
        buckets,
        tick=TickInput(heartbeats=1, chat_turns=1, reflex_firings=0, wall_clock_s_delta=60.0),
        now_ts=now_ts,
    )
    assert result["week"].counters.chat_turns == 6   # accumulated, NOT reset
    assert result["week"].prev_counters.chat_turns == 0  # no rollover yet
    assert result["month"].counters.chat_turns == 21


def test_apply_horizon_tick_rolls_over_week_at_7_days():
    start_ts = "2026-06-01T10:00:00+00:00"  # exactly 7 days before
    now_ts = "2026-06-08T10:00:00+00:00"
    buckets = {
        "week": HorizonBucket(
            counters=PressureCounters(chat_turns=10, heartbeats=50),
            prev_counters=PressureCounters(chat_turns=3),
            period_start_ts=start_ts,
        ),
        "month": HorizonBucket(
            counters=PressureCounters(chat_turns=20),
            prev_counters=PressureCounters(),
            period_start_ts=start_ts,
        ),
    }
    result = apply_horizon_tick(
        buckets,
        tick=TickInput(heartbeats=1, chat_turns=1, reflex_firings=0, wall_clock_s_delta=60.0),
        now_ts=now_ts,
    )
    # Week rolled: prev = old current (10), new current = tick only (1)
    assert result["week"].prev_counters.chat_turns == 10
    assert result["week"].counters.chat_turns == 1
    assert result["week"].period_start_ts == now_ts
    # Month has not rolled (only 7 days, needs 30)
    assert result["month"].counters.chat_turns == 21


def test_apply_horizon_tick_rolls_over_month_at_30_days():
    start_ts = "2026-05-09T10:00:00+00:00"  # 30 days before
    now_ts = "2026-06-08T10:00:00+00:00"
    buckets = {
        "week": HorizonBucket(
            counters=PressureCounters(chat_turns=5),
            prev_counters=PressureCounters(),
            period_start_ts="2026-06-02T10:00:00+00:00",  # 6 days — no week rollover
        ),
        "month": HorizonBucket(
            counters=PressureCounters(chat_turns=30, heartbeats=200),
            prev_counters=PressureCounters(chat_turns=20),
            period_start_ts=start_ts,
        ),
    }
    result = apply_horizon_tick(
        buckets,
        tick=TickInput(heartbeats=1, chat_turns=1, reflex_firings=0, wall_clock_s_delta=60.0),
        now_ts=now_ts,
    )
    assert result["month"].prev_counters.chat_turns == 30
    assert result["month"].counters.chat_turns == 1
    assert result["week"].counters.chat_turns == 6   # week just accumulated


def test_apply_horizon_tick_fails_open_on_malformed_ts():
    # Malformed now_ts — should return buckets unchanged, not raise
    buckets = {
        "week": HorizonBucket(
            counters=PressureCounters(chat_turns=3),
            prev_counters=PressureCounters(),
            period_start_ts="2026-06-01T00:00:00+00:00",
        ),
    }
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = apply_horizon_tick(
            buckets,
            tick=TickInput(heartbeats=1, chat_turns=1, reflex_firings=0, wall_clock_s_delta=60.0),
            now_ts="not-a-timestamp",
        )
    assert len(w) == 1
    assert "malformed" in str(w[0].message).lower()
    # Buckets returned unchanged (fail open)
    assert result["week"].counters.chat_turns == 3


def test_apply_horizon_tick_malformed_period_start_ts_resets():
    buckets = {
        "week": HorizonBucket(
            counters=PressureCounters(chat_turns=5),
            prev_counters=PressureCounters(),
            period_start_ts="not-a-date",  # malformed
        ),
        "month": HorizonBucket(
            counters=PressureCounters(chat_turns=10),
            prev_counters=PressureCounters(),
            period_start_ts="2026-05-01T00:00:00+00:00",
        ),
    }
    now_ts = "2026-06-08T10:00:00+00:00"
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = apply_horizon_tick(
            buckets,
            tick=TickInput(heartbeats=1, chat_turns=2, reflex_firings=0, wall_clock_s_delta=60.0),
            now_ts=now_ts,
        )
    # Warning emitted for the malformed bucket
    assert any("malformed" in str(warning.message).lower() for warning in w)
    # Accumulated (not rolled — no prev_counters change)
    assert result["week"].counters.chat_turns == 7  # 5 + 2
    assert result["week"].prev_counters.chat_turns == 0  # no rollover
    # Corrupt timestamp replaced with now_ts
    assert result["week"].period_start_ts == now_ts
