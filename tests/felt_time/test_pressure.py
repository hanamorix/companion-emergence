"""Tests for pressure.py tick aggregation."""

from brain.felt_time.pressure import TickInput, apply_tick
from brain.felt_time.state import Anchor, PressureCounters


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
