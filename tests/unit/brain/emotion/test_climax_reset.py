"""Unit tests for the climax reset hook in aggregate.py.

Verifies the inviolate properties from spec §7.1:
- Reset is gated on aggregated climax >= 7
- physical-arousal reset uses existing `arousal` (Option A reconciliation)
- arousal *= 0.2 with floor 0.5
- desire *= 0.6
- comfort_seeking += 2 (clamp 10)
- rest_need += 2 (clamp 10)
- Idempotent: applying twice produces same result as once
- Pure: returns NEW EmotionalState, never mutates input
"""

from __future__ import annotations

from brain.emotion.aggregate import _apply_climax_reset
from brain.emotion.state import EmotionalState


def _state(**emotions: float) -> EmotionalState:
    s = EmotionalState()
    for name, v in emotions.items():
        s.set(name, v)
    return s


def test_no_op_when_climax_below_threshold():
    s = _state(climax=6.9, arousal=8.0, desire=8.0)
    out = _apply_climax_reset(s)
    assert out.emotions["arousal"] == 8.0
    assert out.emotions["desire"] == 8.0
    assert out.emotions.get("comfort_seeking", 0.0) == 0.0
    assert out.emotions.get("rest_need", 0.0) == 0.0


def test_arousal_dampened_by_factor_0_2_with_floor_0_5():
    s = _state(climax=8.0, arousal=8.0)
    out = _apply_climax_reset(s)
    # 8.0 * 0.2 = 1.6 (above floor)
    assert abs(out.emotions["arousal"] - 1.6) < 1e-9


def test_arousal_floor_kicks_in_when_starting_low():
    s = _state(climax=8.0, arousal=2.0)
    out = _apply_climax_reset(s)
    # 2.0 * 0.2 = 0.4 → floor 0.5
    assert out.emotions["arousal"] == 0.5


def test_desire_dampened_by_factor_0_6():
    s = _state(climax=8.0, desire=8.0)
    out = _apply_climax_reset(s)
    # 8.0 * 0.6 = 4.8
    assert abs(out.emotions["desire"] - 4.8) < 1e-9


def test_comfort_seeking_raised_by_2_clamp_10():
    s = _state(climax=8.0, comfort_seeking=5.0)
    out = _apply_climax_reset(s)
    assert out.emotions["comfort_seeking"] == 7.0
    # clamp test
    s2 = _state(climax=8.0, comfort_seeking=9.0)
    out2 = _apply_climax_reset(s2)
    assert out2.emotions["comfort_seeking"] == 10.0


def test_rest_need_raised_by_2_clamp_10():
    s = _state(climax=8.0, rest_need=5.0)
    out = _apply_climax_reset(s)
    assert out.emotions["rest_need"] == 7.0
    s2 = _state(climax=8.0, rest_need=9.5)
    out2 = _apply_climax_reset(s2)
    assert out2.emotions["rest_need"] == 10.0


def test_comfort_seeking_added_when_absent():
    """Reset must add comfort_seeking even if not yet set."""
    s = _state(climax=8.0)
    out = _apply_climax_reset(s)
    assert out.emotions["comfort_seeking"] == 2.0


def test_rest_need_added_when_absent():
    s = _state(climax=8.0)
    out = _apply_climax_reset(s)
    assert out.emotions["rest_need"] == 2.0


def test_does_not_mutate_input_state():
    """Pure function — input state must be unchanged after call."""
    s = _state(climax=8.0, arousal=8.0, desire=8.0)
    snapshot_before = dict(s.emotions)
    _apply_climax_reset(s)
    assert dict(s.emotions) == snapshot_before


def test_idempotent_when_climax_still_high():
    """Applying reset twice produces same result as once.

    This is the matrix row #1 invariant: reset never compounds across calls.
    Same input → same output. The "chain" is in time, not in space.
    """
    s = _state(climax=8.0, arousal=8.0, desire=8.0, comfort_seeking=5.0)
    once = _apply_climax_reset(s)
    twice = _apply_climax_reset(once)
    assert once.emotions == twice.emotions


def test_aggregate_state_applies_reset_via_integration():
    """End-to-end: aggregate_state returns post-reset state when climax memories present."""
    from brain.emotion.aggregate import aggregate_state
    from brain.memory.store import Memory

    mem = Memory.create_new(
        memory_type="conversation",
        content="post-climax",
        emotions={"climax": 8.0, "arousal": 8.0, "desire": 8.0},
        domain="general",
    )
    state = aggregate_state([mem])
    # Reset should have applied:
    assert abs(state.emotions["arousal"] - 1.6) < 1e-9
    assert abs(state.emotions["desire"] - 4.8) < 1e-9
    assert state.emotions["comfort_seeking"] == 2.0
    assert state.emotions["rest_need"] == 2.0
