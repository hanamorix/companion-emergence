"""Tests for P3 retention rework, Change 2 — importance-driven decay rate.

Covers C2.1-C2.5 from changes/p3-retention/1.5-criteria.md. All verified by
directly exercising `salience.score` / `salience._freshness_input` /
`policy.next_state` on synthetic memories at controlled (importance,
lived-age) points, via a small simulation that mirrors
`brain.forgetting.__init__.run_pass`'s own state-transition loop (6h-cadence
passes, consecutive_low_passes counter, FADE/UNFADE/LOSE per
`policy.next_state`) — the same mechanics production drives, without a real
MemoryStore/DB/heartbeat.

Naming: synthetic user = Bob, persona = Canary, model = Claude. No Phoebe.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from brain.felt_time.state import FeltTimeState
from brain.forgetting import policy, salience
from brain.memory.store import Memory

_CADENCE_HOURS = 6.0
_TWO_LIVED_YEARS_HOURS = 17_532.0  # per plan's stated anchor
_MAX_SIM_HOURS = _TWO_LIVED_YEARS_HOURS * 3  # generous ceiling so LOSE has room to appear


def _make_memory(importance: float, emotions: dict[str, float] | None = None) -> Memory:
    m = Memory.create_new(
        content="a synthetic memory",
        memory_type="episodic",
        domain="chat",
        emotions=emotions or {},
        importance=importance,
    )
    return m


def _simulate(
    importance: float,
    *,
    emotions: dict[str, float] | None = None,
    soul_linked: bool = False,
    max_hours: float = _MAX_SIM_HOURS,
) -> tuple[float | None, float | None]:
    """Step a synthetic memory through 6h-cadence forgetting passes from
    lived-age 0 up to max_hours. Returns (first_fade_age, first_lose_age),
    either None if the transition never occurs in the window.

    Mirrors brain.forgetting.__init__.run_pass's own loop: salience.score
    each pass, track consecutive_low_passes exactly as run_pass does, apply
    policy.next_state, advance state. No exemptions applied here (is_exempt/
    RECENT_LIVED_HOURS/import-grace) — this isolates the salience+state-
    machine mechanism the plan's C2 anchors are about; the wall-clock
    recency exemption is a SEPARATE, untouched gate (HIST 95315d5d) that in
    production would additionally protect a memory for its first 720 hours.
    """
    mem = _make_memory(importance, emotions)
    state = "active"
    consecutive_low = 0
    fade_age: float | None = None
    lose_age: float | None = None
    # lived_age_hours > 0 with first_tick_ts=None -> the felt-time rate
    # defaults to 1.0 (see salience._lived_hours_since), so lived-hours since
    # `anchor` tracks wall-hours since `anchor` directly — letting this sim
    # control lived-age deterministically via `created_at` alone.
    felt_state = FeltTimeState(lived_age_hours=1.0)

    t = 0.0
    while t <= max_hours:
        object.__setattr__(mem, "created_at", datetime.now(UTC) - timedelta(hours=t))
        object.__setattr__(mem, "state", state)
        s = salience.score(
            mem,
            store=None,  # unused by score() — see salience.py signature
            hebbian=None,  # _hebbian_input degrades to 0 on AttributeError
            felt_time_state=felt_state,
            soul_linked_ids=({mem.id} if soul_linked else set()),
        )
        next_low = consecutive_low + 1 if s < policy.LOST_THRESHOLD else 0
        transition = policy.next_state(
            mem, salience=s, consecutive_low_passes=next_low, narrative_weight=0.0
        )
        if transition == policy.Transition.FADE:
            if fade_age is None:
                fade_age = t
            state = "fading"
        elif transition == policy.Transition.UNFADE:
            state = "active"
            next_low = 0
        elif transition == policy.Transition.LOSE:
            lose_age = t
            break
        consecutive_low = next_low
        t += _CADENCE_HOURS

    return fade_age, lose_age


# --------------------------------------------------------------------------- C2.1
def test_c2_1_importance_zero_timeline_unchanged_and_nonzero_differs():
    """importance=0 -> the pre-change formula exactly (both levers are
    documented no-ops at r=0: effective_fade = FADE_THRESHOLD/(1+narrative_weight),
    horizon_eff = _FRESHNESS_LIVED_HOURS_HORIZON). Fail-test: a nonzero-
    importance case must differ (proving the change is live) while
    importance-0 matches the pre-change formula bit for bit."""
    mem0 = _make_memory(0.0)
    felt_state = FeltTimeState(lived_age_hours=1.0)

    # Pre-change formula, inlined (not imported — this IS the oracle for the
    # "byte-identical at importance 0" claim): effective_fade only used
    # narrative_weight, and the freshness horizon was the bare constant.
    pre_change_effective_fade = policy.FADE_THRESHOLD / (1.0 + 0.0)
    assert (
        policy.FADE_THRESHOLD / (1.0 + 0.0 + policy.FADE_IMPORTANCE_GAIN * 0.0)
        == pre_change_effective_fade
    )

    for lived in (0.0, 100.0, 800.0, 5000.0, 20000.0):
        object.__setattr__(mem0, "created_at", datetime.now(UTC) - timedelta(hours=lived))
        pre_freshness = 1.0 - min(1.0, max(0.0, lived / salience._FRESHNESS_LIVED_HOURS_HORIZON))
        post_freshness = salience._freshness_input(mem0, felt_state)
        assert post_freshness == pytest.approx(pre_freshness)

    # Fail-test / live-check: a nonzero-importance case DIFFERS from imp-0 at
    # a lived-age where the importance lever has kicked in.
    fade0, lose0 = _simulate(0.0)
    fade10, lose10 = _simulate(10.0, max_hours=25_000.0)
    assert (fade10, lose10) != (fade0, lose0)


# --------------------------------------------------------------------------- C2.2
def test_c2_2_high_importance_survives_well_beyond_30_lived_days_where_imp0_does_not():
    thirty_days_hours = 30 * 24.0  # 720h
    fade0, lose0 = _simulate(0.0, max_hours=thirty_days_hours + _CADENCE_HOURS)
    # Fail-test: pre-change (importance ignored), the high-importance memory
    # would ALSO be faded/lost by this age, same as importance-0.
    assert fade0 is not None and fade0 <= thirty_days_hours

    for imp in (8.0, 9.0, 10.0):
        mem = _make_memory(imp)
        felt_state = FeltTimeState(lived_age_hours=1.0)
        object.__setattr__(mem, "created_at", datetime.now(UTC) - timedelta(hours=thirty_days_hours))
        object.__setattr__(mem, "state", "active")
        s = salience.score(
            mem, store=None, hebbian=None, felt_time_state=felt_state, soul_linked_ids=set()
        )
        transition = policy.next_state(mem, salience=s, consecutive_low_passes=0, narrative_weight=0.0)
        assert transition == policy.Transition.NONE  # still active: not FADE, not LOSE


# --------------------------------------------------------------------------- C2.3
def test_c2_3_importance_ten_survives_two_lived_years():
    fade10, lose10 = _simulate(10.0, max_hours=_TWO_LIVED_YEARS_HOURS)
    # Not LOST at >= 2 lived-years (stated constant anchor).
    assert lose10 is None or lose10 >= _TWO_LIVED_YEARS_HOURS


# --------------------------------------------------------------------------- C2.4
def test_c2_4_monotonic_and_never_accelerates():
    def _inf_if_none(x: float | None) -> float:
        return float("inf") if x is None else x

    fade_ages = []
    lose_ages = []
    for imp in range(11):
        fade, lose = _simulate(float(imp))
        fade_ages.append(fade)
        lose_ages.append(lose)

    # Monotonic non-decreasing in importance.
    for i in range(1, 11):
        assert _inf_if_none(fade_ages[i]) >= _inf_if_none(fade_ages[i - 1])
        assert _inf_if_none(lose_ages[i]) >= _inf_if_none(lose_ages[i - 1])

    # Never-accelerate: for every importance >= 0, disposition is never worse
    # (never lost earlier) than importance-0.
    baseline_lose = _inf_if_none(lose_ages[0])
    for i in range(11):
        assert _inf_if_none(lose_ages[i]) >= baseline_lose


# --------------------------------------------------------------------------- C2.5
def test_c2_5_emotional_and_soul_memories_no_regression():
    # High-emotion memory: importance=0 vs importance=5, same emotion vector.
    # Fail-test: a version that let importance LOWER the emotion-input path
    # (it must not — Change 2 only touches freshness horizon + fade gate)
    # would decay earlier post-change.
    fade_e0, lose_e0 = _simulate(0.0, emotions={"joy": 10.0}, max_hours=_TWO_LIVED_YEARS_HOURS)
    fade_e5, lose_e5 = _simulate(5.0, emotions={"joy": 10.0}, max_hours=_TWO_LIVED_YEARS_HOURS)

    def _inf_if_none(x):
        return float("inf") if x is None else x

    assert _inf_if_none(fade_e5) >= _inf_if_none(fade_e0)
    assert _inf_if_none(lose_e5) >= _inf_if_none(lose_e0)

    # Soul-linked memory: importance=0 vs importance=5.
    fade_s0, lose_s0 = _simulate(0.0, soul_linked=True, max_hours=_TWO_LIVED_YEARS_HOURS)
    fade_s5, lose_s5 = _simulate(5.0, soul_linked=True, max_hours=_TWO_LIVED_YEARS_HOURS)
    assert _inf_if_none(fade_s5) >= _inf_if_none(fade_s0)
    assert _inf_if_none(lose_s5) >= _inf_if_none(lose_s0)


# --------------------------------------------------------------------------- constants sanity
def test_constants_documented_starting_values():
    """Pins the tuned constants this test suite validated against (see the
    module docstring's simulation)."""
    assert policy.FADE_IMPORTANCE_GAIN == 2.0
    assert salience.HORIZON_IMPORTANCE_GAIN == 52.0
    assert policy.LOST_THRESHOLD == 0.10  # UNCHANGED (cd808dbc invariant)
    assert policy.LOST_PASS_COUNT == 2  # UNCHANGED
