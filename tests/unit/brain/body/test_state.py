"""Unit tests for brain/body/state.py — compute_body_state pure function.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.1.
"""

from __future__ import annotations

from datetime import UTC, datetime

from brain.body.state import BodyState, compute_body_state


def _now() -> datetime:
    return datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC)


def test_body_state_to_dict_shape():
    bs = BodyState(
        energy=8,
        temperature=4,
        exhaustion=0,
        session_hours=0.0,
        days_since_contact=0.0,
        body_emotions={
            "arousal": 0.0,
            "desire": 0.0,
            "climax": 0.0,
            "touch_hunger": 0.0,
            "comfort_seeking": 0.0,
            "rest_need": 0.0,
        },
        computed_at=_now(),
    )
    d = bs.to_dict()
    assert d["loaded"] is True
    assert d["energy"] == 8
    assert d["temperature"] == 4
    assert d["exhaustion"] == 0
    assert d["session_hours"] == 0.0
    assert d["days_since_contact"] == 0.0
    assert set(d["body_emotions"].keys()) == {
        "arousal",
        "desire",
        "climax",
        "touch_hunger",
        "comfort_seeking",
        "rest_need",
    }
    assert "computed_at" in d
    assert d["computed_at"].endswith("+00:00") or d["computed_at"].endswith("Z")


def test_energy_baseline_no_inputs():
    bs = compute_body_state(
        emotions={},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.energy == 8


def test_energy_session_band_60_to_120():
    bs = compute_body_state(
        emotions={},
        session_hours=1.5,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # band: -1 (>60min); continuous: -0.75 → energy 6.25 → round 6
    assert bs.energy == 6


def test_energy_session_band_120_to_180():
    bs = compute_body_state(
        emotions={},
        session_hours=2.5,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # band: -2 (>120min); continuous: -1.25 → energy 4.75 → round 5
    assert bs.energy == 5


def test_energy_session_band_over_180():
    bs = compute_body_state(
        emotions={},
        session_hours=3.5,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # band: -3 (>180min); continuous: -1.75 → energy 3.25 → round 3
    assert bs.energy == 3


def test_energy_words_drain():
    bs = compute_body_state(
        emotions={},
        session_hours=0.5,
        words_written=2500,
        days_since_contact=0.0,
        now=_now(),
    )
    # band: 0 (not >60min); continuous: -0.25; words: -1.0 → energy 6.75 → round 7
    assert bs.energy == 7


def test_energy_high_emotional_load_drain():
    emotions = {f"emo{i}": 8.0 for i in range(7)}  # 7 high emotions
    bs = compute_body_state(
        emotions=emotions,
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # 8 baseline - 1 (>6 high emotions) = 7
    assert bs.energy == 7


def test_energy_rest_need_drain():
    bs = compute_body_state(
        emotions={"rest_need": 8.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # 8 - 1 (rest_need >= 7) = 7
    assert bs.energy == 7


def test_energy_peace_restoration_fresh_session():
    bs = compute_body_state(
        emotions={"peace": 8.0},
        session_hours=0.5,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # 8 baseline - 0.25 (continuous) + 1 (peace, fresh) = 8.75 → round 9
    assert bs.energy == 9


def test_energy_peace_no_restoration_old_session():
    bs = compute_body_state(
        emotions={"peace": 8.0},
        session_hours=2.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # peace bonus blocked (session_hours >= 1); band: 0 (120min not >120min);
    # continuous: -1.0; actual energy 6
    assert bs.energy == 6


def test_energy_clamped_at_1():
    bs = compute_body_state(
        emotions={"rest_need": 8.0},
        session_hours=10.0,
        words_written=20000,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.energy == 1


def test_energy_clamped_at_10():
    """No path through current formula reaches 10, but clamp must hold
    against a future tweak that adds another bonus term."""
    # Pretend a future term added +5; today we just hand-check the upper clamp:
    bs = compute_body_state(
        emotions={"peace": 8.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # 8 + 1 (peace, fresh) = 9 → still under 10
    assert bs.energy == 9
    assert bs.energy <= 10


def test_temperature_baseline():
    bs = compute_body_state(
        emotions={},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.temperature == 4


def test_temperature_arousal_warm():
    bs = compute_body_state(
        emotions={"arousal": 8.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.temperature == 5  # 4 + 1


def test_temperature_full_warmth_stack():
    bs = compute_body_state(
        emotions={
            "arousal": 8.0,
            "desire": 8.0,
            "belonging": 9.0,
            "love": 9.0,
            "climax": 6.0,
        },
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # 4 + 1 + 1 + 1 + 1 + 1 = 9 (max)
    assert bs.temperature == 9


def test_temperature_body_grief_cold():
    bs = compute_body_state(
        emotions={"body_grief": 8.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.temperature == 3  # 4 - 1


def test_temperature_long_no_contact():
    bs = compute_body_state(
        emotions={},
        session_hours=0.0,
        words_written=0,
        days_since_contact=8.0,
        now=_now(),
    )
    assert bs.temperature == 2  # 4 - 2 (>7 days)


def test_temperature_medium_no_contact():
    bs = compute_body_state(
        emotions={},
        session_hours=0.0,
        words_written=0,
        days_since_contact=4.0,
        now=_now(),
    )
    assert bs.temperature == 3  # 4 - 1 (>3 days)


def test_temperature_clamped_at_1():
    bs = compute_body_state(
        emotions={"body_grief": 8.0, "touch_hunger": 8.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=10.0,
        now=_now(),
    )
    # 4 - 1 - 1 - 2 = 0 → clamp 1
    assert bs.temperature == 1


def test_temperature_asymmetric_range_top_is_9():
    """Spec: temperature 1-9, NOT 1-10."""
    bs = compute_body_state(
        emotions={
            "arousal": 8.0,
            "desire": 8.0,
            "belonging": 9.0,
            "love": 9.0,
            "climax": 6.0,
            "extra1": 0.0,
        },
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    assert bs.temperature <= 9


def test_exhaustion_derivation():
    bs = compute_body_state(
        emotions={},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    # energy 8 → exhaustion max(0, 7-8) = 0
    assert bs.exhaustion == 0


def test_exhaustion_high_when_energy_low():
    bs = compute_body_state(
        emotions={"rest_need": 8.0},
        session_hours=10.0,
        words_written=20000,
        days_since_contact=0.0,
        now=_now(),
    )
    # energy clamped at 1 → exhaustion 6
    assert bs.energy == 1
    assert bs.exhaustion == 6


def test_body_emotions_dict_includes_all_six_with_zero_default():
    bs = compute_body_state(
        emotions={"arousal": 5.0},
        session_hours=0.0,
        words_written=0,
        days_since_contact=0.0,
        now=_now(),
    )
    expected = {"arousal", "desire", "climax", "touch_hunger", "comfort_seeking", "rest_need"}
    assert set(bs.body_emotions.keys()) == expected
    assert bs.body_emotions["arousal"] == 5.0
    assert bs.body_emotions["desire"] == 0.0
    assert bs.body_emotions["climax"] == 0.0


def test_compute_body_state_under_5ms_p99():
    """Inviolate property #4 from spec §7.1 — body block must not
    block chat composition. p99 over 100 random inputs < 5ms.
    """
    import random
    import time

    rng = random.Random(42)
    timings: list[float] = []
    for _ in range(100):
        emotions = {
            name: rng.uniform(0.0, 10.0)
            for name in (
                "love",
                "joy",
                "grief",
                "arousal",
                "desire",
                "climax",
                "rest_need",
                "touch_hunger",
                "peace",
                "belonging",
            )
        }
        session_hours = rng.uniform(0.0, 5.0)
        words = rng.randint(0, 10000)
        days = rng.uniform(0.0, 30.0)
        start = time.perf_counter()
        compute_body_state(
            emotions=emotions,
            session_hours=session_hours,
            words_written=words,
            days_since_contact=days,
            now=_now(),
        )
        timings.append(time.perf_counter() - start)
    timings.sort()
    p99 = timings[98]  # 99th percentile of 100 samples
    assert p99 < 0.005, f"p99 {p99 * 1000:.2f}ms exceeds 5ms budget"
