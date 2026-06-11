"""v0.0.33 Track 3: peak emotion blend in forgetting salience.

Property tests pin the spec's calibration CONSTRAINTS, not the constants —
retuning _PEAK_LAMBDA / the linger horizon later must not churn these as
long as the constraints hold:
  - peak >= 5.0, unrecalled, no other inputs: above FADE (0.25) for >= 45 days
  - peak <= 1.0: below FADE within 14 days (trivia still fades)
  - LOSE (0.10) reachable for peak-5 within ~6 months (forgetting stays real)
"""
from datetime import UTC, datetime, timedelta

from brain.felt_time.state import FeltTimeState
from brain.forgetting.policy import FADE_THRESHOLD
from brain.forgetting.salience import DEFAULT_WEIGHTS, _emotion_input, _peak_input
from brain.memory.store import Memory


def _mem(peak: float, days_old: float, emotions: dict | None = None) -> Memory:
    created = datetime.now(UTC) - timedelta(days=days_old)
    return Memory(
        id="m" * 36, content="x", memory_type="conversation", domain="us",
        created_at=created, emotions=emotions or {},
        last_accessed_at=created, peak_emotion_intensity=peak,
    )


def _fts(rate_one_hours: float = 24 * 200) -> FeltTimeState:
    """FeltTimeState whose lived/wall rate ≈ 1 (lived hours == wall hours)."""
    anchor = (datetime.now(UTC) - timedelta(hours=rate_one_hours)).isoformat()
    return FeltTimeState(lived_age_hours=rate_one_hours, last_tick_ts=anchor)


def _emotion_term(mem: Memory) -> float:
    blended = max(_emotion_input(mem), _peak_input(mem, _fts()))
    return DEFAULT_WEIGHTS["emotion"] * blended


def test_peak5_above_fade_at_45_days():
    assert _emotion_term(_mem(peak=5.0, days_old=45)) >= FADE_THRESHOLD


def test_peak1_below_fade_by_14_days():
    assert _emotion_term(_mem(peak=1.0, days_old=14)) < FADE_THRESHOLD


def test_peak5_reaches_lose_within_six_months():
    from brain.forgetting.policy import LOST_THRESHOLD
    assert _emotion_term(_mem(peak=5.0, days_old=180)) < LOST_THRESHOLD


def test_blend_takes_living_feeling_when_hotter():
    """A still-hot current emotion dominates a modest peak — max, not sum."""
    mem = _mem(peak=2.0, days_old=0, emotions={"joy": 9.0})
    assert max(_emotion_input(mem), _peak_input(mem, _fts())) == _emotion_input(mem)


def test_peak_zero_contributes_nothing():
    assert _peak_input(_mem(peak=0.0, days_old=1), _fts()) == 0.0


def test_peak_survives_empty_emotions():
    """The Phoebe failure shape: emotions dict noise-floor-emptied, peak intact."""
    mem = _mem(peak=5.0, days_old=7, emotions={})
    assert _emotion_input(mem) == 0.0
    assert _peak_input(mem, _fts()) > 0.0
