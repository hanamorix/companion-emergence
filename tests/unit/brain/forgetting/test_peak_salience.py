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
from brain.memory.store import Memory, MemoryStore


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


def test_peak_flows_through_live_forgetting_pass(tmp_path):
    """Organ DoD: the producer (store choke point) and consumer (salience)
    must connect THROUGH the real forgetting pass — not only in unit isolation.
    Two week-old memories, emotions emptied (the post-noise-floor shape); the
    peak-5 one survives the pass active, the peak-0 one fades.

    Requires a minimal felt_time_state.json with lived_age_hours > 0 so
    policy.is_exempt()'s cold-start guard (lived_age_hours <= 0 → all exempt)
    does not short-circuit the pass.  A rate of 1 lived-hour per wall-hour
    keeps the test deterministic without importing felt-time machinery.
    """
    import json as _json

    from brain.forgetting import run_pass

    class _Bus:
        def publish(self, event):  # noqa: ANN001
            pass

    persona_dir = tmp_path
    # Minimal felt-time state: lived rate ≈ 1 (lived hours == wall hours).
    # lived_age_hours=4800 (200 days) puts memories squarely outside the
    # RECENT_LIVED_HOURS exemption window.  last_tick_ts is set 4800 wall
    # hours before now so the lived/wall rate ≈ 1 — no distortion to linger.
    last_tick_iso = (datetime.now(UTC) - timedelta(hours=4800)).isoformat()
    (persona_dir / "felt_time_state.json").write_text(
        _json.dumps({
            "lived_age_hours": 4800.0,
            "last_tick_ts": last_tick_iso,
            "anchors": {},
            "pressure": {},
            "horizon_pressure": {},
            "arc_anchors": [],
            "weather_baselines": {},
            "replayed": False,
        })
    )

    store = MemoryStore(persona_dir / "memories.db")
    kept = Memory.create_new("loopy the dog", "conversation", "us",
                             emotions={"joy": 5.0})
    faded = Memory.create_new("weather small-talk", "conversation", "us")
    week_ago = datetime.now(UTC) - timedelta(days=7)
    for m in (kept, faded):
        m.created_at = week_ago
        m.last_accessed_at = week_ago
        store.create(m)
    store.update(kept.id, emotions={})  # noise-floor shape; peak stays 5.0
    store.close()  # run_pass opens its own connection

    summary = run_pass(persona_dir, event_bus=_Bus())
    assert summary["total"] == 2

    store = MemoryStore(persona_dir / "memories.db")
    assert store.get(kept.id).state == "active"
    assert store.get(faded.id).state == "fading"
