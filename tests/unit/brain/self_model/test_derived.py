"""Tests for brain/self_model/derived.py — normalized recency-weighted mean cross.

  - compute_derived = SHORT-half-life normalized mean per channel + unnamed_pressure
  - compute_baseline = LONG-half-life normalized mean per channel
Contracts: recency weighting (an old event is down-weighted, not excluded, and a
recent event dominates the short read); constraint-1 (non-bearing padding is
invisible); fail-open; conservative unnamed_pressure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.memory.store import Memory
from brain.self_model.derived import DerivedRead, compute_baseline, compute_derived

NOW = datetime(2026, 8, 22, tzinfo=UTC)


def _mem(emotions: dict, age_days: float) -> Memory:
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions=emotions)
    object.__setattr__(m, "created_at", NOW - timedelta(days=age_days))
    return m


def test_derived_short_read_is_recency_weighted_mean_not_max_not_plainmean():
    """An old high grief (9.0, 10d) + a recent low grief (3.0, 0d): the SHORT read
    is a recency-weighted mean — the recent event dominates (result well below the
    old peak and below a plain mean), but the old event still contributes (result
    strictly above the recent value, i.e. down-weighted, NOT excluded).

    Fails against: a max-pool stub (→ 9.0), a plain-mean stub (→ 6.0), and a
    window-EXCLUSION stub that drops the old event (→ exactly 3.0).
    """
    mems = [_mem({"grief": 9.0}, age_days=10), _mem({"grief": 3.0}, age_days=0)]
    short = compute_derived(mems, body_energy=5, body_exhaustion=2, now=NOW).channels["grief"]
    assert 3.0 < short < 4.5, f"short={short} (recency-weighted mean expected ~3.5)"
    # long read weights the old event more → baseline above the short read (death cross)
    long = compute_baseline(mems, now=NOW).channels["grief"]
    assert long > short, f"long={long} should exceed short={short} (recent below baseline)"


def test_single_event_channel_short_equals_long_equals_value():
    """One event for a channel → short == long == its intensity (weight cancels),
    so a genuinely steady/one-off channel produces no gap."""
    mems = [_mem({"loneliness": 4.0}, age_days=1)]
    short = compute_derived(mems, body_energy=5, body_exhaustion=2, now=NOW).channels["loneliness"]
    long = compute_baseline(mems, now=NOW).channels["loneliness"]
    assert short == 4.0 and long == 4.0


def test_non_bearing_padding_is_invisible_constraint1():
    """Padding with memories bearing OTHER channels does not move a fixed channel
    (each channel is normalized by its OWN event weights)."""
    base = [_mem({"grief": 6.0}, age_days=1), _mem({"grief": 8.0}, age_days=5)]
    pad = [_mem({"fear": 5.0}, age_days=0.1 * i) for i in range(50)]
    s0 = compute_derived(base, body_energy=5, body_exhaustion=2, now=NOW).channels["grief"]
    s1 = compute_derived(base + pad, body_energy=5, body_exhaustion=2, now=NOW).channels["grief"]
    assert s0 == s1


def test_empty_memories_fails_open_to_no_gap():
    out = compute_derived([], body_energy=5, body_exhaustion=2, now=NOW)
    assert out.channels == {}
    assert out.unnamed_pressure == 0.0
    assert compute_baseline([], now=NOW).channels == {}


def test_compute_derived_never_raises_on_bad_input():
    """A memory with None emotions must not crash (fail-open)."""
    bad = _mem({}, age_days=1)
    object.__setattr__(bad, "emotions", None)
    out = compute_derived([bad], body_energy=5, body_exhaustion=2, now=NOW)
    assert isinstance(out, DerivedRead)


def test_ordinary_state_zero_unnamed_pressure():
    """Ordinary body state (energy=5, exhaustion=2) yields exactly 0.0."""
    mems = [_mem({"joy": 3.0}, age_days=1), _mem({"loneliness": 2.0}, age_days=2)]
    out = compute_derived(mems, body_energy=5, body_exhaustion=2, now=NOW)
    assert out.unnamed_pressure == 0.0


def test_strong_bodily_signal_with_no_channel_home_flags_pressure():
    """Extreme low-arousal body (energy=1, exhaustion=9) with NO low-arousal channel
    in history → residual reported as unnamed_pressure > 0."""
    mems = [_mem({"joy": 1.0}, age_days=30)]  # joy is not a low-arousal channel
    out = compute_derived(mems, body_energy=1, body_exhaustion=9, now=NOW)
    assert out.unnamed_pressure > 0.0


def test_diverse_channels_no_dilution_artifact():
    """Regression for the live magnitude-354 bug. Many memories EACH carrying a
    different channel at a constant intensity: each channel's short and long reads
    are ~equal (constant intensity) → the gap is ~0, not a large uniform offset."""
    from brain.self_model.gap import compute_gap

    chans = ["joy", "grief", "curiosity", "love"]
    mems = [_mem({chans[i % len(chans)]: 7.0}, age_days=i % 3) for i in range(40)]
    short = compute_derived(mems, body_energy=5, body_exhaustion=2, now=NOW)
    long = compute_baseline(mems, now=NOW)
    gap = compute_gap(short, long)
    assert gap.magnitude < 3.0, f"dilution artifact present: magnitude={gap.magnitude:.1f}"
