"""Tests for brain/self_model/gap.py — compute_gap(short, long).

Load-bearing:
  - a genuine short-vs-long divergence (|delta| >= noise floor) → non-zero gap,
    bidirectional (short − long);
  - identical short/long, or a sub-floor divergence → magnitude exactly 0.0;
  - unregistered channels dropped; unnamed_pressure carried through from short.
"""

from __future__ import annotations

from brain.self_model.derived import DerivedRead
from brain.self_model.gap import _GAP_NOISE_FLOOR, compute_gap


def test_divergent_short_vs_long_nonzero_gap():
    """Recent grief above baseline (short 5.0 vs long 0) → golden cross, magnitude > 0."""
    short = DerivedRead(channels={"grief": 5.0, "joy": 1.0}, unnamed_pressure=0.0, sources={})
    long = DerivedRead(channels={"joy": 1.0}, unnamed_pressure=0.0, sources={})  # no grief baseline
    gap = compute_gap(short, long)
    assert gap.magnitude > 0
    assert gap.per_channel.get("grief", 0) > 0  # short sees grief long doesn't (golden)
    assert "joy" not in gap.per_channel  # joy identical → below floor → dropped


def test_identical_signals_zero_gap():
    """Identical short and long → magnitude exactly 0.0 (registered channel, equal deltas)."""
    short = DerivedRead(channels={"grief": 4.0}, unnamed_pressure=0.0, sources={})
    long = DerivedRead(channels={"grief": 4.0}, unnamed_pressure=0.0, sources={})
    gap = compute_gap(short, long)
    assert gap.magnitude == 0.0


def test_gap_per_channel_is_short_minus_long_registered_only():
    """Unregistered channels dropped; registered delta = short − long (bidirectional)."""
    short = DerivedRead(channels={"joy": 1.0, "zorblefright": 9.0}, unnamed_pressure=0.0, sources={})
    long = DerivedRead(channels={"joy": 3.0}, unnamed_pressure=0.0, sources={})
    gap = compute_gap(short, long)
    assert "zorblefright" not in gap.per_channel  # registered channels only
    assert gap.per_channel.get("joy") == -2.0  # death cross, short below long


def test_noise_floor_drops_sub_threshold_channels_keeps_genuine():
    """A |delta| < floor channel is dropped; a >= floor channel survives at full delta."""
    small = _GAP_NOISE_FLOOR - 0.1
    big = _GAP_NOISE_FLOOR + 0.5
    short = DerivedRead(channels={"joy": 5.0 + small, "grief": 5.0 + big}, unnamed_pressure=0.0, sources={})
    long = DerivedRead(channels={"joy": 5.0, "grief": 5.0}, unnamed_pressure=0.0, sources={})
    gap = compute_gap(short, long)
    assert "joy" not in gap.per_channel  # sub-floor → dropped
    assert abs(gap.per_channel.get("grief") - big) < 1e-9  # genuine cross survives, full delta
    assert abs(gap.magnitude - big) < 1e-9


def test_gap_carries_unnamed_pressure_through_from_short():
    """unnamed_pressure from the short (felt) read passes through to Gap unchanged."""
    gap = compute_gap(DerivedRead({}, 0.4, {}), DerivedRead({}, 0.0, {}))
    assert gap.unnamed_pressure == 0.4


def test_noise_floor_equals_ambient_channel_floor_invariant():
    """The gap noise floor is intentionally equal to the ambient block's naming
    floor (surface only what is loud enough to name). The two are declared as
    separate local constants to avoid an import cycle; lock the documented
    equality so a future edit to one cannot silently drift from the other."""
    from brain.self_model.ambient import _CHANNEL_FLOOR

    assert _GAP_NOISE_FLOOR == _CHANNEL_FLOOR
