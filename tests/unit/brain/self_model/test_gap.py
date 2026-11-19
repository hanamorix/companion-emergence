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


def test_gap_directions_default_empty_and_reset_seen_default_unset():
    """compute_gap builds no directions and never stored a reset (B2 phase B, step 20)."""
    from brain.self_model.gap import UNSET, Gap

    gap = Gap(per_channel={}, magnitude=0.0, unnamed_pressure=0.0)
    assert gap.directions == {}
    assert gap.reset_seen is UNSET
    assert repr(UNSET) == "UNSET"
    # Two gaps do not share one directions dict.
    other = Gap(per_channel={}, magnitude=0.0, unnamed_pressure=0.0)
    gap.directions["joy"] = 1
    assert other.directions == {}


# ── B2 phase B: gap_from_read ────────────────────────────────────────────────


def _reading(difference=0.0, shift=0):
    from brain.emotion.engine import Reading

    return Reading(short=0.0, long=0.0, difference=difference, shift=shift)


def test_gap_from_read_keeps_shifted_channels_with_signed_difference_and_direction():
    from brain.self_model.gap import gap_from_read

    read = {
        "joy": _reading(difference=2.5, shift=1),
        "grief": _reading(difference=-1.0, shift=-1),
        "calm": _reading(difference=3.0, shift=0),  # not shifted: left out whatever its difference
    }
    gap = gap_from_read(read, unnamed_pressure=0.4)
    assert gap.per_channel == {"joy": 2.5, "grief": -1.0}
    assert gap.directions == {"joy": 1, "grief": -1}
    assert gap.magnitude == 3.5
    assert gap.unnamed_pressure == 0.4


def test_gap_from_read_keeps_a_zero_difference_channel_that_shifted_and_has_no_floor():
    from brain.self_model.gap import gap_from_read

    gap = gap_from_read({"joy": _reading(difference=0.0, shift=1), "awe": _reading(difference=0.01, shift=-1)},
                        unnamed_pressure=0.0)
    assert gap.per_channel == {"joy": 0.0, "awe": 0.01}
    assert gap.directions == {"joy": 1, "awe": -1}


def test_gap_from_an_empty_read_is_empty():
    from brain.self_model.gap import gap_from_read

    gap = gap_from_read({}, unnamed_pressure=0.0)
    assert gap.per_channel == {} and gap.directions == {} and gap.magnitude == 0.0
