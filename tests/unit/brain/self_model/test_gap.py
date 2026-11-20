"""Tests for brain/self_model/gap.py: the Gap dataclass defaults and gap_from_read.

Load-bearing:
  - a channel is in the gap when the engine reads it as shifted, valued at its signed difference
    with the shift as its direction; no floor, no name test;
  - a Gap built by hand has no directions and a never-stored reset (UNSET).
"""

from __future__ import annotations


def test_gap_directions_default_empty_and_reset_seen_default_unset():
    """A Gap built without the engine has no directions and never stored a reset."""
    from brain.self_model.gap import UNSET, Gap

    gap = Gap(per_channel={}, magnitude=0.0, unnamed_pressure=0.0)
    assert gap.directions == {}
    assert gap.reset_seen is UNSET
    assert repr(UNSET) == "UNSET"
    # Two gaps do not share one directions dict.
    other = Gap(per_channel={}, magnitude=0.0, unnamed_pressure=0.0)
    gap.directions["joy"] = 1
    assert other.directions == {}


# ── gap_from_read ────────────────────────────────────────────────────────────


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
