"""Tests for brain/self_model/gap.py — compute_gap (R-C1 orthogonality assertion).

Load-bearing: divergent declared/derived → non-zero gap (R-C1);
              identical → zero (R-C1 converse).
"""

from __future__ import annotations

from brain.emotion.state import EmotionalState
from brain.self_model.derived import DerivedRead
from brain.self_model.gap import compute_gap


def test_divergent_declared_vs_derived_nonzero_gap():
    """R-C1: declared peak-joy vs derived trend-grief → magnitude > 0."""
    declared = EmotionalState(emotions={"joy": 9.0})          # max-pool peak
    derived = DerivedRead(channels={"grief": 5.0, "joy": 1.0}, unnamed_pressure=0.0, sources={})
    gap = compute_gap(declared, derived)
    assert gap.magnitude > 0
    assert gap.per_channel.get("grief", 0) > 0  # derived sees grief declared doesn't


def test_identical_signals_zero_gap():
    """R-C1 converse: identical declared and derived → magnitude exactly 0.0."""
    declared = EmotionalState(emotions={"calm": 4.0})
    derived = DerivedRead(channels={"calm": 4.0}, unnamed_pressure=0.0, sources={})
    gap = compute_gap(declared, derived)
    assert gap.magnitude == 0.0


def test_gap_per_channel_is_derived_minus_declared_registered_only():
    """Unregistered channels dropped; registered delta = derived − declared."""
    declared = EmotionalState(emotions={"joy": 3.0})
    derived = DerivedRead(channels={"joy": 1.0, "zorblefright": 9.0}, unnamed_pressure=0.0, sources={})
    gap = compute_gap(declared, derived)
    assert "zorblefright" not in gap.per_channel  # registered channels only
    assert gap.per_channel.get("joy") == -2.0


def test_gap_carries_unnamed_pressure_through():
    """unnamed_pressure from the derived read passes through to Gap unchanged."""
    gap = compute_gap(EmotionalState(emotions={}), DerivedRead({}, 0.4, {}))
    assert gap.unnamed_pressure == 0.4
