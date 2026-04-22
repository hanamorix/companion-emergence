"""Tests for brain.emotion.blend — emergent co-occurrence detection."""

from __future__ import annotations

from brain.emotion.blend import BlendDetector
from brain.emotion.state import EmotionalState


def _with(**intensities: float) -> EmotionalState:
    """Helper: build an EmotionalState with the given emotion:intensity pairs."""
    state = EmotionalState()
    for name, value in intensities.items():
        state.set(name, value)
    return state


def test_detector_starts_empty() -> None:
    """A fresh detector has no recorded blends."""
    detector = BlendDetector()
    assert detector.detected() == []


def test_single_observation_does_not_detect() -> None:
    """A single co-occurrence is not enough — threshold is ≥5."""
    detector = BlendDetector()
    detector.observe(_with(tenderness=7.0, desire=6.0))
    assert detector.detected() == []


def test_five_repeats_detects_blend() -> None:
    """Five observations of the same high-intensity pair register a blend."""
    detector = BlendDetector()
    for _ in range(5):
        detector.observe(_with(tenderness=7.0, desire=6.0))
    detected = detector.detected()
    assert len(detected) == 1
    assert detected[0].components == ("desire", "tenderness")
    assert detected[0].count == 5


def test_blend_respects_intensity_threshold() -> None:
    """Observations with low intensities don't count toward the threshold."""
    detector = BlendDetector(intensity_threshold=5.0)
    detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.observe(_with(tenderness=3.0, desire=2.0))
    assert detector.detected() == []


def test_unrelated_pairs_tracked_independently() -> None:
    """Different emotion pairs are tracked separately."""
    detector = BlendDetector()
    for _ in range(5):
        detector.observe(_with(tenderness=7.0, desire=6.0))
    for _ in range(5):
        detector.observe(_with(creative_hunger=8.0, defiance=7.0))

    detected = detector.detected()
    assert len(detected) == 2
    component_sets = {d.components for d in detected}
    assert ("desire", "tenderness") in component_sets
    assert ("creative_hunger", "defiance") in component_sets


def test_naming_assigns_curated_name() -> None:
    """A detected blend can be given a human-readable name via name_blend()."""
    detector = BlendDetector()
    for _ in range(5):
        detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.name_blend(("desire", "tenderness"), "building_love")

    detected = detector.detected()
    assert detected[0].name == "building_love"


def test_name_unknown_blend_raises() -> None:
    """Naming a blend that hasn't been detected raises KeyError."""
    detector = BlendDetector()
    try:
        detector.name_blend(("love", "grief"), "heartbreak")
    except KeyError as e:
        assert "not detected" in str(e).lower() or "love" in str(e).lower()
    else:
        raise AssertionError("Expected KeyError")


def test_three_component_blend() -> None:
    """Three emotions co-occurring at high intensity form a three-component blend."""
    detector = BlendDetector()
    for _ in range(5):
        detector.observe(_with(creative_hunger=8.0, defiance=7.0, joy=6.0))
    detected = detector.detected()
    assert len(detected) >= 1
    assert any(len(d.components) == 3 for d in detected)


def test_to_dict_round_trips() -> None:
    """Detector state serialises and restores."""
    detector = BlendDetector()
    for _ in range(5):
        detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.name_blend(("desire", "tenderness"), "building_love")

    data = detector.to_dict()
    restored = BlendDetector.from_dict(data)
    assert restored.detected()[0].name == "building_love"
    assert restored.detected()[0].count == 5
