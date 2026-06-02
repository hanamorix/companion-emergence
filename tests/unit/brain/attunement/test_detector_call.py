"""Tests for the Haiku-backed attunement detector wrapper.

The Claude CLI call is mocked; we verify:
- valid JSON output parses correctly into DetectorOutput
- pattern_candidates are converted into PatternCandidate dataclasses
- malformed JSON returns a decline output (tone/cadence='unknown', empty candidates)
- invalid category in a candidate is filtered (rejection_notes records it)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from brain.attunement.detector import run_detector
from brain.attunement.schemas import SCHEMA_VERSION
from brain.attunement.store import BufferTurn


@pytest.fixture
def buffer() -> list[BufferTurn]:
    return [
        BufferTurn(id="t1", content="Hi there!"),
        BufferTurn(id="t2", content="I'm okay, just tired."),
    ]


def test_detector_returns_structured_output_on_valid_json(buffer):
    fake_response = """{
        "current_read": {
            "tone_label": "tired",
            "tone_justification": "she said tired",
            "cadence_label": "terse",
            "cadence_justification": "two short lines",
            "mood_valence": -0.2,
            "mood_intensity": 0.4,
            "predicted_arc_shape": "winding down"
        },
        "pattern_candidates": []
    }"""
    with patch("brain.attunement.detector._call_haiku", return_value=fake_response):
        output = run_detector(buffer_slice=buffer, reply_text="goodnight love")
    assert output.current_read.tone_label == "tired"
    assert output.current_read.schema_version == SCHEMA_VERSION
    assert output.pattern_candidates == []


def test_detector_handles_pattern_candidates(buffer):
    fake_response = """{
        "current_read": {
            "tone_label": "tired", "tone_justification": "x",
            "cadence_label": "terse", "cadence_justification": "y",
            "mood_valence": -0.2, "mood_intensity": 0.4,
            "predicted_arc_shape": "z"
        },
        "pattern_candidates": [
            {
                "category": "cadence",
                "canonical_key": "cadence:terse-at-night",
                "description": "Cadence shortens late at night",
                "evidence": [{"quote": "I'm okay, just tired.", "turn_id": "t2"}]
            }
        ]
    }"""
    with patch("brain.attunement.detector._call_haiku", return_value=fake_response):
        output = run_detector(buffer_slice=buffer, reply_text="goodnight")
    assert len(output.pattern_candidates) == 1
    assert output.pattern_candidates[0].canonical_key == "cadence:terse-at-night"
    assert output.pattern_candidates[0].evidence[0].quote == "I'm okay, just tired."
    assert output.pattern_candidates[0].evidence[0].turn_id == "t2"


def test_detector_returns_decline_output_on_malformed_json(buffer):
    with patch("brain.attunement.detector._call_haiku", return_value="not json {"):
        output = run_detector(buffer_slice=buffer, reply_text="anything")
    assert output.current_read.tone_label == "unknown"
    assert output.current_read.cadence_label == "unknown"
    assert output.pattern_candidates == []


def test_detector_returns_decline_output_on_invalid_category(buffer):
    """A candidate with a category not in the alpha.1 enum is dropped silently."""
    fake_response = """{
        "current_read": {
            "tone_label": "warm", "tone_justification": "x",
            "cadence_label": "measured", "cadence_justification": "y",
            "mood_valence": 0.0, "mood_intensity": 0.0,
            "predicted_arc_shape": "z"
        },
        "pattern_candidates": [
            {
                "category": "made-up-category",
                "canonical_key": "k",
                "description": "d",
                "evidence": [{"quote": "Hi there!", "turn_id": "t1"}]
            }
        ]
    }"""
    with patch("brain.attunement.detector._call_haiku", return_value=fake_response):
        output = run_detector(buffer_slice=buffer, reply_text="hi")
    assert output.pattern_candidates == []
    assert output.rejection_notes  # the invalid candidate is noted


def test_detector_parses_addressed_pattern_ids(buffer):
    """addressed_pattern_ids from the top-level payload flows through to DetectorOutput."""
    fake_response = """{
        "current_read": {
            "tone_label": "warm", "tone_justification": "x",
            "cadence_label": "measured", "cadence_justification": "y",
            "mood_valence": 0.1, "mood_intensity": 0.3,
            "predicted_arc_shape": "steady"
        },
        "pattern_candidates": [],
        "addressed_pattern_ids": ["abc", "def"]
    }"""
    with patch("brain.attunement.detector._call_haiku", return_value=fake_response):
        output = run_detector(buffer_slice=buffer, reply_text="I noticed you mentioned that")
    assert output.addressed_pattern_ids == ["abc", "def"]


def test_detector_returns_decline_on_empty_buffer():
    output = run_detector(buffer_slice=[], reply_text="anything")
    assert output.current_read.tone_label == "unknown"
    assert output.pattern_candidates == []
