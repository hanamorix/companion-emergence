"""Tests for the detector's hallucination-control filtering of pattern candidates.

Verifies that _parse_output drops candidates whose evidence list is empty or
malformed, and that valid multi-evidence candidates parse correctly.
The Claude CLI call is mocked throughout.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from brain.attunement.detector import run_detector
from brain.attunement.schemas import Evidence, PatternCandidate
from brain.attunement.store import BufferTurn


@pytest.fixture
def buffer() -> list[BufferTurn]:
    return [
        BufferTurn(id="t1", content="work has been brutal this week"),
        BufferTurn(id="t2", content="anyway I keep thinking about it"),
    ]


def test_pattern_candidate_construction_uses_evidence_list():
    """Direct unit: PatternCandidate accepts evidence=[Evidence(...)] form."""
    cand = PatternCandidate(
        category="tone",
        canonical_key="tone:tired",
        description="tired tone late at night",
        evidence=[Evidence(quote="I'm okay, just tired.", turn_id="t2")],
    )
    assert len(cand.evidence) == 1
    assert cand.evidence[0].quote == "I'm okay, just tired."
    assert cand.evidence[0].turn_id == "t2"


def _current_read_block(tone: str = "heavy", cadence: str = "slow") -> str:
    return (
        f'"current_read": {{"tone_label": "{tone}", "tone_justification": "x",'
        f'"cadence_label": "{cadence}", "cadence_justification": "y",'
        '"mood_valence": -0.3, "mood_intensity": 0.6, "predicted_arc_shape": "descending"}'
    )


def test_candidate_with_empty_evidence_list_is_dropped(buffer):
    """An evidence list of [] fails PatternCandidate.__post_init__ → dropped into rejection_notes."""
    fake_response = (
        "{"
        + _current_read_block()
        + ', "pattern_candidates": [{'
        '"category": "tone", "canonical_key": "tone:heavy", '
        '"description": "heavy tone", "evidence": []}]}'
    )
    with patch("brain.attunement.detector._call_haiku", return_value=fake_response):
        output = run_detector(buffer_slice=buffer, reply_text="yeah")
    assert output.pattern_candidates == []
    assert output.rejection_notes


def test_candidate_with_multiple_evidence_entries_parses_all(buffer):
    """A relational candidate with two evidence entries produces both Evidence objects."""
    fake_response = (
        "{"
        + _current_read_block()
        + ', "pattern_candidates": [{'
        '"category": "relational", "canonical_key": "relational:work-rumination", '
        '"description": "returns to work stress", "evidence": ['
        '{"quote": "work has been brutal this week", "turn_id": "t1"},'
        '{"quote": "keep thinking about it", "turn_id": "t2"}'
        "]}]}"
    )
    with patch("brain.attunement.detector._call_haiku", return_value=fake_response):
        output = run_detector(buffer_slice=buffer, reply_text="I hear you")
    assert len(output.pattern_candidates) == 1
    cand = output.pattern_candidates[0]
    assert len(cand.evidence) == 2
    assert cand.evidence[0].turn_id == "t1"
    assert cand.evidence[1].turn_id == "t2"
