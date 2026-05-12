"""Unit tests for the D-reflection module."""
from __future__ import annotations

import pytest

from brain.initiate.reflection import (
    DDecision,
    DReflectionResult,
    parse_structured_response,
)


def test_d_decision_dataclass_shape():
    d = DDecision(
        candidate_index=1,
        decision="promote",
        reason="genuinely surprising memory return",
        confidence="high",
    )
    assert d.candidate_index == 1
    assert d.decision == "promote"
    assert d.confidence == "high"


def test_parse_structured_response_happy_path():
    raw = """
    {
      "decisions": [
        {"candidate_index": 1, "decision": "promote",
         "reason": "worth saying", "confidence": "high"},
        {"candidate_index": 2, "decision": "filter",
         "reason": "private weather", "confidence": "medium"}
      ],
      "tick_note": "one worth surfacing today"
    }
    """
    result = parse_structured_response(raw)
    assert isinstance(result, DReflectionResult)
    assert len(result.decisions) == 2
    assert result.decisions[0].decision == "promote"
    assert result.decisions[1].decision == "filter"
    assert result.tick_note == "one worth surfacing today"


def test_parse_structured_response_extracts_from_text_with_prose():
    """Models sometimes wrap JSON in prose. Parser should still find it."""
    raw = 'Here is my decision:\n```json\n{"decisions": [], "tick_note": null}\n```\nThanks.'
    result = parse_structured_response(raw)
    assert result.decisions == []
    assert result.tick_note is None


def test_parse_structured_response_raises_on_malformed():
    with pytest.raises(ValueError):
        parse_structured_response("not even close to json")
