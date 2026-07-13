"""Tests for the pass-2 extractor's conversation-born interest_candidate inlet.

Uses a real temp NELLBRAIN_HOME per the project no-mocked-state-files rule,
mirroring the persona_dir fixture pattern in test_extractor_apply.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from brain.chat.extractor import ExtractorOutput, apply_side_effects
from brain.engines._interests import Interest, InterestSet

DEFAULTS = Path(__file__).parents[4] / "brain" / "engines" / "default_interests.json"


@pytest.fixture
def persona_dir_with_interests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Persona dir with an interests.json pre-populated with 'Existing Topic'."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona = tmp_path / "personas" / "nell"
    persona.mkdir(parents=True)
    existing = Interest.from_dict(
        {
            "id": "bootstrap-existing",
            "topic": "Existing Topic",
            "pull_score": 6.5,
            "scope": "either",
            "related_keywords": [],
            "notes": "",
            "first_seen": "2026-04-01T10:00:00Z",
            "last_fed": "2026-04-01T10:00:00Z",
            "last_researched_at": None,
            "feed_count": 0,
            "source_types": ["bootstrap"],
        }
    )
    InterestSet(interests=(existing,)).save(persona / "interests.json")
    return persona


def test_interest_candidate_optional_and_parsed():
    out = ExtractorOutput.model_validate(
        {"interest_candidate": {"topic": "art restoration", "keywords": ["fresco"], "why": "she lit up"}}
    )
    assert out.interest_candidate.topic == "art restoration"
    assert ExtractorOutput.model_validate({}).interest_candidate is None


def test_apply_creates_interest_below_threshold(persona_dir_with_interests):
    out = ExtractorOutput.model_validate({"interest_candidate": {"topic": "new thing"}})
    apply_side_effects(out, persona_dir=persona_dir_with_interests)
    s = InterestSet.load(persona_dir_with_interests / "interests.json", default_path=DEFAULTS)
    created = [i for i in s.interests if i.topic == "new thing"]
    assert len(created) == 1 and created[0].origin == "conversation" and created[0].pull_score == 5.0


def test_apply_respects_daily_cap(persona_dir_with_interests):
    for n in range(4):
        out = ExtractorOutput.model_validate({"interest_candidate": {"topic": f"topic {n}"}})
        apply_side_effects(out, persona_dir=persona_dir_with_interests)
    s = InterestSet.load(persona_dir_with_interests / "interests.json", default_path=DEFAULTS)
    assert len([i for i in s.interests if i.origin == "conversation"]) == 3


def test_apply_dedupes_against_existing(persona_dir_with_interests):
    # persona dir already has interest "Existing Topic"
    out = ExtractorOutput.model_validate({"interest_candidate": {"topic": "existing topic"}})
    apply_side_effects(out, persona_dir=persona_dir_with_interests)
    s = InterestSet.load(persona_dir_with_interests / "interests.json", default_path=DEFAULTS)
    assert len([i for i in s.interests if i.topic.casefold() == "existing topic"]) == 1
