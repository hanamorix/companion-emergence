"""Tests for brain.attunement.schemas."""
from __future__ import annotations

import pytest

from brain.attunement.schemas import (
    ADDRESS_COOLDOWN_HOURS,
    DAILY_BUDGET_DEFAULT,
    MATURITY_FALSIFIED_MAX,
    MATURITY_FORMING_MIN,
    MATURITY_KNOWN_MIN,
    SCHEMA_VERSION,
    BackfillState,
    CurrentRead,
    DetectorOutput,
    LearnedPattern,
    PatternCandidate,
    pattern_id,
)


def test_schema_version_matches_alpha_1() -> None:
    assert SCHEMA_VERSION == "0.0.28-alpha.1"


def test_maturity_thresholds_have_expected_values() -> None:
    assert MATURITY_FORMING_MIN == 3
    assert MATURITY_KNOWN_MIN == 10
    assert MATURITY_FALSIFIED_MAX == 3


def test_cooldown_window_is_six_hours() -> None:
    assert ADDRESS_COOLDOWN_HOURS == 6.0


def test_daily_budget_default_is_one_fifty() -> None:
    assert DAILY_BUDGET_DEFAULT == 150


def test_current_read_constructs_with_required_fields() -> None:
    read = CurrentRead(
        ts="2026-05-31T12:00:00Z",
        source_turn_id="turn-001",
        tone_label="warm",
        tone_justification="soft phrasing throughout",
        cadence_label="measured",
        cadence_justification="full sentences, no fragments",
        mood_valence=0.4,
        mood_intensity=0.5,
        predicted_arc_shape="settling in for a long conversation",
        schema_version=SCHEMA_VERSION,
    )
    assert read.tone_label == "warm"
    assert read.schema_version == SCHEMA_VERSION


def test_learned_pattern_constructs_with_all_fields() -> None:
    pattern = LearnedPattern(
        id="hash-abc",
        category="tone",
        canonical_key="tone:warm-when-dog",
        description="Softens whenever we talk about the dog",
        evidence_count=5,
        maturity="forming",
        first_seen_at="2026-04-01T00:00:00Z",
        last_confirmed_at="2026-05-31T12:00:00Z",
        last_addressed_at=None,
        crystallised_at=None,
        falsified_at=None,
        examples=["example 1", "example 2"],
        schema_version=SCHEMA_VERSION,
    )
    assert pattern.maturity == "forming"


def test_pattern_candidate_requires_evidence_quote_and_turn_id() -> None:
    candidate = PatternCandidate(
        category="tone",
        canonical_key="tone:warm-when-dog",
        description="Softens when discussing the dog",
        evidence_quote="The dog rolled over today and I cried a little",
        evidence_turn_id="turn-042",
    )
    assert candidate.evidence_quote
    assert candidate.evidence_turn_id


def test_detector_output_carries_current_read_and_candidates() -> None:
    read = CurrentRead(
        ts="2026-05-31T12:00:00Z",
        source_turn_id="turn-001",
        tone_label="warm",
        tone_justification="x",
        cadence_label="measured",
        cadence_justification="y",
        mood_valence=0.0,
        mood_intensity=0.0,
        predicted_arc_shape="z",
        schema_version=SCHEMA_VERSION,
    )
    output = DetectorOutput(
        current_read=read,
        pattern_candidates=[],
        addressed_pattern_ids=[],
        rejection_notes=[],
    )
    assert output.current_read is read
    assert output.pattern_candidates == []


def test_backfill_state_constructs() -> None:
    state = BackfillState(
        started_at="2026-05-31T12:00:00Z",
        total_windows=100,
        sampled_windows=20,
        processed_windows=0,
        patterns_emitted=0,
        status="running",
        last_cursor="window-000",
        schema_version=SCHEMA_VERSION,
    )
    assert state.status == "running"


def test_pattern_id_is_stable_hash_of_category_and_key() -> None:
    id1 = pattern_id("tone", "warm-when-dog")
    id2 = pattern_id("tone", "warm-when-dog")
    id3 = pattern_id("tone", "different-key")
    assert id1 == id2
    assert id1 != id3


def test_invalid_category_raises() -> None:
    with pytest.raises(ValueError, match="invalid category"):
        PatternCandidate(
            category="not-a-real-category",
            canonical_key="x",
            description="y",
            evidence_quote="z",
            evidence_turn_id="t",
        )
