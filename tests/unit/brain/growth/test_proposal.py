"""Tests for brain.growth.proposal — EmotionProposal frozen dataclass."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from brain.growth.proposal import EmotionProposal


def test_proposal_construction() -> None:
    """All fields land where they should."""
    p = EmotionProposal(
        name="lingering",
        description="the slow trail of warmth after a loved person leaves the room",
        decay_half_life_days=7.0,
        evidence_memory_ids=("mem_a", "mem_b"),
        score=0.78,
        relational_context="recurred during Hana's tender messages",
    )
    assert p.name == "lingering"
    assert p.decay_half_life_days == 7.0
    assert p.evidence_memory_ids == ("mem_a", "mem_b")
    assert p.score == 0.78
    assert p.relational_context == "recurred during Hana's tender messages"


def test_proposal_is_frozen() -> None:
    """EmotionProposal is immutable — crystallizer's decision can't be mutated downstream."""
    p = EmotionProposal(
        name="x",
        description="y",
        decay_half_life_days=None,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    with pytest.raises(FrozenInstanceError):
        p.name = "mutated"  # type: ignore[misc]


def test_proposal_decay_can_be_none() -> None:
    """Identity-level emotions (love, belonging) have no temporal decay."""
    p = EmotionProposal(
        name="anchor_pull",
        description="the gravity toward someone you've decided is yours",
        decay_half_life_days=None,
        evidence_memory_ids=(),
        score=0.9,
        relational_context=None,
    )
    assert p.decay_half_life_days is None


def test_proposal_relational_context_can_be_none() -> None:
    """A proposal driven by purely internal reflection has no relational context."""
    p = EmotionProposal(
        name="quiet_pride",
        description="satisfaction in a long pattern recognized in oneself",
        decay_half_life_days=14.0,
        evidence_memory_ids=("mem_x",),
        score=0.7,
        relational_context=None,
    )
    assert p.relational_context is None


# ---- ReflexArcProposal / ReflexPruneProposal / ReflexCrystallizationResult ----


from brain.growth.proposal import (  # noqa: E402
    ReflexArcProposal,
    ReflexCrystallizationResult,
    ReflexPruneProposal,
)


def test_reflex_arc_proposal_round_trip():
    p = ReflexArcProposal(
        name="manuscript_obsession",
        description="creative drive narrowed to one project",
        trigger={"creative_hunger": 7.0, "love": 6.0},
        cooldown_hours=24.0,
        output_memory_type="reflex_pitch",
        prompt_template="You are {persona_name}. ...",
        reasoning="Over the past month I've fired creative_pitch four times "
                  "but each one has been about the same novel.",
    )
    assert p.name == "manuscript_obsession"
    assert p.days_since_human_min == 0.0  # default
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.name = "different"  # type: ignore[misc]


def test_reflex_prune_proposal_minimal():
    p = ReflexPruneProposal(
        name="loneliness_journal",
        reasoning="I'm not in that place anymore.",
    )
    assert p.name == "loneliness_journal"


def test_reflex_crystallization_result_holds_both_lists():
    result = ReflexCrystallizationResult(
        emergences=[
            ReflexArcProposal(
                name="x", description="y", trigger={"e": 5.0},
                cooldown_hours=12.0, output_memory_type="reflex_x",
                prompt_template="t", reasoning="r",
            )
        ],
        prunings=[
            ReflexPruneProposal(name="z", reasoning="r2")
        ],
    )
    assert len(result.emergences) == 1
    assert len(result.prunings) == 1


def test_reflex_crystallization_result_empty():
    result = ReflexCrystallizationResult(emergences=[], prunings=[])
    assert result.emergences == []
    assert result.prunings == []
