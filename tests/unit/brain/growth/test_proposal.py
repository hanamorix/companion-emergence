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
