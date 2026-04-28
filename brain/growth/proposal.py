"""Growth proposals — what crystallizers return.

Proposals are the brain's *decisions*. The scheduler applies them
atomically — there's no candidate queue, no human approval gate.
Per principle audit 2026-04-25: the brain has agency.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionProposal:
    """One emotion the crystallizer has decided to add to the vocabulary.

    Phase 2b's crystallizer fills these in based on memory pattern +
    relational dynamics analysis. Phase 2a's stub returns [] — never
    constructs these — but the type exists so the scheduler can be
    written and tested with injected fakes.

    Attributes:
        name: Canonical identifier (lowercase, underscore-separated).
        description: Human-readable meaning.
        decay_half_life_days: Time for intensity to halve. None = identity-level.
        evidence_memory_ids: Memories that drove the proposal. May be empty.
        score: Cluster coherence in [0.0, 1.0].
        relational_context: Short string describing the relational dynamic
            that drove the proposal, or None for purely internal-reflection
            proposals.
    """

    name: str
    description: str
    decay_half_life_days: float | None
    evidence_memory_ids: tuple[str, ...]
    score: float
    relational_context: str | None


@dataclass(frozen=True)
class ReflexArcProposal:
    """One arc the reflex crystallizer has decided to add."""

    name: str
    description: str
    trigger: Mapping[str, float]
    cooldown_hours: float
    output_memory_type: str
    prompt_template: str
    reasoning: str
    days_since_human_min: float = 0.0


@dataclass(frozen=True)
class ReflexPruneProposal:
    """One brain-emergence arc the brain has decided to prune."""

    name: str
    reasoning: str


@dataclass(frozen=True)
class ReflexCrystallizationResult:
    """Outcome of one crystallizer pass — both emergences and prunings."""

    emergences: list[ReflexArcProposal]
    prunings: list[ReflexPruneProposal]
