"""First-cross detection for learned-pattern crystallisation.

A pattern crystallises the first time it reaches `known` maturity.
That crossing emits a feed event. Subsequent confirmations are silent.
Stored state: LearnedPattern.crystallised_at (None until first cross).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from brain.attunement.schemas import SCHEMA_VERSION, LearnedPattern
from brain.attunement.store import _append_pattern, read_learned_patterns


@dataclass(frozen=True)
class CrystallisationEvent:
    pattern_id: str
    category: str
    description: str
    ts: str


def check_crystallisations(
    persona_dir: Path, *, now_iso: str
) -> list[CrystallisationEvent]:
    """Find patterns that crossed `known` for the first time; mark + emit events."""
    events: list[CrystallisationEvent] = []
    for prev in read_learned_patterns(persona_dir):
        if prev.maturity == "known" and prev.crystallised_at is None:
            updated = LearnedPattern(
                id=prev.id,
                category=prev.category,
                canonical_key=prev.canonical_key,
                description=prev.description,
                evidence_count=prev.evidence_count,
                maturity=prev.maturity,
                first_seen_at=prev.first_seen_at,
                last_confirmed_at=prev.last_confirmed_at,
                last_addressed_at=prev.last_addressed_at,
                crystallised_at=now_iso,
                falsified_at=prev.falsified_at,
                examples=prev.examples,
                schema_version=SCHEMA_VERSION,
            )
            _append_pattern(persona_dir, updated)
            events.append(CrystallisationEvent(
                pattern_id=prev.id,
                category=prev.category,
                description=prev.description,
                ts=now_iso,
            ))
    return events
