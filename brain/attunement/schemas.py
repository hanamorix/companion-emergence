"""Schemas and constants for the attunement subsystem."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

SCHEMA_VERSION = "0.0.28-alpha.1"

MATURITY_FORMING_MIN = 3
MATURITY_KNOWN_MIN = 10
MATURITY_FALSIFIED_MAX = 3

ADDRESS_COOLDOWN_HOURS = 6.0
DAILY_BUDGET_DEFAULT = 150

# All five categories shipped across alpha.1 → v0.0.28; the schema validates all of them now
# so future tasks don't have to expand the enum repeatedly. alpha.1 only emits tone + cadence.
_VALID_CATEGORIES = frozenset({
    "tone",
    "cadence",
    "topic_affinity",
    "response_shape",
    "relational",
})

_VALID_MATURITIES = frozenset({"immature", "forming", "known", "falsified"})


@dataclass(frozen=True)
class CurrentRead:
    ts: str
    source_turn_id: str
    tone_label: str
    tone_justification: str
    cadence_label: str
    cadence_justification: str
    mood_valence: float
    mood_intensity: float
    predicted_arc_shape: str
    schema_version: str


@dataclass(frozen=True)
class LearnedPattern:
    id: str
    category: str
    canonical_key: str
    description: str
    evidence_count: int
    maturity: str
    first_seen_at: str
    last_confirmed_at: str
    last_addressed_at: str | None
    crystallised_at: str | None
    falsified_at: str | None
    examples: list[str]
    schema_version: str

    def __post_init__(self) -> None:
        if self.category not in _VALID_CATEGORIES:
            raise ValueError(f"invalid category: {self.category}")
        if self.maturity not in _VALID_MATURITIES:
            raise ValueError(f"invalid maturity: {self.maturity}")


@dataclass(frozen=True)
class Evidence:
    quote: str        # VERBATIM substring of the named turn
    turn_id: str

    def __post_init__(self) -> None:
        if not self.quote:
            raise ValueError("evidence quote is required")
        if not self.turn_id:
            raise ValueError("evidence turn_id is required")


@dataclass(frozen=True)
class PatternCandidate:
    category: str
    canonical_key: str
    description: str
    evidence: list[Evidence]

    def __post_init__(self) -> None:
        if self.category not in _VALID_CATEGORIES:
            raise ValueError(f"invalid category: {self.category}")
        if not self.evidence:
            raise ValueError("at least one evidence entry is required")
        if self.category == "relational" and len(self.evidence) < 2:
            raise ValueError("relational patterns require >=2 evidence entries")


@dataclass(frozen=True)
class DetectorOutput:
    current_read: CurrentRead
    pattern_candidates: list[PatternCandidate]
    addressed_pattern_ids: list[str] = field(default_factory=list)
    rejection_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BackfillState:
    started_at: str
    total_windows: int
    sampled_windows: int
    processed_windows: int
    patterns_emitted: int
    status: str
    last_cursor: str
    schema_version: str


def pattern_id(category: str, canonical_key: str) -> str:
    """Stable hash of category + canonical_key for use as LearnedPattern.id."""
    payload = f"{category}::{canonical_key}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]
