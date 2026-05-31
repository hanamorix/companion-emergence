"""Persistence + grounding gate for the attunement subsystem."""
from __future__ import annotations

import re
from dataclasses import dataclass

from brain.attunement.schemas import PatternCandidate

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class BufferTurn:
    """Minimal buffer-turn view consumed by the grounding validator.

    Constructed from `active_conversations/*.jsonl` rows or buffer slices.
    """

    id: str
    content: str


def _normalise(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def validate_grounded(
    candidate: PatternCandidate, buffer_slice: list[BufferTurn]
) -> bool:
    """Reject candidates whose evidence_quote can't be located in the named turn.

    Hard gate: candidate is dropped when False is returned. Load-bearing
    hallucination control per spec §13 Risk 1.

    Normalisation: whitespace collapsed + lowercased before substring match.
    """
    turn = next((t for t in buffer_slice if t.id == candidate.evidence_turn_id), None)
    if turn is None:
        return False
    return _normalise(candidate.evidence_quote) in _normalise(turn.content)
