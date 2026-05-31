"""Persistence + grounding gate for the attunement subsystem."""
from __future__ import annotations

import re
import unicodedata
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
    """Normalise text for substring matching.

    Applies NFC (canonical composition) so precomposed and decomposed
    Unicode forms of the same string match. Uses casefold() rather than
    .lower() for correctness on a few edge cases (German ß).

    NFKD is intentionally NOT used — it folds compatibility forms
    (ligatures, superscripts, full-width digits) which would enable
    false-positive matches against text the user did not write. Dashes
    (-/–/—) and smart-quote variants are deliberately left distinct;
    folding them is out of scope for the spec's grounding gate.
    """
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def validate_grounded(
    candidate: PatternCandidate, buffer_slice: list[BufferTurn]
) -> bool:
    """Reject candidates whose evidence_quote can't be located in the named turn.

    Hard gate: candidate is dropped when False is returned. Load-bearing
    hallucination control per spec §13 Risk 1.

    Normalisation: whitespace collapsed, NFC Unicode normalisation, and
    casefold applied before substring match.
    """
    turn = next((t for t in buffer_slice if t.id == candidate.evidence_turn_id), None)
    if turn is None:
        return False
    return _normalise(candidate.evidence_quote) in _normalise(turn.content)
