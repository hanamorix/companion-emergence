"""Persistence + grounding gate for the attunement subsystem."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from brain.attunement.schemas import CurrentRead, PatternCandidate

log = logging.getLogger(__name__)

_ATTUNEMENT_DIR = "attunement"
_CURRENT_READ_FILE = "current_read.json"


def _attunement_dir(persona_dir: Path) -> Path:
    return persona_dir / _ATTUNEMENT_DIR


def _current_read_path(persona_dir: Path) -> Path:
    return _attunement_dir(persona_dir) / _CURRENT_READ_FILE


def write_current_read(persona_dir: Path, read: CurrentRead) -> None:
    """Overwrite the current-read snapshot file. Atomic via .tmp + rename."""
    target = _current_read_path(persona_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(read), indent=2, sort_keys=True))
    tmp.replace(target)


def read_current_read(persona_dir: Path) -> CurrentRead | None:
    """Return the latest current-read snapshot, or None if missing/corrupt."""
    target = _current_read_path(persona_dir)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text())
        return CurrentRead(**payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("attunement: corrupt current_read.json — treating as missing: %s", exc)
        return None

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
