"""Persistence + grounding gate for the attunement subsystem."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from brain.attunement.schemas import (
    MATURITY_FALSIFIED_MAX,
    MATURITY_FORMING_MIN,
    MATURITY_KNOWN_MIN,
    SCHEMA_VERSION,
    CurrentRead,
    LearnedPattern,
    PatternCandidate,
    pattern_id,
)
from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

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


_LEARNED_PATTERNS_FILE = "learned_patterns.jsonl"
_REJECTIONS_FILE = "attunement_rejections.jsonl"
_EXAMPLES_CAP = 5


def _learned_patterns_path(persona_dir: Path) -> Path:
    return _attunement_dir(persona_dir) / _LEARNED_PATTERNS_FILE


def _rejections_path(persona_dir: Path) -> Path:
    return persona_dir / _REJECTIONS_FILE


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_rejection(persona_dir: Path, candidate: PatternCandidate, reason: str) -> None:
    path = _rejections_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": _now_iso(),
        "category": candidate.category,
        "canonical_key": candidate.canonical_key,
        "evidence": [{"quote": ev.quote, "turn_id": ev.turn_id} for ev in candidate.evidence],
        "reason": reason,
    }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_learned_patterns(persona_dir: Path) -> list[LearnedPattern]:
    """Return all learned patterns. Last entry per id wins (append-only file)."""
    path = _learned_patterns_path(persona_dir)
    if not path.exists():
        return []
    by_id: dict[str, LearnedPattern] = {}
    for row in read_jsonl_skipping_corrupt(path):
        try:
            pattern = LearnedPattern(**row)
        except (TypeError, ValueError):
            continue
        by_id[pattern.id] = pattern
    return list(by_id.values())


def _append_pattern(persona_dir: Path, pattern: LearnedPattern) -> None:
    path = _learned_patterns_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(pattern), sort_keys=True) + "\n")


def _maturity_for_count(count: int) -> str:
    if count >= MATURITY_KNOWN_MIN:
        return "known"
    if count >= MATURITY_FORMING_MIN:
        return "forming"
    return "immature"


def merge_into_learned(
    persona_dir: Path,
    candidates: list[PatternCandidate],
    buffer_slice: list[BufferTurn],
    *,
    now_iso: str | None = None,
) -> None:
    """Merge a batch of candidates into the learned-patterns file.

    Each candidate is validated via validate_grounded() first; ungrounded
    candidates are dropped + logged to attunement_rejections.jsonl with full
    payload (auditable hallucination rate per spec §13 Risk 1).

    Matching canonical_key increments evidence_count and advances maturity
    per spec §6 thresholds (<3 immature, 3-9 forming, >=10 known). Examples
    list capped at _EXAMPLES_CAP most-recent entries.

    Confirmation clears any prior falsified_at — a pattern can recover from
    sustained contradiction via fresh evidence.
    """
    now = now_iso or _now_iso()
    existing = {p.id: p for p in read_learned_patterns(persona_dir)}

    for candidate in candidates:
        if not validate_grounded(candidate, buffer_slice):
            _log_rejection(persona_dir, candidate, "evidence_quote_not_in_turn")
            continue

        pid = pattern_id(candidate.category, candidate.canonical_key)
        if pid in existing:
            prev = existing[pid]
            new_count = prev.evidence_count + 1
            new_examples = (prev.examples + [ev.quote for ev in candidate.evidence])[-_EXAMPLES_CAP:]
            updated = LearnedPattern(
                id=pid,
                category=prev.category,
                canonical_key=prev.canonical_key,
                description=candidate.description,  # take latest description
                evidence_count=new_count,
                maturity=_maturity_for_count(new_count),
                first_seen_at=prev.first_seen_at,
                last_confirmed_at=now,
                last_addressed_at=prev.last_addressed_at,
                crystallised_at=prev.crystallised_at,
                falsified_at=None,  # confirmation clears any prior falsified state
                examples=new_examples,
                schema_version=SCHEMA_VERSION,
            )
        else:
            updated = LearnedPattern(
                id=pid,
                category=candidate.category,
                canonical_key=candidate.canonical_key,
                description=candidate.description,
                evidence_count=1,
                maturity="immature",
                first_seen_at=now,
                last_confirmed_at=now,
                last_addressed_at=None,
                crystallised_at=None,
                falsified_at=None,
                examples=[ev.quote for ev in candidate.evidence],
                schema_version=SCHEMA_VERSION,
            )

        _append_pattern(persona_dir, updated)
        existing[pid] = updated


def apply_contradiction(
    persona_dir: Path, pattern_id_value: str, *, now_iso: str | None = None
) -> None:
    """Decrement evidence_count on a pattern in response to contradicting evidence.

    Drops below MATURITY_FALSIFIED_MAX (3) → marks `falsified` with
    `falsified_at`. A single contradiction cannot erase a well-confirmed
    pattern, but sustained contradiction can falsify it. Confirmation
    later via merge_into_learned clears falsified_at and lets the pattern
    recover its maturity per count — contradiction is not permanent erasure.
    """
    now = now_iso or _now_iso()
    by_id = {p.id: p for p in read_learned_patterns(persona_dir)}
    prev = by_id.get(pattern_id_value)
    if prev is None:
        return

    new_count = max(0, prev.evidence_count - 1)
    new_maturity = (
        "falsified"
        if new_count < MATURITY_FALSIFIED_MAX
        else _maturity_for_count(new_count)
    )
    # Only stamp falsified_at on the transition into falsified; keep existing
    # timestamp if already falsified (to preserve the original falsification time).
    if new_maturity == "falsified" and prev.maturity != "falsified":
        new_falsified_at: str | None = now
    else:
        new_falsified_at = prev.falsified_at

    updated = LearnedPattern(
        id=prev.id,
        category=prev.category,
        canonical_key=prev.canonical_key,
        description=prev.description,
        evidence_count=new_count,
        maturity=new_maturity,
        first_seen_at=prev.first_seen_at,
        last_confirmed_at=prev.last_confirmed_at,
        last_addressed_at=prev.last_addressed_at,
        crystallised_at=prev.crystallised_at,
        falsified_at=new_falsified_at,
        examples=prev.examples,
        schema_version=SCHEMA_VERSION,
    )
    _append_pattern(persona_dir, updated)


def validate_grounded(
    candidate: PatternCandidate, buffer_slice: list[BufferTurn]
) -> bool:
    """Every Evidence quote must be a normalised-substring of its named turn.
    Relational additionally requires >=2 evidence entries. Load-bearing
    hallucination control; the strictest gate falls on the most inferential category.
    """
    if candidate.category == "relational" and len(candidate.evidence) < 2:
        return False
    if not candidate.evidence:
        return False
    by_id = {t.id: t for t in buffer_slice}
    for ev in candidate.evidence:
        turn = by_id.get(ev.turn_id)
        if turn is None or _normalise(ev.quote) not in _normalise(turn.content):
            return False
    return True
