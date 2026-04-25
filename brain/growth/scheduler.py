"""Growth scheduler — orchestrates crystallizers + applies decisions atomically.

The scheduler is the *only* mutator of `emotion_vocabulary.json` and
`emotion_growth.log.jsonl` during a growth tick. No engine touches these
files except through `run_growth_tick`.

Per principle audit 2026-04-25 (Phase 2a §4): the brain owns its own
growth. Crystallizers decide; the scheduler applies; the log records
biographically. No human approval gate, no candidate queue.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from brain.growth.crystallizers.vocabulary import crystallize_vocabulary
from brain.growth.log import GrowthLogEvent, append_growth_event
from brain.growth.proposal import EmotionProposal
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Same character allowlist as brain.paths.get_persona_dir — names that
# could appear as filesystem path components or trip JSON parsing get
# rejected before they reach disk.
_INVALID_NAME_CHARS = ("/", "\\", "{", "}")


@dataclass(frozen=True)
class GrowthTickResult:
    """Outcome of one growth tick."""

    emotions_added: int
    proposals_seen: int
    proposals_rejected: int


def run_growth_tick(
    persona_dir: Path,
    store: MemoryStore,
    now: datetime,
    *,
    dry_run: bool = False,
) -> GrowthTickResult:
    """Run all crystallizers, apply their proposals atomically.

    For each proposal:
      1. Skip silently if name already in current vocabulary (re-proposing
         is normal; not a rejection).
      2. Reject (with warning) if name fails character validation.
      3. Else: append to {persona_dir}/emotion_vocabulary.json (atomic
         `.new + os.replace`) and append a GrowthLogEvent to
         {persona_dir}/emotion_growth.log.jsonl (atomic per `log.py`).

    `dry_run=True` calls the crystallizer but skips both writes; the
    returned `emotions_added` reflects "would-have-added" semantics.
    """
    vocab_path = persona_dir / "emotion_vocabulary.json"
    log_path = persona_dir / "emotion_growth.log.jsonl"

    current_names = _read_current_vocabulary_names(vocab_path)

    proposals = crystallize_vocabulary(store, current_vocabulary_names=current_names)

    emotions_added = 0
    proposals_rejected = 0

    for proposal in proposals:
        if proposal.name in current_names:
            # Idempotent skip — re-proposal is normal, not a rejection.
            continue
        if not _is_valid_name(proposal.name):
            logger.warning(
                "growth scheduler: rejecting proposal with invalid name %r", proposal.name
            )
            proposals_rejected += 1
            continue

        emotions_added += 1
        if dry_run:
            continue

        _append_to_vocabulary(vocab_path, proposal)
        current_names.add(proposal.name)
        append_growth_event(
            log_path,
            GrowthLogEvent(
                timestamp=now,
                type="emotion_added",
                name=proposal.name,
                description=proposal.description,
                decay_half_life_days=proposal.decay_half_life_days,
                reason=_default_reason_for(proposal),
                evidence_memory_ids=proposal.evidence_memory_ids,
                score=proposal.score,
                relational_context=proposal.relational_context,
            ),
        )

    return GrowthTickResult(
        emotions_added=emotions_added,
        proposals_seen=len(proposals),
        proposals_rejected=proposals_rejected,
    )


def _read_current_vocabulary_names(vocab_path: Path) -> set[str]:
    """Return the set of emotion names currently in the persona's vocabulary file.

    Distinguishes three load outcomes:
      - Missing file → return empty set silently (fresh persona; expected).
      - Corrupt JSON or wrong schema → return empty set BUT log a WARNING
        with the file path. A silent fallback would mask the corruption,
        and the scheduler would then think "no emotions exist" and might
        re-add framework baselines on the next growth tick.
      - Well-formed → return the set of names.
    """
    if not vocab_path.exists():
        return set()
    try:
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(
            "growth scheduler: emotion_vocabulary at %s is corrupt JSON (%.200s); "
            "treating as empty for this tick — fix or quarantine the file",
            vocab_path,
            exc,
        )
        return set()
    if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
        logger.warning(
            "growth scheduler: emotion_vocabulary at %s has invalid schema "
            "(missing 'emotions' list); treating as empty for this tick",
            vocab_path,
        )
        return set()
    return {e["name"] for e in data["emotions"] if isinstance(e, dict) and "name" in e}


def _is_valid_name(name: str) -> bool:
    if not name:
        return False
    return not any(c in name for c in _INVALID_NAME_CHARS)


def _append_to_vocabulary(vocab_path: Path, proposal: EmotionProposal) -> None:
    """Atomic append to emotion_vocabulary.json — read, append entry, write `.new`, rename."""
    if vocab_path.exists():
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "emotions": []}
    data["emotions"].append(
        {
            "name": proposal.name,
            "description": proposal.description,
            "category": "persona_extension",
            "decay_half_life_days": proposal.decay_half_life_days,
            "intensity_clamp": 10.0,
        }
    )
    tmp = vocab_path.with_suffix(vocab_path.suffix + ".new")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, vocab_path)


def _default_reason_for(proposal: EmotionProposal) -> str:
    """Phase 2a default — Phase 2b crystallizer fills `proposal.reason` directly.

    For now we synthesize a short reason since EmotionProposal doesn't carry
    one — the dataclass design here matches Phase 2b's likely shape but until
    the crystallizer produces one we describe by score + evidence count.
    """
    return f"score={proposal.score:.2f}, evidence_count={len(proposal.evidence_memory_ids)}"
