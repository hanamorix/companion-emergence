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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from brain.growth.crystallizers.vocabulary import crystallize_vocabulary
from brain.growth.log import GrowthLogEvent, append_growth_event
from brain.growth.proposal import EmotionProposal
from brain.memory.store import MemoryStore

if TYPE_CHECKING:
    from brain.health.anomaly import BrainAnomaly

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
    anomalies_collector: list[BrainAnomaly] | None = None,
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

    `anomalies_collector` (optional): when the heartbeat tick passes its
    per-tick anomaly list, any anomaly produced by reading the vocabulary
    file (corruption, schema mismatch) gets appended so it surfaces in the
    audit log + compact CLI alongside heartbeat-engine anomalies. Pass None
    when calling `run_growth_tick` standalone (e.g., from tests).
    """
    vocab_path = persona_dir / "emotion_vocabulary.json"
    log_path = persona_dir / "emotion_growth.log.jsonl"

    current_names, vocab_anomaly = _read_current_vocabulary_names(vocab_path)
    if vocab_anomaly is not None and anomalies_collector is not None:
        anomalies_collector.append(vocab_anomaly)

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


def _read_current_vocabulary_names(
    vocab_path: Path,
) -> tuple[set[str], BrainAnomaly | None]:
    """Return (set of emotion names, optional anomaly) for the vocabulary file.

    Distinguishes three load outcomes:
      - Missing file → (empty set, None) silently (fresh persona; expected).
      - Corrupt JSON or wrong schema → quarantine + heal from .bak or reset to
        default; returns (names_from_recovered_data, BrainAnomaly). The caller
        (run_growth_tick) feeds the anomaly into its `anomalies_collector` so
        it surfaces in the heartbeat audit log. Logs WARNING locally too.
      - Well-formed → (set of names, None).
    """
    if not vocab_path.exists():
        return set(), None

    from brain.health.attempt_heal import attempt_heal

    def _schema_validator(data: object) -> None:
        if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
            raise ValueError("emotion_vocabulary schema invalid: missing 'emotions' list")

    data, anomaly = attempt_heal(
        vocab_path,
        lambda: {"version": 1, "emotions": []},
        schema_validator=_schema_validator,
    )

    if anomaly is not None:
        logger.warning(
            "growth scheduler: emotion_vocabulary at %s anomaly %s (action=%s); "
            "proceeding with recovered data",
            vocab_path,
            anomaly.kind,
            anomaly.action,
        )

    names = {e["name"] for e in data.get("emotions", []) if isinstance(e, dict) and "name" in e}
    return names, anomaly


def _is_valid_name(name: str) -> bool:
    if not name:
        return False
    return not any(c in name for c in _INVALID_NAME_CHARS)


def _append_to_vocabulary(vocab_path: Path, proposal: EmotionProposal) -> None:
    """Atomic append to emotion_vocabulary.json via save_with_backup."""
    from brain.health.adaptive import compute_treatment
    from brain.health.attempt_heal import save_with_backup

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
    treatment = compute_treatment(vocab_path.parent, vocab_path.name)
    save_with_backup(vocab_path, data, backup_count=treatment.backup_count)


def _default_reason_for(proposal: EmotionProposal) -> str:
    """Phase 2a default — Phase 2b crystallizer fills `proposal.reason` directly.

    For now we synthesize a short reason since EmotionProposal doesn't carry
    one — the dataclass design here matches Phase 2b's likely shape but until
    the crystallizer produces one we describe by score + evidence count.
    """
    return f"score={proposal.score:.2f}, evidence_count={len(proposal.evidence_memory_ids)}"
