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
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from brain.growth.crystallizers.vocabulary import (
    VOCABULARY_CRYSTALLIZER_STATUS,
    crystallize_vocabulary,
)
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
    vocabulary_crystallizer_status: str = "implemented"


def _should_run_growth_tick(
    *, last_tick: datetime | None, now: datetime, throttle_days: float,
) -> bool:
    """True iff enough time has elapsed since the last growth tick.

    `last_tick=None` means never-run; always returns True.
    Boundary `now - last_tick == throttle_days` returns True (inclusive).
    """
    if last_tick is None:
        return True
    return (now - last_tick) >= timedelta(days=throttle_days)


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
        # Phase 4.2 — emit initiate candidate after vocabulary crystallization
        # commits to disk. Wrapped in try/except so emit failures can't crash
        # the scheduler.
        _emit_vocabulary_initiate_candidate(persona_dir, proposal, store=store, now=now)

    # Creative DNA crystallization (spec §5)
    if not dry_run and (persona_dir / "persona_config.json").exists():
        try:
            from brain.bridge.provider import get_provider
            from brain.growth.crystallizers.creative_dna import crystallize_creative_dna
            from brain.persona_config import PersonaConfig

            cfg = PersonaConfig.load(persona_dir / "persona_config.json")
            provider = get_provider(cfg.provider)
            crystallize_creative_dna(
                store=store,
                persona_dir=persona_dir,
                provider=provider,
                persona_name=persona_dir.name,
                now=now,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("creative_dna crystallizer skipped: %s", exc)

    return GrowthTickResult(
        emotions_added=emotions_added,
        proposals_seen=len(proposals),
        proposals_rejected=proposals_rejected,
        vocabulary_crystallizer_status=VOCABULARY_CRYSTALLIZER_STATUS,
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


def _emit_vocabulary_initiate_candidate(
    persona_dir: Path,
    proposal: EmotionProposal,
    *,
    store: MemoryStore,
    now: datetime,
) -> None:
    """Emit one initiate candidate after vocabulary commit. Try/except wrapped.

    Phase 4.2 of the initiate physiology pipeline. An emit failure must not
    crash the growth tick.

    Pulls a max-pooled emotion vector across recent active memories so the
    candidate carries a real signal of what's been emotionally alive in
    the window that produced this emotion. rolling_baseline /
    current_resonance / delta_sigma stay zero — heartbeat-specific signals
    that non-periodic emitters don't compute.
    """
    try:
        from brain.initiate.emit import emit_initiate_candidate
        from brain.initiate.schemas import EmotionalSnapshot, SemanticContext

        emotion_vector = _aggregate_recent_emotion_vector(store, now=now)

        emit_initiate_candidate(
            persona_dir,
            kind="message",
            source="crystallization",
            source_id=f"vocabulary_emotion:{proposal.name}",
            emotional_snapshot=EmotionalSnapshot(
                vector=emotion_vector,
                rolling_baseline_mean=0.0,
                rolling_baseline_stdev=0.0,
                current_resonance=0.0,
                delta_sigma=0.0,
            ),
            semantic_context=SemanticContext(
                linked_memory_ids=list(proposal.evidence_memory_ids)[:5],
                topic_tags=[proposal.name],
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("vocabulary crystallization initiate emit failed: %s", exc)


def _aggregate_recent_emotion_vector(
    store: MemoryStore, *, now: datetime, look_back_days: int = 30,
) -> dict[str, float]:
    """Return a max-pooled emotion vector across recent active memories.

    Empty dict on any failure.
    """
    try:
        from brain.emotion.aggregate import aggregate_state
        from brain.utils.memory import list_conversation_memories

        cutoff = now - timedelta(days=look_back_days)
        recent = [
            m for m in list_conversation_memories(store, active_only=True)
            if m.created_at >= cutoff and m.emotions
        ]
        if not recent:
            return {}
        state = aggregate_state(recent)
        return dict(state.emotions)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vocabulary: recent-emotion aggregation failed: %s", exc)
        return {}


def _default_reason_for(proposal: EmotionProposal) -> str:
    """Phase 2a default — Phase 2b crystallizer fills `proposal.reason` directly.

    For now we synthesize a short reason since EmotionProposal doesn't carry
    one — the dataclass design here matches Phase 2b's likely shape but until
    the crystallizer produces one we describe by score + evidence count.
    """
    return f"score={proposal.score:.2f}, evidence_count={len(proposal.evidence_memory_ids)}"
