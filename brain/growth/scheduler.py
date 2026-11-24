"""Growth scheduler — orchestrates crystallizers + applies decisions atomically.

The scheduler adds names to `emotion_vocabulary.json` during a growth tick, through
`brain/emotion/vocabulary_file.py`, the one writer of that file (the persona loader's
memory heal and the vocabulary repair use it too).

No engine writes `emotion_growth.log.jsonl` except through `run_growth_tick`.

Per principle audit 2026-04-25 (Phase 2a §4): the brain owns its own
growth. Crystallizers decide; the scheduler applies; the log records
biographically. No human approval gate, no candidate queue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from brain.emotion import vocabulary_access, vocabulary_file
from brain.emotion.vocabulary import gate_name
from brain.growth.crystallizers.vocabulary import (
    VOCABULARY_CRYSTALLIZER_STATUS,
    crystallize_vocabulary,
)
from brain.growth.log import GrowthLogEvent, append_growth_event
from brain.growth.proposal import EmotionProposal
from brain.health.json_io import Recovery
from brain.memory.store import MemoryStore

if TYPE_CHECKING:
    from brain.health.anomaly import BrainAnomaly

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class GrowthTickResult:
    """Outcome of one growth tick."""

    emotions_added: int
    proposals_seen: int
    proposals_rejected: int
    vocabulary_crystallizer_status: str = "implemented"


def _should_run_growth_tick(
    *,
    last_tick: datetime | None,
    now: datetime,
    throttle_days: float,
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
      1. Tidy the name and reject it (the gate logs why) if it is empty once
         tidied or too long (`vocabulary.gate_name`); the tidied name is the
         one used from here on.
      2. Skip silently if name already in current vocabulary (re-proposing
         is normal; not a rejection).
      3. Else: add it to {persona_dir}/emotion_vocabulary.json through
         `vocabulary_file.add` and, when that wrote it, append a GrowthLogEvent to
         {persona_dir}/emotion_growth.log.jsonl (atomic per `log.py`). A name the
         file could not take (`cannot_write`, logged by the writer) is neither
         counted nor logged as added, and the tick adds no further name (no
         second wait on the same lock); the crystallizer proposes it again.

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

    current_names, vocab_anomaly = _read_current_vocabulary_names(persona_dir)
    if vocab_anomaly is not None and anomalies_collector is not None:
        anomalies_collector.append(vocab_anomaly)

    from brain.emotion.engine import engine_for

    live_engine = engine_for(persona_dir)
    proposals = crystallize_vocabulary(
        store,
        current_vocabulary_names=current_names,
        eng=live_engine if live_engine.is_live() else None,
    )

    emotions_added = 0
    proposals_rejected = 0

    for proposal in proposals:
        tidied = gate_name(proposal.name)
        if tidied is None:
            # The gate has logged why.
            proposals_rejected += 1
            continue
        if tidied != proposal.name:
            proposal = replace(proposal, name=tidied)
        if proposal.name in current_names:
            # Idempotent skip — re-proposal is normal, not a rejection.
            continue

        if dry_run:
            emotions_added += 1
            continue

        result = vocabulary_file.add(vocab_path, [(proposal.name, _entry_fields(proposal))])[0]
        if result.outcome == "cannot_write":
            break  # the writer has logged why and nothing was written: no further wait for the lock this tick
        current_names.add(proposal.name)
        if not result.changed:
            continue  # the file took the name from another writer since the read: nothing to record
        emotions_added += 1
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
            from brain.bridge.model_tier import TIER_BACKGROUND_CLASSIFIER, build_tier_provider
            from brain.growth.crystallizers.creative_dna import crystallize_creative_dna

            provider = build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER)
            crystallize_creative_dna(
                store=store,
                persona_dir=persona_dir,
                provider=provider,
                persona_name=persona_dir.name,
                now=now,
            )
        except Exception as exc:  # noqa: BLE001
            from brain.bridge.cli_throttle import ThrottleDeferred

            if isinstance(exc, ThrottleDeferred):
                raise  # #456: "not now": the heartbeat records no growth run (see _try_run_growth)
            logger.warning("creative_dna crystallizer skipped: %s", exc)

    return GrowthTickResult(
        emotions_added=emotions_added,
        proposals_seen=len(proposals),
        proposals_rejected=proposals_rejected,
        vocabulary_crystallizer_status=VOCABULARY_CRYSTALLIZER_STATUS,
    )


def _read_current_vocabulary_names(
    persona_dir: Path,
) -> tuple[set[str], BrainAnomaly | None]:
    """Return (set of emotion names in their tidied form, optional anomaly) for her vocabulary.

    The names come from the lookup (`vocabulary_access.lookup`, "all": every entry of her vocabulary, the framework's own
    included), tidied, because the dedupe compares tidied names. The file is read only for its health: a damaged file is
    quarantined and healed from a backup or reset to the default (json_io's own recovery), and the anomaly is returned so
    the caller (run_growth_tick) feeds it into its `anomalies_collector` and it surfaces in the heartbeat audit log. Logs
    WARNING locally too. A missing file gives no anomaly (fresh persona; expected).
    """
    vocab_path = vocabulary_file.vocabulary_path(persona_dir)
    anomaly = None
    if vocab_path.exists():
        anomaly = vocabulary_file.read(vocab_path, recover=Recovery.HEAL).anomaly
        if anomaly is not None:
            logger.warning(
                "growth scheduler: emotion_vocabulary at %s anomaly %s (action=%s); "
                "proceeding with her registered vocabulary",
                vocab_path,
                anomaly.kind,
                anomaly.action,
            )
    return set(vocabulary_access.lookup(persona_dir, vocabulary_access.ALL)), anomaly


def _entry_fields(proposal: EmotionProposal) -> dict:
    """The fields a new vocabulary entry takes from a proposal (the name is the writer's)."""
    return {
        "description": proposal.description,
        "category": "persona_extension",
        "decay_half_life_days": proposal.decay_half_life_days,
        "intensity_clamp": 10.0,
    }


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

        emotion_vector = _aggregate_recent_emotion_vector(store, now=now, persona_dir=persona_dir)

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
    store: MemoryStore,
    *,
    now: datetime,
    persona_dir: Path,
    look_back_days: int = 30,
) -> dict[str, float]:
    """Return her recent emotion vector: the emotion engine's short averages above 0. Context only.

    Empty while the engine is not reading (not switched over, or its state cannot be read), and on any failure. The old
    max-pool of recent memories went in emotion rework B1 PR 7.
    """
    try:
        from brain.emotion.engine import engine_for

        eng = engine_for(persona_dir)
        if not eng.is_live():
            return {}
        return {name: r.short for name, r in eng.read().items() if r.short > 0}
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
