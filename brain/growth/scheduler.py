"""Growth scheduler — orchestrates crystallizers + applies decisions atomically.

The scheduler is the primary mutator of `emotion_vocabulary.json` during a
growth tick, but other paths also write to that file:
- `brain/tools/impls/add_memory.py` — a new emotion word in a kept memory is
  registered at once, through `persona_loader.register_new_word` (the same
  call the scheduler uses for a crystallized word).
- `brain/emotion/persona_loader.py` — at load time, `_heal_referenced_but_unregistered`
  registers emotions memories use but the file lacks, and a missing or reset
  file is rebuilt from memories.
- `brain/health/vocab_repair.py` — merges spelling-variant twins, bumps legacy
  stub half-lives and fills in placeholder descriptions; runs at startup and
  every 6 hours.
- `brain/health/walker.py` and `brain/health/json_io` — repair a corrupt file
  (restore a `.bak` or reset to empty).
Only `register_new_word` (so the scheduler and `add_memory`) and vocab repair
take the file's cross-process lock (`file_lock(vocab_path)`) and re-read the
file under it; the load-time heal and rebuild and the json_io/walker repairs do not.

No engine writes `emotion_growth.log.jsonl` except through `run_growth_tick`.

Per principle audit 2026-04-25 (Phase 2a §4): the brain owns its own
growth. Crystallizers decide; the scheduler applies; the log records
biographically. No human approval gate, no candidate queue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from brain.emotion import vocabulary
from brain.emotion.persona_loader import register_new_word
from brain.growth.crystallizers.vocabulary import (
    VOCABULARY_CRYSTALLIZER_STATUS,
    crystallize_vocabulary,
)
from brain.growth.log import GrowthLogEvent, append_growth_event
from brain.growth.proposal import EmotionProposal
from brain.health.json_io import Recovery, read_json_result
from brain.memory.store import MemoryStore
from brain.paths import cadence_state_path
from brain.utils.file_lock import file_lock

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
      1. Skip silently if name already in current vocabulary (re-proposing
         is normal; not a rejection).
      2. Reject (with warning) if the name fails the new-word gate or the
         vocabulary file cannot be written safely.
      3. Else: register it through `register_new_word` (this process and
         {persona_dir}/emotion_vocabulary.json, under the file lock) and append a GrowthLogEvent to
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
        if proposal.name in current_names:
            # Idempotent skip — re-proposal is normal, not a rejection.
            continue
        if dry_run:
            # Would-have-added: counted when the gate accepts the name; nothing is written.
            if vocabulary.validate_new_name(proposal.name) is None:
                proposals_rejected += 1
            else:
                emotions_added += 1
            continue

        registered = register_new_word(
            vocab_path,
            proposal.name,
            half_life_days=proposal.decay_half_life_days,
            description=proposal.description,
        )
        if registered is None:
            # Refused by the gate, or the file could not be read or written safely (logged there).
            proposals_rejected += 1
            continue

        emotions_added += 1
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

    def _schema_validator(data: object) -> None:
        if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
            raise ValueError("emotion_vocabulary schema invalid: missing 'emotions' list")

    # The heal can restore or reset the file: it holds the lock register_new_word and the loader take, so a word added
    # meanwhile is kept.
    with file_lock(vocab_path):
        res = read_json_result(
            vocab_path,
            default=lambda: {"version": 1, "emotions": []},
            recover=Recovery.HEAL,
            validate=_schema_validator,
        )
    data, anomaly = res.data, res.anomaly
    if anomaly is not None:
        # This repair has no store: a restored .bak or a reset can drop words only the memories still carry. Delete the
        # one-time heal marker so the next load with a store re-registers them, as the health walker does (#784).
        marker = cadence_state_path(vocab_path.parent, "vocab_heal_final.json")
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove %s after repairing %s; the vocabulary heal may not re-run", marker, vocab_path.name, exc_info=True)

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
