"""New v0.0.10 candidate emitters: reflex firings and research completions.

Each emitter has a gate function that decides whether to emit, plus an
emit function that calls into brain.initiate.emit.emit_initiate_candidate.

Rejected gate checks write to gate_rejections.jsonl (separate from the
main audit log — rejection volume would otherwise drown signal).

Thresholds are loaded from gate_thresholds.json in the persona dir,
with defaults baked in. Operator can tune without code change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.schemas import CandidateSource, SemanticContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateThresholds:
    reflex_confidence_min: float = 0.70
    reflex_flinch_intensity_min: float = 0.60
    reflex_anti_flood_hours: float = 4.0
    research_maturity_min: float = 0.75
    research_topic_overlap_min: float = 0.30
    research_freshness_minutes: float = 30
    meta_anti_flood_minutes: float = 30
    meta_max_queue_depth: int = 6


def load_gate_thresholds(persona_dir: Path) -> GateThresholds:
    """Load thresholds from <persona_dir>/gate_thresholds.json with defaults.

    Defaults defined on the dataclass. Persona file overrides any subset
    of fields. Missing file => all defaults.
    """
    path = persona_dir / "gate_thresholds.json"
    if not path.exists():
        return GateThresholds()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("gate_thresholds.json read failed (%s); using defaults", exc)
        return GateThresholds()
    if not isinstance(raw, dict):
        logger.warning("gate_thresholds.json is not a JSON object; using defaults")
        return GateThresholds()
    valid_names = {f.name for f in fields(GateThresholds)}
    overrides: dict[str, Any] = {k: v for k, v in raw.items() if k in valid_names}
    return GateThresholds(**overrides)


def write_gate_rejection(
    persona_dir: Path,
    *,
    ts: datetime,
    source: str,
    source_id: str,
    gate_name: str,
    threshold_value: float,
    observed_value: float,
) -> None:
    """Append one rejection row to gate_rejections.jsonl. Never raises."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    path = persona_dir / "gate_rejections.jsonl"
    row = {
        "ts": ts.isoformat(),
        "source": source,
        "source_id": source_id,
        "gate_name": gate_name,
        "threshold_value": threshold_value,
        "observed_value": observed_value,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("gate_rejections append failed for %s: %s", path, exc)


def check_shared_meta_gates(
    persona_dir: Path,
    *,
    source: CandidateSource,
    now: datetime,
    is_rest_state: bool,
    thresholds: GateThresholds,
) -> tuple[bool, str | None]:
    """Apply the meta-gates that hold for every new v0.0.10 emitter.

    Returns (allowed, reason). When allowed is False, `reason` is a
    structured tag suitable for gate_rejections.jsonl.
    """
    if is_rest_state:
        return False, "rest_state"

    # Read current queue once for the two remaining checks.
    candidates = read_candidates(persona_dir)

    # Per-source anti-flood: at most 1 candidate of this source in last N min.
    anti_flood_cutoff = now - timedelta(minutes=thresholds.meta_anti_flood_minutes)
    for c in candidates:
        if c.source != source:
            continue
        try:
            c_ts = datetime.fromisoformat(c.ts)
        except ValueError:
            continue
        if c_ts >= anti_flood_cutoff:
            return False, "per_source_anti_flood"

    # Queue depth ceiling.
    if len(candidates) >= thresholds.meta_max_queue_depth:
        return False, "queue_depth_max"

    return True, None


class ReflexFiringLike(Protocol):
    """Duck-typed shape gate_reflex_firing inspects on a reflex firing."""
    pattern_id: str
    confidence: float
    flinch_intensity: float
    linked_memory_ids: list[str]
    triggered_by_companion_outbound: bool
    ts: datetime


def gate_reflex_firing(
    persona_dir: Path,
    *,
    firing: ReflexFiringLike,
    thresholds: GateThresholds,
) -> tuple[bool, str | None]:
    """Per-source gate for the reflex_firing emitter.

    Order of checks: confidence -> flinch -> anti-feedback -> pattern anti-flood.
    """
    if firing.confidence < thresholds.reflex_confidence_min:
        return False, "confidence_min"
    if firing.flinch_intensity < thresholds.reflex_flinch_intensity_min:
        return False, "flinch_intensity_min"
    if firing.triggered_by_companion_outbound:
        return False, "anti_feedback"

    # Anti-flood: same pattern_id must not have emitted in the last N hours.
    cutoff = firing.ts - timedelta(hours=thresholds.reflex_anti_flood_hours)
    for c in read_candidates(persona_dir):
        if c.source != "reflex_firing":
            continue
        meta = c.semantic_context.source_meta or {}
        if meta.get("pattern_id") != firing.pattern_id:
            continue
        try:
            c_ts = datetime.fromisoformat(c.ts)
        except ValueError:
            continue
        if c_ts >= cutoff:
            return False, "pattern_anti_flood"

    return True, None


def emit_reflex_firing_candidate(
    persona_dir: Path,
    *,
    firing: ReflexFiringLike,
    firing_log_id: str,
    now: datetime,
) -> None:
    """Write a reflex_firing candidate to the initiate queue.

    Idempotent on (source, source_id) per emit_initiate_candidate's contract.
    Callers must run gate_reflex_firing + check_shared_meta_gates first;
    this function does NOT re-check gates.
    """
    sc = SemanticContext(
        linked_memory_ids=list(firing.linked_memory_ids),
        topic_tags=[],
        source_meta={
            "pattern_id": firing.pattern_id,
            "confidence": firing.confidence,
            "flinch_intensity": firing.flinch_intensity,
        },
    )
    emit_initiate_candidate(
        persona_dir,
        kind="message",
        source="reflex_firing",
        source_id=firing_log_id,
        semantic_context=sc,
        now=now,
    )


class ResearchThreadLike(Protocol):
    """Duck-typed shape gate_research_completion inspects on a research thread."""
    thread_id: str
    topic: str
    maturity_score: float
    summary_excerpt: str
    linked_memory_ids: list[str]
    completed_at: datetime
    previously_linked_to_audit: bool


def gate_research_completion(
    persona_dir: Path,
    *,
    thread: ResearchThreadLike,
    now: datetime,
    topic_overlap_score: float,
    thresholds: GateThresholds,
) -> tuple[bool, str | None]:
    """Per-source gate for the research_completion emitter.

    Order: maturity -> previously-linked -> topic-overlap -> freshness.
    `topic_overlap_score` is computed by the caller (the research engine
    has access to recent conversation embeddings; we don't re-compute here).
    """
    if thread.maturity_score < thresholds.research_maturity_min:
        return False, "maturity_min"
    if thread.previously_linked_to_audit:
        return False, "previously_linked"
    if topic_overlap_score < thresholds.research_topic_overlap_min:
        return False, "topic_overlap_min"

    freshness_cutoff = now - timedelta(minutes=thresholds.research_freshness_minutes)
    if thread.completed_at < freshness_cutoff:
        return False, "freshness_window"

    return True, None


def emit_research_completion_candidate(
    persona_dir: Path,
    *,
    thread: ResearchThreadLike,
    topic_overlap_score: float,
    now: datetime,
) -> None:
    """Write a research_completion candidate to the initiate queue.

    Idempotent on (source, source_id). Callers must run
    gate_research_completion + check_shared_meta_gates first.
    """
    sc = SemanticContext(
        linked_memory_ids=list(thread.linked_memory_ids),
        topic_tags=[thread.topic],
        source_meta={
            "thread_topic": thread.topic,
            "maturity_score": thread.maturity_score,
            "summary_excerpt": thread.summary_excerpt,
            "topic_overlap_score": topic_overlap_score,
        },
    )
    emit_initiate_candidate(
        persona_dir,
        kind="message",
        source="research_completion",
        source_id=thread.thread_id,
        semantic_context=sc,
        now=now,
    )
