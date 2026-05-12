"""Initiate review tick — orchestrator that ties emit + compose + audit + gates.

Single entry point: run_initiate_review_tick(persona_dir, provider, ...).

Per tick:
  1. Read up to cap_per_tick candidates from initiate_candidates.jsonl
  2. For each: run three-prompt pipeline (subject -> tone -> decision)
  3. If decision is a send, check the cost-cap gate
  4. Write audit row (decision + gate result)
  5. Remove the candidate from the queue (whether sent, held, or errored)

Fault isolation: a per-candidate exception is logged + recorded as
decision="error"; the candidate is removed (not requeued) so the queue
doesn't accumulate poison rows. A fresh emission on the same source_id
will rejoin the queue if the underlying event is still relevant.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.initiate.audit import (
    append_audit_row,
    read_recent_audit,
    update_audit_state,
)
from brain.initiate.compose import (
    DecisionResult,
    compose_decision,
    compose_subject,
    compose_tone,
)
from brain.initiate.emit import read_candidates, remove_candidate
from brain.initiate.gates import check_send_allowed
from brain.initiate.memory import write_initiate_memory
from brain.initiate.schemas import AuditRow, InitiateCandidate, make_audit_id
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def _build_send_history(persona_dir: Path, now: datetime) -> list[dict]:
    """Recent outbound shape for the decision prompt."""
    return [
        {
            "ts": row.ts,
            "urgency": "notify" if row.decision == "send_notify" else "quiet",
            "subject_preview": row.subject[:60],
        }
        for row in read_recent_audit(persona_dir, window_hours=24, now=now)
        if row.decision in ("send_notify", "send_quiet")
    ]


def _process_one_candidate(
    persona_dir: Path,
    candidate: InitiateCandidate,
    *,
    provider: Any,
    voice_template: str,
    now: datetime,
) -> None:
    """Run the three-prompt pipeline on a single candidate, write audit, remove."""
    audit_id = make_audit_id(now)
    try:
        subject = compose_subject(
            provider,
            candidate,
            semantic_memory_excerpts=candidate.semantic_context.linked_memory_ids,
        )
        tone_rendered = compose_tone(
            provider,
            subject=subject,
            candidate=candidate,
            voice_template=voice_template,
        )
        decision_result: DecisionResult = compose_decision(
            provider,
            rendered_message=tone_rendered,
            recent_send_history=_build_send_history(persona_dir, now),
            current_local_time=now,
            voice_edit_acceptance_rate=None,
        )
    except Exception as exc:
        logger.exception(
            "initiate composition failed for candidate %s", candidate.candidate_id
        )
        row = AuditRow(
            audit_id=audit_id,
            candidate_id=candidate.candidate_id,
            ts=now.isoformat(),
            kind=candidate.kind,
            subject="",
            tone_rendered="",
            decision="error",
            decision_reasoning=f"composition exception: {exc}",
            gate_check={"allowed": False, "reason": "composition_exception"},
            delivery=None,
        )
        append_audit_row(persona_dir, row)
        remove_candidate(persona_dir, candidate.candidate_id)
        return

    final_decision = decision_result.decision
    final_reasoning = decision_result.reasoning
    gate_check = {"allowed": True, "reason": None}

    if final_decision in ("send_notify", "send_quiet"):
        urgency = "notify" if final_decision == "send_notify" else "quiet"
        allowed, reason = check_send_allowed(
            persona_dir, urgency=urgency, now=now
        )
        gate_check = {"allowed": allowed, "reason": reason}
        if not allowed:
            final_decision = "hold"
            final_reasoning = f"blocked_by_gate: {reason}"

    row = AuditRow(
        audit_id=audit_id,
        candidate_id=candidate.candidate_id,
        ts=now.isoformat(),
        kind=candidate.kind,
        subject=subject,
        tone_rendered=tone_rendered,
        decision=final_decision,
        decision_reasoning=final_reasoning,
        gate_check=gate_check,
        delivery=None,
    )
    append_audit_row(persona_dir, row)

    if final_decision in ("send_notify", "send_quiet"):
        update_audit_state(
            persona_dir,
            audit_id=audit_id,
            new_state="delivered",
            at=now.isoformat(),
        )

        # Episodic memory mirror — dual-write the lived-experience texture.
        # The audit row above is the durable forensic record; the memory
        # entry surfaces this outreach to ambient recall on later turns.
        # Failure here is degraded (no recall surface) but not fatal.
        try:
            store = MemoryStore(persona_dir / "memories.db")
            try:
                write_initiate_memory(
                    store,
                    audit_id=audit_id,
                    subject=subject,
                    message=tone_rendered,
                    state="delivered",
                    ts=now.isoformat(),
                )
            finally:
                store.close()
        except Exception:
            logger.exception(
                "initiate memory write failed for audit %s", audit_id
            )

    remove_candidate(persona_dir, candidate.candidate_id)


def run_initiate_review_tick(
    persona_dir: Path,
    *,
    provider: Any,
    voice_template: str,
    cap_per_tick: int = 3,
    now: datetime | None = None,
) -> None:
    """Process up to cap_per_tick queued candidates through the pipeline.

    Fault-isolated per candidate: an exception in one candidate's
    processing does not block the others.
    """
    now = now or datetime.now(UTC)
    candidates = read_candidates(persona_dir)[:cap_per_tick]
    for candidate in candidates:
        try:
            _process_one_candidate(
                persona_dir,
                candidate,
                provider=provider,
                voice_template=voice_template,
                now=now,
            )
        except Exception:
            logger.exception(
                "initiate review tick: unrecoverable error on candidate %s",
                candidate.candidate_id,
            )
