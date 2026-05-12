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
    compose_decision,
    compose_decision_voice_edit,
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
    """Run the three-prompt pipeline on a single candidate, write audit, remove.

    Two routes by ``candidate.kind``:

    * ``message`` — full three-prompt path (subject -> tone -> decision)
      plus cost-cap gate for ``send_*`` decisions.
    * ``voice_edit_proposal`` — skip subject+tone (the proposal already
      carries the diff); use ``compose_decision_voice_edit`` with the
      current voice template + recent evolutions. Voice edits bypass the
      cost-cap gate (the daily reflection tick is the rate limiter).
    """
    audit_id = make_audit_id(now)
    try:
        if candidate.kind == "voice_edit_proposal":
            proposal = candidate.proposal or {}
            subject = proposal.get("rationale", "voice edit proposal")
            tone_rendered = (
                f"Proposing to change my voice: "
                f"{proposal.get('old_text', '')!r} -> "
                f"{proposal.get('new_text', '')!r}. Rationale: "
                f"{proposal.get('rationale', '')}"
            )
            # Pull recent accepted voice evolutions so the decision prompt
            # can see Nell's evolution arc. Best-effort — a missing/locked
            # SoulStore degrades to "(no recent voice edits)".
            recent_evolutions: list[dict[str, Any]] = []
            try:
                from brain.soul.store import SoulStore

                soul_store = SoulStore(
                    str(persona_dir / "crystallizations.db")
                )
                try:
                    recent_evolutions = [
                        {
                            "accepted_at": v.accepted_at,
                            "old_text": v.old_text,
                            "new_text": v.new_text,
                        }
                        for v in soul_store.list_voice_evolution()
                    ]
                finally:
                    soul_store.close()
            except Exception:
                logger.exception(
                    "voice-edit review: SoulStore read failed for %s",
                    candidate.candidate_id,
                )
            voice_path = persona_dir / "nell-voice.md"
            current_voice = (
                voice_path.read_text(encoding="utf-8")
                if voice_path.exists()
                else ""
            )
            decision_result = compose_decision_voice_edit(
                provider,
                proposal=proposal,
                current_voice_template=current_voice,
                recent_voice_evolutions=recent_evolutions,
                current_local_time=now,
            )
        else:
            # Hydrate semantic memory IDs into content excerpts. The compose
            # prompt needs the lived texture, not opaque "m_xyz" handles, or
            # the LLM has no signal to thread the message back to anything.
            # Best-effort: any failure falls back to the IDs so the pipeline
            # still produces *some* output rather than crashing the tick.
            excerpts: list[str]
            try:
                mem_store = MemoryStore(
                    persona_dir / "memories.db", integrity_check=False,
                )
                try:
                    excerpts = []
                    for mem_id in candidate.semantic_context.linked_memory_ids[:5]:
                        mem = mem_store.get(mem_id)
                        if mem is not None:
                            excerpts.append(mem.content[:240])
                finally:
                    mem_store.close()
            except Exception:
                logger.exception(
                    "memory hydration for compose_subject failed; "
                    "using IDs as fallback for candidate %s",
                    candidate.candidate_id,
                )
                excerpts = list(candidate.semantic_context.linked_memory_ids[:5])

            subject = compose_subject(
                provider,
                candidate,
                semantic_memory_excerpts=excerpts,
            )
            tone_rendered = compose_tone(
                provider,
                subject=subject,
                candidate=candidate,
                voice_template=voice_template,
            )
            decision_result = compose_decision(
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
    gate_check: dict[str, Any] = {"allowed": True, "reason": None}

    # Voice-edit candidates are bucketed with `quiet` but bypass the cost-
    # cap gate — the daily reflection tick is the rate limiter (Task 14).
    if (
        candidate.kind != "voice_edit_proposal"
        and final_decision in ("send_notify", "send_quiet")
    ):
        urgency = "notify" if final_decision == "send_notify" else "quiet"
        allowed, reason = check_send_allowed(
            persona_dir, urgency=urgency, now=now
        )
        gate_check = {"allowed": allowed, "reason": reason}
        if not allowed:
            final_decision = "hold"
            final_reasoning = f"blocked_by_gate: {reason}"

    diff_payload: str | None = None
    if candidate.kind == "voice_edit_proposal" and candidate.proposal:
        diff_payload = candidate.proposal.get("diff", "") or None

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
        diff=diff_payload,
    )
    append_audit_row(persona_dir, row)

    if final_decision in ("send_notify", "send_quiet"):
        update_audit_state(
            persona_dir,
            audit_id=audit_id,
            new_state="delivered",
            at=now.isoformat(),
        )

        # Bridge event-bus publish — wakes the renderer's banner pipeline.
        # The ChatPanel listens for `initiate_delivered` and surfaces the
        # outreach inline; without this, the user never sees the message
        # the brain just decided to send. Best-effort: a publish failure
        # leaves the audit + memory rows intact so the next ambient recall
        # still has a record to surface.
        try:
            from brain.bridge import events

            events.publish(
                "initiate_delivered",
                audit_id=audit_id,
                body=tone_rendered,
                urgency="notify" if final_decision == "send_notify" else "quiet",
                state="delivered",
                timestamp=now.isoformat(),
            )
        except Exception:
            logger.exception(
                "publish initiate_delivered failed for audit %s "
                "(audit row still written)", audit_id,
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
