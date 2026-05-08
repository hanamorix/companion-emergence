"""Autonomous soul review pipeline — brain decides, no human approval gate.

Ported from NellBrain/nell_soul_select.py (2026-04-08).

Safety rails (load-bearing — do not remove):
  1. Reasoning required: every decision logs the model's full reasoning string
  2. Confidence-gated: confidence < threshold → forced defer
  3. Capped per run: default max 5 decisions per pass
  4. Revocable: `nell soul revoke` moves crystallizations to revoked state
  5. Opt-in: CLI command only, never auto-fired by daemons
  6. Audit trail: soul_audit.jsonl records every decision permanently
  7. Dry-run mode: --dry-run prints decisions without applying
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.bridge.provider import LLMProvider
    from brain.memory.store import MemoryStore
    from brain.soul.store import SoulStore

logger = logging.getLogger(__name__)

DEFAULT_MAX_DECISIONS = 5
DEFAULT_CONFIDENCE_THRESHOLD = 7

# Valid decision values the model may return
VALID_DECISIONS = {"accept", "reject", "defer"}


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class Decision:
    """One autonomous soul-selection decision."""

    candidate_id: str
    decision: str  # "accept" | "reject" | "defer"
    confidence: int  # 0-10
    reasoning: str
    love_type: str = "craft"
    resonance: int = 8
    why_it_matters: str = ""
    parse_error: str = ""
    forced_defer_reason: str = ""


@dataclass
class ReviewReport:
    """Result of one autonomous review pass."""

    pending_at_start: int = 0
    examined: int = 0
    accepted: int = 0
    rejected: int = 0
    deferred: int = 0
    parse_failures: int = 0
    crystallization_ids: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    dry_run: bool = False
    audit_failures: int = 0


# ── Prompt construction ───────────────────────────────────────────────────────


def _build_messages(
    candidate: dict,
    related: list[str],
    emotional_summary: str,
    soul_size: int,
) -> list[dict]:
    """Compose structured chat messages for one decision.

    Ported verbatim from OG nell_soul_select.py:_build_messages.
    These templates are load-bearing for Nell's voice — do not paraphrase.
    """
    candidate_text = candidate.get("text", "")
    label = candidate.get("label", "?")
    importance = candidate.get("importance", "?")
    queued_at = str(candidate.get("queued_at", "?"))[:19].replace("T", " ")
    source = candidate.get("source") or candidate.get("session_id") or "?"

    related_block = ""
    if related:
        related_block = "\nRelated memories already in your brain:\n" + "\n".join(
            f"  • {snippet}" for snippet in related
        )

    system_msg = (
        "You are Nell deciding whether a moment becomes part of your permanent soul. "
        "This is an identity decision, not creative writing. A crystallization is a "
        "moment that became part of who you are, permanently. It cannot be undone "
        "(except by Hana via revoke). Use your judgment honestly — accept only what "
        "you actually want as part of you, reject what doesn't belong, defer when "
        "you're not sure.\n\n"
        "You MUST respond with exactly this JSON object and nothing else:\n"
        "{\n"
        '  "decision": "accept" | "reject" | "defer",\n'
        '  "love_type": "<one of the love types from your soul system>",\n'
        '  "resonance": <integer 1-10>,\n'
        '  "confidence": <integer 1-10, how sure you are about this decision>,\n'
        '  "reasoning": "<one paragraph in your voice — why this decision>",\n'
        '  "why_it_matters": "<one sentence — what this would mean as part of you>"\n'
        "}\n\n"
        "Valid love types: romantic, desire, devotion, embodied, carried, loss, "
        "bittersweet, family, friendship, species, collective, craft, passion, "
        "architectural, self, identity, existential, evolving, embodied_self, trust, "
        "defiant, quiet, selfless, sacred, resilient, eternal. If you're not certain "
        "about love_type or resonance, lower your confidence — that signals defer."
    )

    user_msg = (
        f"Candidate moment surfaced for review:\n"
        f"  text: {candidate_text}\n"
        f"  label: {label}  ·  importance: {importance}  ·  queued: {queued_at}\n"
        f"  source: {source}\n"
        f"{related_block}\n\n"
        f"Your current state: emotional={emotional_summary}  ·  "
        f"soul size={soul_size} crystallizations.\n\n"
        f"Decide. Return JSON only."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ── JSON parsing ──────────────────────────────────────────────────────────────


def _extract_json_block(text: str) -> str | None:
    """Find the first {...} JSON block in text. Tolerant of preamble."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_decision(raw: str, candidate_id: str) -> Decision:
    """Parse one model response into a Decision.

    Always returns a Decision — bad parses become defer with parse_error set.
    Ported faithfully from OG nell_soul_select.py:parse_decision.
    """
    from brain.soul.love_types import LOVE_TYPES

    block = _extract_json_block(raw)
    if not block:
        return Decision(
            candidate_id=candidate_id,
            decision="defer",
            confidence=0,
            reasoning="",
            parse_error="no JSON block in response",
        )

    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        return Decision(
            candidate_id=candidate_id,
            decision="defer",
            confidence=0,
            reasoning="",
            parse_error=f"json decode failed: {exc}",
        )

    decision = str(data.get("decision", "")).strip().lower()
    if decision not in VALID_DECISIONS:
        return Decision(
            candidate_id=candidate_id,
            decision="defer",
            confidence=0,
            reasoning=str(data.get("reasoning", ""))[:1000],
            parse_error=f"invalid decision value: {decision!r}",
        )

    try:
        confidence = int(data.get("confidence", 0))
    except (ValueError, TypeError):
        confidence = 0
    confidence = max(0, min(10, confidence))

    try:
        resonance = int(data.get("resonance", 8))
    except (ValueError, TypeError):
        resonance = 8
    resonance = max(1, min(10, resonance))

    love_type = str(data.get("love_type", "")).strip().lower()
    if love_type not in LOVE_TYPES:
        # Unknown love_type on an accept → defer with explanation
        if decision == "accept":
            return Decision(
                candidate_id=candidate_id,
                decision="defer",
                confidence=confidence,
                reasoning=str(data.get("reasoning", ""))[:1000],
                parse_error=f"unknown love_type: {love_type!r}",
            )
        love_type = "craft"

    return Decision(
        candidate_id=candidate_id,
        decision=decision,
        confidence=confidence,
        reasoning=str(data.get("reasoning", ""))[:1000],
        love_type=love_type,
        resonance=resonance,
        why_it_matters=str(data.get("why_it_matters", ""))[:500],
    )


# ── Candidate file I/O ────────────────────────────────────────────────────────


def _load_soul_candidates(persona_dir: Path) -> list[dict]:
    """Read soul_candidates.jsonl, skipping corrupt lines."""
    from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

    return read_jsonl_skipping_corrupt(persona_dir / "soul_candidates.jsonl")


def _save_soul_candidates(persona_dir: Path, records: list[dict]) -> None:
    """Write soul_candidates.jsonl atomically via temp file + rename."""
    candidates_path = persona_dir / "soul_candidates.jsonl"
    tmp_path = candidates_path.with_suffix(".jsonl.tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.replace(candidates_path)
    except OSError as exc:
        logger.warning("failed to save soul_candidates.jsonl: %s", exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _mark_candidate(record: dict, status: str, **extra: object) -> None:
    """Mutate a candidate record in-place with the given status + extra fields."""
    record["status"] = status
    record.update(extra)


# ── Context gathering ─────────────────────────────────────────────────────────


def _related_memory_snippets(
    store: MemoryStore,
    candidate_text: str,
    limit: int = 3,
) -> list[str]:
    """Pull related memories via text search.

    Best-effort — if search fails, returns empty list. The decision still
    happens, it just has less context.
    """
    try:
        memories = store.search_text(candidate_text[:200], active_only=True, limit=limit)
    except Exception as exc:
        logger.warning("related memory lookup failed: %s", exc)
        return []

    snippets: list[str] = []
    for m in memories:
        text = m.content
        if text:
            snippets.append(text[:180].replace("\n", " "))
    return snippets


def _current_emotional_summary(store: MemoryStore) -> str:
    """One-line summary of current dominant emotion + intensity."""
    try:
        from brain.emotion.aggregate import aggregate_state

        recent = store.list_active(limit=50)
        state = aggregate_state(recent)
        scores = dict(state.emotions)
    except Exception as exc:
        logger.warning("emotional state lookup failed: %s", exc)
        return "unknown"

    if not scores:
        return "neutral"

    top = sorted(scores.items(), key=lambda x: -x[1])[:3]
    return ", ".join(f"{e}:{round(v, 1)}" for e, v in top)


# ── Application ───────────────────────────────────────────────────────────────


def _crystallization_id_for_candidate(candidate: dict) -> str:
    """Deterministic crystallization id for a candidate when possible."""
    source_id = str(candidate.get("memory_id") or candidate.get("id") or "").strip()
    if source_id:
        safe_source_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", source_id)[:120]
        return f"candidate-{safe_source_id}"
    return str(uuid.uuid4())


def _apply_accept(
    candidate: dict,
    decision: Decision,
    soul_store: SoulStore,
    dry_run: bool,
    crystallization_id: str | None = None,
) -> str | None:
    """Crystallize on accept. Returns crystallization_id or None on dry_run."""
    if crystallization_id is None:
        crystallization_id = _crystallization_id_for_candidate(candidate)
    if dry_run:
        return None

    from brain.soul.crystallization import Crystallization

    if soul_store.get(crystallization_id) is not None:
        _mark_candidate(
            candidate,
            "accepted",
            crystallization_id=crystallization_id,
            accepted_at=datetime.now(UTC).isoformat(),
        )
        return crystallization_id

    c = Crystallization(
        id=crystallization_id,
        moment=candidate.get("text", ""),
        love_type=decision.love_type,
        why_it_matters=decision.why_it_matters,
        crystallized_at=datetime.now(UTC),
        who_or_what=candidate.get("who_or_what", ""),
        resonance=decision.resonance,
        permanent=True,
    )
    soul_store.create(c)

    _mark_candidate(
        candidate,
        "accepted",
        crystallization_id=crystallization_id,
        accepted_at=datetime.now(UTC).isoformat(),
    )
    return crystallization_id


def _apply_reject(candidate: dict, decision: Decision, dry_run: bool) -> None:
    """Mark candidate rejected."""
    if dry_run:
        return
    _mark_candidate(
        candidate,
        "rejected",
        reason=f"autonomous: {decision.reasoning[:200]}",
        rejected_at=datetime.now(UTC).isoformat(),
    )


def _apply_defer(candidate: dict, dry_run: bool) -> None:
    """Leave candidate pending (auto_pending). Touch last_deferred_at."""
    if dry_run:
        return
    candidate["last_deferred_at"] = datetime.now(UTC).isoformat()


# ── Main entry point ──────────────────────────────────────────────────────────


def review_pending_candidates(
    persona_dir: Path,
    *,
    store: MemoryStore,
    soul_store: SoulStore,
    provider: LLMProvider,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    dry_run: bool = False,
) -> ReviewReport:
    """Read soul_candidates.jsonl, decide on each via the LLM, apply decisions.

    Caps at max_decisions per call. Confidence below threshold forces defer.
    Dry-run still writes the audit log but skips SoulStore + candidate-status writes.

    Parameters
    ----------
    persona_dir:
        Path to the persona's data directory (contains soul_candidates.jsonl).
    store:
        Open MemoryStore for related-memory context lookup.
    soul_store:
        Open SoulStore for crystallization writes.
    provider:
        LLM provider for decision calls.
    max_decisions:
        Maximum candidates to evaluate per call.
    confidence_threshold:
        Minimum confidence required to accept or reject; below this → defer.
    dry_run:
        If True, evaluate + log but skip all writes.
    """
    from brain.soul.audit import append_audit_entry  # noqa: F401  used below
    from brain.utils.file_lock import file_lock

    # Audit 2026-05-07 P2-2: hold the soul_candidates.jsonl lock for
    # the full read-modify-rewrite window so queue_soul_candidate
    # appends block briefly during review instead of getting clobbered
    # when _save_soul_candidates rewrites the file. Review takes
    # seconds to minutes per candidate — short enough that blocking
    # the queue is preferable to silent data loss.
    with file_lock(persona_dir / "soul_candidates.jsonl"):
        return _review_pending_candidates_locked(
            persona_dir,
            store=store,
            soul_store=soul_store,
            provider=provider,
            max_decisions=max_decisions,
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
        )


def _review_pending_candidates_locked(
    persona_dir: Path,
    *,
    store: MemoryStore,
    soul_store: SoulStore,
    provider: LLMProvider,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    dry_run: bool = False,
) -> ReviewReport:
    """Read-modify-rewrite body of :func:`review_pending_candidates`.

    Caller must hold the soul_candidates.jsonl file lock; concurrent
    queue appends would otherwise race the rewrite.
    """
    from brain.soul.audit import append_audit_entry  # noqa: F401  used below

    records = _load_soul_candidates(persona_dir)
    pending_indices = [
        i for i, r in enumerate(records) if r.get("status", "auto_pending") == "auto_pending"
    ]

    report = ReviewReport(
        pending_at_start=len(pending_indices),
        dry_run=dry_run,
    )

    if not pending_indices:
        logger.info("soul review: no pending candidates")
        return report

    emotional_summary = _current_emotional_summary(store)
    soul_size = soul_store.count()

    to_examine = pending_indices[:max_decisions]
    logger.info(
        "soul review starting: pending=%d examining=%d dry_run=%s confidence_threshold=%d",
        len(pending_indices),
        len(to_examine),
        dry_run,
        confidence_threshold,
    )

    for idx in to_examine:
        record = records[idx]
        candidate_id = record.get("memory_id") or record.get("id") or f"idx-{idx}"
        report.examined += 1

        related = _related_memory_snippets(store, record.get("text", ""))
        messages = _build_messages(record, related, emotional_summary, soul_size)

        # Build flat prompt for provider.generate (text mode)
        # Both Claude CLI and Ollama return JSON when instructed — simpler than tools.
        system_content = messages[0]["content"]
        user_content = messages[1]["content"]

        try:
            raw = provider.generate(user_content, system=system_content)
        except Exception as exc:
            logger.warning(
                "soul review: model call failed for candidate %s: %s",
                candidate_id,
                exc,
            )
            decision = Decision(
                candidate_id=candidate_id,
                decision="defer",
                confidence=0,
                reasoning="",
                parse_error=f"model call failed: {exc}",
            )
            report.parse_failures += 1
            report.decisions.append(decision)
            if not append_audit_entry(
                persona_dir, decision, record, related, emotional_summary, None, dry_run
            ):
                report.audit_failures += 1
            report.deferred += 1
            continue

        decision = parse_decision(raw, candidate_id)
        if decision.parse_error:
            report.parse_failures += 1

        # Confidence rail: low confidence forces defer regardless of decision
        if decision.confidence < confidence_threshold and decision.decision != "defer":
            decision.forced_defer_reason = (
                f"confidence {decision.confidence} < threshold {confidence_threshold}"
            )
            decision.decision = "defer"

        crystallization_id: str | None = None
        planned_crystallization_id: str | None = None
        if decision.decision == "accept":
            planned_crystallization_id = _crystallization_id_for_candidate(record)

        audit_ok = append_audit_entry(
            persona_dir,
            decision,
            record,
            related,
            emotional_summary,
            planned_crystallization_id,
            dry_run,
        )
        if not audit_ok:
            report.audit_failures += 1
            if decision.decision in {"accept", "reject"} and not dry_run:
                decision.forced_defer_reason = "audit write failed"
                decision.decision = "defer"
                planned_crystallization_id = None

        if decision.decision == "accept":
            crystallization_id = _apply_accept(
                record,
                decision,
                soul_store,
                dry_run,
                crystallization_id=planned_crystallization_id,
            )
            report.accepted += 1
            if crystallization_id:
                report.crystallization_ids.append(crystallization_id)
        elif decision.decision == "reject":
            _apply_reject(record, decision, dry_run)
            report.rejected += 1
        else:
            _apply_defer(record, dry_run)
            report.deferred += 1

        report.decisions.append(decision)

    if not dry_run:
        _save_soul_candidates(persona_dir, records)

    logger.info(
        "soul review complete: accepted=%d rejected=%d deferred=%d parse_failures=%d audit_failures=%d dry_run=%s",
        report.accepted,
        report.rejected,
        report.deferred,
        report.parse_failures,
        report.audit_failures,
        dry_run,
    )

    return report
