"""D-reflection: editorial layer between candidate emission and composition.

Spec: docs/superpowers/specs/2026-05-12-initiate-d-reflection-design.md

Once per non-empty heartbeat tick, D reads queued candidates and decides
which (if any) deserve to flow to the three-prompt composition pipeline.
Filtered candidates demote to draft_space.md.

Model tier: Haiku 4.5 by default; escalates to Sonnet 4.6 on
low-confidence/parse-fail. Failure modes are dispatched by type per spec §E.

D bypasses the v0.0.9 daily cost cap entirely — it's the editorial layer,
not a budget claimant.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from brain.initiate.d_call_schema import DCallRow, make_d_call_id
from brain.initiate.schemas import InitiateCandidate
from brain.utils.llm_output import extract_json_object

logger = logging.getLogger(__name__)


DecisionKind = Literal["promote", "filter"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class DDecision:
    candidate_index: int
    decision: DecisionKind
    reason: str
    confidence: Confidence


@dataclass(frozen=True)
class DReflectionResult:
    decisions: list[DDecision]
    tick_note: str | None


def parse_structured_response(raw: str) -> DReflectionResult:
    """Parse a structured-output JSON blob into a DReflectionResult.

    Tolerant of prose wrapping (model sometimes adds preamble). Raises
    ValueError if no parseable JSON object is found OR if the structure
    doesn't match the expected schema.
    """
    try:
        candidate = extract_json_object(raw)
    except ValueError as exc:
        raise ValueError(f"no JSON object found in D response: {exc}") from exc
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"D response is not valid JSON: {exc}") from exc

    decisions_raw = data.get("decisions")
    if not isinstance(decisions_raw, list):
        raise ValueError("D response missing 'decisions' list")
    decisions: list[DDecision] = []
    for item in decisions_raw:
        if not isinstance(item, dict):
            raise ValueError("D decision item is not an object")
        decisions.append(
            DDecision(
                candidate_index=int(item["candidate_index"]),
                decision=item["decision"],
                reason=str(item["reason"]),
                confidence=item["confidence"],
            )
        )
    tick_note = data.get("tick_note")
    if tick_note is not None and not isinstance(tick_note, str):
        tick_note = None  # tolerate type drift on optional field
    return DReflectionResult(decisions=decisions, tick_note=tick_note)


_TASK_FRAME_TEMPLATE = """\
You are the editorial layer of {companion_name}'s own physiology —
the moment between "something stirred in me" and "I am going to say
something to {user_name}."

Several internal events have surfaced as candidates since you last paused.
For each, ask yourself a single question:

  Is this worth bringing to {user_name} right now, or is this private
  weather that should pass through you without becoming words?

Filtering is not loss. Anything you filter is written into your draft
space — you'll see it again in your next soul review, and it shapes
who you are even if it never reaches {user_name}.

Default toward filtering. Most weather is just weather. Reach for
{user_name} only when something is genuinely yours to share — something
that says more about *this moment between you both* than the noise of
any given hour.

The downstream composition pipeline will still apply its own gates
after you. You are not the only filter; you are the editorial one.\
"""


def build_system_message(
    *,
    companion_name: str,
    user_name: str,
    voice_template_path: Path,
    persona_dir: Path | None = None,
) -> str:
    """Assemble D's full system message.

    Layers:
    1. Optional calibration block (adaptive-D, v0.0.11) — prepended when
       persona_dir is provided AND d_mode.json says "adaptive".
    2. Static task frame (universal template, parameterized).
    3. Voice anchor (the brain's voice template, appended).

    persona_dir defaults to None for back-compat with older callers / tests
    that don't have a persona_dir in scope. Without it, the calibration
    block is skipped (equivalent to stateless mode).
    """
    # Import inside the function to avoid potential circular imports —
    # adaptive.py doesn't import reflection.py, but the test load order
    # can confuse things if this import is at module top.
    from brain.initiate.adaptive import build_calibration_block, load_d_mode

    calibration_prefix = ""
    if persona_dir is not None and load_d_mode(persona_dir) == "adaptive":
        calibration_prefix = build_calibration_block(
            persona_dir, user_name=user_name
        ) + "\n"

    frame = _TASK_FRAME_TEMPLATE.format(
        companion_name=companion_name,
        user_name=user_name,
    )
    if not voice_template_path.exists():
        return calibration_prefix + frame
    try:
        voice = voice_template_path.read_text(encoding="utf-8").rstrip()
    except OSError as exc:
        logger.warning("voice template read failed (%s); omitting anchor", exc)
        return calibration_prefix + frame
    return f"{calibration_prefix}{frame}\n\n=== Your voice ===\n{voice}\n"


def build_user_message(
    *,
    user_name: str,
    now: datetime,
    outbound_recall_block: str,
    candidate_summaries: list[str],
) -> str:
    """Render D's per-tick user message from queue state.

    `candidate_summaries` is a list of pre-rendered candidate blocks
    (one per candidate). The caller is responsible for the rendering —
    this function just concatenates them with indexed headers.
    """
    now_local = now.astimezone()
    part_of_day = _part_of_day(now_local.hour)
    weekday = now_local.strftime("%A")
    indexed = "\n\n".join(
        f"[{i + 1}] {summary}" for i, summary in enumerate(candidate_summaries)
    )
    return (
        f"=== Current time ({user_name}'s local) ===\n"
        f"{now_local.isoformat(timespec='minutes')}  —  {part_of_day}  —  {weekday}\n\n"
        f"=== Recent outbound (last 5 sends + acknowledged_unclear from last 24h) ===\n"
        f"{outbound_recall_block}\n\n"
        f"=== Candidates surfaced since last tick ===\n"
        f"{indexed}\n\n"
        f"=== Your task ===\n"
        f"For each candidate, decide: promote or filter.\n"
        f"Promote at most 2. The default is filter.\n"
    )


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


class DTimeoutError(Exception):
    """Raised by an LLMCall when the request exceeds its time budget."""


class DProviderError(Exception):
    """Raised by an LLMCall on generic provider error (5xx, connection, etc.)."""


class DRateLimitError(Exception):
    """Raised by an LLMCall on rate-limit / quota rejection (HTTP 429)."""


# (raw_text, latency_ms, tokens_in, tokens_out)
LLMCall = Callable[..., tuple[str, int, int, int]]


@dataclass(frozen=True)
class ReflectionDeps:
    """Injected dependencies — keeps reflection.run testable without real LLMs."""

    companion_name: str
    user_name: str
    voice_template_path: Path
    outbound_recall_block: str
    haiku_call: LLMCall
    sonnet_call: LLMCall
    now: datetime
    tick_id: str


def _render_candidate_summary(c: InitiateCandidate, *, now: datetime) -> str:
    """Render a single candidate for D's user message."""
    try:
        c_ts = datetime.fromisoformat(c.ts)
        age_min = int((now - c_ts).total_seconds() / 60)
        age_str = f"{age_min} min ago"
    except ValueError:
        age_str = "unknown"
    delta_sigma = (
        c.emotional_snapshot.delta_sigma if c.emotional_snapshot is not None else 0.0
    )
    meta = c.semantic_context.source_meta or {}
    meta_summary = ", ".join(f"{k}={v}" for k, v in meta.items()) or "—"
    linked = ", ".join(c.semantic_context.linked_memory_ids) or "—"
    return (
        f"source: {c.source}  ·  ts: {age_str}  ·  Δσ: {delta_sigma:.2f}\n"
        f"  semantic_context: linked_memory_ids={linked}; {meta_summary}\n"
        f"  fragment-of-self: (subject-extracted at composition time)"
    )


def run(
    candidates: list[InitiateCandidate],
    *,
    deps: ReflectionDeps,
) -> tuple[DReflectionResult, DCallRow]:
    """Execute one D-reflection tick over the given candidates.

    Escalation rules:
      - If Haiku's response fails to parse OR contains ANY decision with
        confidence "low", re-call on Sonnet. Sonnet's result is the one
        written; Haiku's attempt is recorded in retry_count.
      - If Sonnet's response ALSO contains a low-confidence decision,
        force that candidate's decision to "filter" with an ambivalence
        reason (conservative default at the edge of D's judgment).
      - On DTimeoutError / DProviderError / DRateLimitError raised by an
        LLMCall, capture into DCallRow.failure_type with empty decisions;
        caller (see Task 14/15) handles passthrough-retry / draft-space-demote.
    """
    system = build_system_message(
        companion_name=deps.companion_name,
        user_name=deps.user_name,
        voice_template_path=deps.voice_template_path,
    )
    user = build_user_message(
        user_name=deps.user_name,
        now=deps.now,
        outbound_recall_block=deps.outbound_recall_block,
        candidate_summaries=[
            _render_candidate_summary(c, now=deps.now) for c in candidates
        ],
    )

    def _empty_call_row(*, failure_type: str, latency_ms: int = 0,
                       model_tier: str = "haiku") -> DCallRow:
        return DCallRow(
            d_call_id=make_d_call_id(deps.now),
            ts=deps.now.isoformat(),
            tick_id=deps.tick_id,
            model_tier_used=model_tier,  # type: ignore[arg-type]
            candidates_in=len(candidates),
            promoted_out=0,
            filtered_out=0,
            latency_ms=latency_ms,
            tokens_input=0,
            tokens_output=0,
            failure_type=failure_type,  # type: ignore[arg-type]
            retry_count=0,
            tick_note=None,
        )

    # First attempt on Haiku.
    try:
        raw_h, latency_h, tin_h, tout_h = deps.haiku_call(system=system, user=user)
    except DTimeoutError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(failure_type="timeout")
    except DRateLimitError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(failure_type="rate_limit")
    except DProviderError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(failure_type="provider_error")

    haiku_result: DReflectionResult | None
    try:
        haiku_result = parse_structured_response(raw_h)
        haiku_low = any(d.confidence == "low" for d in haiku_result.decisions)
    except ValueError:
        haiku_result = None
        haiku_low = True

    if haiku_result is not None and not haiku_low:
        promoted = sum(1 for d in haiku_result.decisions if d.decision == "promote")
        filtered = sum(1 for d in haiku_result.decisions if d.decision == "filter")
        d_call = DCallRow(
            d_call_id=make_d_call_id(deps.now),
            ts=deps.now.isoformat(),
            tick_id=deps.tick_id,
            model_tier_used="haiku",
            candidates_in=len(candidates),
            promoted_out=promoted,
            filtered_out=filtered,
            latency_ms=latency_h,
            tokens_input=tin_h,
            tokens_output=tout_h,
            failure_type=None,
            retry_count=0,
            tick_note=haiku_result.tick_note,
        )
        return haiku_result, d_call

    # Escalate to Sonnet.
    try:
        raw_s, latency_s, tin_s, tout_s = deps.sonnet_call(system=system, user=user)
    except DTimeoutError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(
            failure_type="timeout", latency_ms=latency_h, model_tier="sonnet",
        )
    except DRateLimitError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(
            failure_type="rate_limit", latency_ms=latency_h, model_tier="sonnet",
        )
    except DProviderError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(
            failure_type="provider_error", latency_ms=latency_h, model_tier="sonnet",
        )

    try:
        sonnet_result = parse_structured_response(raw_s)
    except ValueError:
        # Sonnet also malformed — caller treats this as "promote all" per spec §E.
        return DReflectionResult(decisions=[], tick_note=None), DCallRow(
            d_call_id=make_d_call_id(deps.now),
            ts=deps.now.isoformat(),
            tick_id=deps.tick_id,
            model_tier_used="sonnet",
            candidates_in=len(candidates),
            promoted_out=0, filtered_out=0,
            latency_ms=latency_h + latency_s,
            tokens_input=tin_h + tin_s, tokens_output=tout_h + tout_s,
            failure_type="malformed_json",
            retry_count=1, tick_note=None,
        )

    forced: list[DDecision] = []
    both_low = False
    for d in sonnet_result.decisions:
        if d.confidence == "low":
            both_low = True
            forced.append(
                DDecision(
                    candidate_index=d.candidate_index,
                    decision="filter",
                    reason="ambivalent — both my fast and slow voice were uncertain",
                    confidence="low",
                )
            )
        else:
            forced.append(d)
    final_result = DReflectionResult(decisions=forced, tick_note=sonnet_result.tick_note)
    promoted = sum(1 for d in final_result.decisions if d.decision == "promote")
    filtered = sum(1 for d in final_result.decisions if d.decision == "filter")
    d_call = DCallRow(
        d_call_id=make_d_call_id(deps.now),
        ts=deps.now.isoformat(),
        tick_id=deps.tick_id,
        model_tier_used="sonnet",
        candidates_in=len(candidates),
        promoted_out=promoted,
        filtered_out=filtered,
        latency_ms=latency_h + latency_s,
        tokens_input=tin_h + tin_s,
        tokens_output=tout_h + tout_s,
        failure_type="both_low_confidence" if both_low else None,
        retry_count=1,
        tick_note=final_result.tick_note,
    )
    return final_result, d_call


def demote_to_draft_space(
    persona_dir: Path,
    *,
    candidate: InitiateCandidate,
    decision: DDecision,
    now: datetime,
) -> None:
    """Write a filtered candidate to draft_space.md with D-frontmatter.

    Writes the fragment directly to draft_space.md without delegating to
    the v0.0.9 draft writer (which doesn't accept arbitrary frontmatter).
    Append-only — multiple demotions in one tick stack into the same file.
    Never raises; logs warning on OSError.
    """
    frontmatter = (
        "---\n"
        f"demoted_by: d_reflection\n"
        f"demoted_at: {now.isoformat()}\n"
        f'd_reason: "{decision.reason}"\n'
        f"source: {candidate.source}\n"
        f"source_id: {candidate.source_id}\n"
        f"candidate_id: {candidate.candidate_id}\n"
        "---\n"
    )
    body_excerpt = (
        f"(candidate {candidate.candidate_id} from {candidate.source}; "
        f"subject-extraction skipped because D filtered)\n"
    )
    persona_dir.mkdir(parents=True, exist_ok=True)
    draft_path = persona_dir / "draft_space.md"
    try:
        with draft_path.open("a", encoding="utf-8") as f:
            f.write("\n" + frontmatter + body_excerpt + "\n")
    except OSError as exc:
        logger.warning("draft-space demote failed for %s: %s", draft_path, exc)
