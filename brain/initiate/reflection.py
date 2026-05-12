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
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

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


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_LOOSE = re.compile(r"(\{.*\})", re.DOTALL)


def parse_structured_response(raw: str) -> DReflectionResult:
    """Parse a structured-output JSON blob into a DReflectionResult.

    Tolerant of prose wrapping (model sometimes adds preamble). Raises
    ValueError if no parseable JSON object is found OR if the structure
    doesn't match the expected schema.
    """
    fenced = _JSON_FENCE.search(raw)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        loose = _JSON_LOOSE.search(raw)
        if loose is None:
            raise ValueError("no JSON object found in D response")
        candidate = loose.group(1)
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
) -> str:
    """Assemble D's full system message: static task frame + voice anchor.

    The static frame is parameterized with companion_name / user_name.
    The voice anchor is the contents of voice_template_path, appended
    under a `=== Your voice ===` header. If the file is missing or
    unreadable, the anchor is omitted (the task frame still works).
    """
    frame = _TASK_FRAME_TEMPLATE.format(
        companion_name=companion_name,
        user_name=user_name,
    )
    if not voice_template_path.exists():
        return frame
    try:
        voice = voice_template_path.read_text(encoding="utf-8").rstrip()
    except OSError as exc:
        logger.warning("voice template read failed (%s); omitting anchor", exc)
        return frame
    return f"{frame}\n\n=== Your voice ===\n{voice}\n"


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
