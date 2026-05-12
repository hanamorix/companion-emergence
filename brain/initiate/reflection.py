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
