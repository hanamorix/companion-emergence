"""Haiku-backed attunement detector.

Subscription-only: routes the LLM call through the Claude CLI subprocess
provider (ClaudeCliProvider), never the Anthropic SDK directly. The
ClaudeCliProvider.generate() path uses --system-prompt-file with a tempfile
to avoid the Windows CreateProcess argv length cap (WinError 206 — see
project gotchas list). No ANTHROPIC_API_KEY references here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from brain.attunement.prompts import build_detector_system_prompt
from brain.attunement.schemas import (
    SCHEMA_VERSION,
    CurrentRead,
    DetectorOutput,
    Evidence,
    PatternCandidate,
)
from brain.attunement.store import BufferTurn
from brain.health.json_io import append_jsonl
from brain.health.parse_failures import record_parse_failure
from brain.utils.llm_output import loads_json_object_ex

log = logging.getLogger(__name__)

_DETECTOR_TIMEOUT_SECONDS = 60


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _decline_output(source_turn_id: str) -> DetectorOutput:
    read = CurrentRead(
        ts=_now_iso(),
        source_turn_id=source_turn_id,
        tone_label="unknown",
        tone_justification="",
        cadence_label="unknown",
        cadence_justification="",
        mood_valence=0.0,
        mood_intensity=0.0,
        predicted_arc_shape="",
        schema_version=SCHEMA_VERSION,
    )
    return DetectorOutput(current_read=read, pattern_candidates=[])


def _format_buffer(buffer_slice: list[BufferTurn]) -> str:
    return "\n".join(f"[turn id={t.id}] {t.content}" for t in buffer_slice)


def _build_user_message(
    buffer_slice: list[BufferTurn],
    reply_text: str,
    *,
    companion_name: str = "",
) -> str:
    if companion_name:
        reply_label = f"{companion_name.upper()}'S LATEST REPLY"
    else:
        reply_label = "LATEST REPLY"
    return (
        "USER'S RECENT TURNS:\n"
        f"{_format_buffer(buffer_slice)}\n\n"
        f"{reply_label} (for context only, do not extract patterns from it):\n"
        f"{reply_text}\n\n"
        "Return JSON only. No prose."
    )


def _call_haiku(
    system_prompt: str,
    user_message: str,
    *,
    persona_dir: Path | None = None,
) -> str:
    """Call Haiku via the Claude CLI. Returns raw stdout (expected JSON).

    Routes through ClaudeCliProvider.generate() which uses
    --system-prompt-file with a tempfile to avoid the Windows CreateProcess
    argv length cap (WinError 206 — see project gotchas list).

    Returns empty string on any failure so callers can decline cleanly.
    """
    from brain.bridge.model_tier import TIER_ATTUNEMENT_DETECTOR, resolve_tier_model
    from brain.bridge.provider import ClaudeCliProvider

    provider = ClaudeCliProvider(
        model=resolve_tier_model(persona_dir, TIER_ATTUNEMENT_DETECTOR)[0],
        timeout_seconds=_DETECTOR_TIMEOUT_SECONDS,
    )
    pd: Path | None = Path(persona_dir) if isinstance(persona_dir, str) else persona_dir
    try:
        return provider.generate(user_message, system=system_prompt, persona_dir=pd)
    except Exception as exc:  # noqa: BLE001
        log.warning("attunement detector: claude CLI call failed: %s", exc)
        if pd is not None:
            try:
                import traceback as _tb

                entry = {
                    "ts": _now_iso(),
                    "error": str(exc),
                    "kind": "haiku_call_failed",
                    "traceback": _tb.format_exc(),
                }
                append_jsonl(pd / "attunement_errors.jsonl", entry, on_error="log", nonfinite="null")
            except Exception:  # noqa: BLE001
                pass  # best-effort — never mask the graceful decline
        return ""


_DETECTOR_KEYS = frozenset({"current_read", "pattern_candidates", "addressed_pattern_ids"})


def _parse_output(raw: str, source_turn_id: str, persona_dir: Path | None = None) -> DetectorOutput:
    """Parse Haiku's JSON output into DetectorOutput. Drops invalid candidates.
    A reply that can't be read is kept in the parse-failure log (#364)."""
    # One tolerance rule for every reply that should be one JSON object (#479): the whole reply, a fenced block, or an object with
    # prose before/after it (ClaudeCliProvider returns raw text, but some models wrap or chatter).
    try:
        payload: dict[str, Any]
        payload, in_prose = loads_json_object_ex(raw)
        if in_prose and not (set(payload) & _DETECTOR_KEYS):
            # An object found in a sentence is only the answer if it looks like one; the whole reply is always taken.
            raise ValueError("an object in the reply has none of the detector's keys")
    except ValueError as exc:  # JSONDecodeError is a ValueError
        record_parse_failure(persona_dir, site="attunement", raw=raw, error=f"non-JSON: {exc}")
        return _decline_output(source_turn_id)

    try:
        cr = payload.get("current_read", {})
        current_read = CurrentRead(
            ts=_now_iso(),
            source_turn_id=source_turn_id,
            tone_label=str(cr.get("tone_label", "unknown")),
            tone_justification=str(cr.get("tone_justification", "")),
            cadence_label=str(cr.get("cadence_label", "unknown")),
            cadence_justification=str(cr.get("cadence_justification", "")),
            mood_valence=float(cr.get("mood_valence", 0.0)),
            mood_intensity=float(cr.get("mood_intensity", 0.0)),
            predicted_arc_shape=str(cr.get("predicted_arc_shape", "")),
            schema_version=SCHEMA_VERSION,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        record_parse_failure(persona_dir, site="attunement", raw=raw, error=f"current_read shape: {exc}")
        return _decline_output(source_turn_id)

    candidates: list[PatternCandidate] = []
    rejections: list[str] = []
    for raw_cand in payload.get("pattern_candidates", []):
        try:
            evidence = [
                Evidence(quote=str(e.get("quote", "")), turn_id=str(e.get("turn_id", "")))
                for e in raw_cand.get("evidence", [])
            ]
            candidates.append(
                PatternCandidate(
                    category=str(raw_cand.get("category", "")),
                    canonical_key=str(raw_cand.get("canonical_key", "")),
                    description=str(raw_cand.get("description", "")),
                    evidence=evidence,
                )
            )
        except (TypeError, ValueError) as exc:
            rejections.append(f"invalid candidate: {exc}")

    return DetectorOutput(
        current_read=current_read,
        pattern_candidates=candidates,
        addressed_pattern_ids=[str(i) for i in payload.get("addressed_pattern_ids", [])],
        rejection_notes=rejections,
    )


def should_run_detector(
    buffer_slice: list[BufferTurn],
    user_message: str,
    reply_text: str,
    *,
    min_words: int = 5,
) -> bool:
    """Decide whether this turn warrants an attunement pass-2 run.

    Skip rules (any one → skip):
    - Buffer slice is empty
    - User message is empty / whitespace-only (tool-only turn)
    - User message has fewer than min_words tokens

    The `reply_text` parameter is accepted for symmetry with the pass-2
    spawn signature (Task 12) but is not currently a gating criterion —
    the detector reads the buffer slice, not Nell's reply.
    """
    if not buffer_slice:
        return False
    if not user_message.strip():
        return False
    if len(user_message.split()) < min_words:
        return False
    return True


def run_detector(
    buffer_slice: list[BufferTurn],
    reply_text: str,
    *,
    only_categories: frozenset[str] | None = None,
    companion_name: str = "",
    user_name: str = "",
    user_pronouns: object = None,
    persona_dir: Path | None = None,
) -> DetectorOutput:
    """Run the Haiku attunement detector against a buffer slice + reply.

    When *only_categories* is provided, appends a restriction instruction to
    the system prompt so the model only emits candidates for those categories.
    Used by the supplementary backfill pass to avoid double-counting tone/cadence
    patterns that are already learned.

    Returns a decline output (unknown labels, empty candidates) when:
    - buffer_slice is empty
    - the CLI call fails (non-zero exit, timeout)
    - the response is malformed JSON
    - current_read shape is invalid

    Invalid pattern candidates are dropped into rejection_notes; valid
    candidates still flow through. Hallucinated quotes are caught downstream
    by validate_grounded at the store layer (defence-in-depth).
    """
    if not buffer_slice:
        return _decline_output(source_turn_id="")

    source_turn_id = buffer_slice[-1].id
    system_prompt = build_detector_system_prompt(
        only_categories=only_categories,
        companion_name=companion_name,
        user_name=user_name,
        user_pronouns=user_pronouns,
    )
    user_message = _build_user_message(buffer_slice, reply_text, companion_name=companion_name)

    raw = _call_haiku(system_prompt, user_message, persona_dir=persona_dir)
    if not raw:
        return _decline_output(source_turn_id)

    return _parse_output(raw, source_turn_id, persona_dir)
