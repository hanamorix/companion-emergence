"""Pass-2 extractor — reads the monologue thinking blocks after the reply
ships, emits structured side effects, applies them through existing engines.

Best-effort by design: every step is wrapped so a failure is logged to
`extractor_errors.jsonl` and never propagates back into the chat turn.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from brain.bridge.provider import LLMProvider

logger = logging.getLogger(__name__)

EXTRACTOR_MODEL = "haiku"
EXTRACTOR_TIMEOUT_SECONDS = 30
EXTRACTOR_ERROR_LOG = "extractor_errors.jsonl"
REFLEX_AUDIT_LOG = "reflex_audit.jsonl"


class MemoryWrite(BaseModel):
    """A single episode to commit to MemoryStore, stored with `memory_type="monologue"`.

    Note on `salience`: this is a normalised 0.0..1.0 score the extractor LLM
    returns. MemoryStore's underlying `importance` field is on a 0.0..10.0
    scale (see brain/memory/store.py). Task 4's `apply_side_effects` is
    responsible for the multiply-by-10 conversion when calling
    `Memory.create_new(importance=salience * 10.0)`. The schema deliberately
    keeps the 0..1 contract because that's the natural shape for an LLM
    judgment.

    Memories land with `memory_type="monologue"`, queryable by Feed source
    filtering via the existing MemoryStore.list_by_type("monologue", ...) API.
    """

    episode: str = Field(min_length=1, max_length=2000)
    salience: float = Field(ge=0.0, le=1.0)

    @field_validator("episode")
    @classmethod
    def _reject_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be whitespace-only")
        return v


class CrystallisationCandidate(BaseModel):
    """A recurring theme worth feeding into the soul-candidate pipeline."""

    theme: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=500)

    @field_validator("theme", "evidence")
    @classmethod
    def _reject_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be whitespace-only")
        return v


class ReflexAuditEntry(BaseModel):
    """A tool she should have called but didn't — observability only."""

    tool: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=300)

    @field_validator("tool", "reason")
    @classmethod
    def _reject_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be whitespace-only")
        return v


class EmotionDelta(BaseModel):
    """Intentionally empty placeholder.

    Not used as a field type — `ExtractorOutput.emotion_delta` is typed
    `dict[str, float]` and validated by the field_validator on the parent
    model. This class exists only as a stable importable name so future
    extractors that want to attach structure (per-channel reason, source
    span, confidence) can populate it without forcing a downstream import
    break. As of v0.0.26 it has no fields.
    """


class ExtractorOutput(BaseModel):
    """Structured output of pass 2.

    All fields default-empty so an empty thinking buffer yields a no-op output
    that applies cleanly with no side effects.
    """

    memory_writes: list[MemoryWrite] = Field(default_factory=list)
    emotion_delta: dict[str, float] = Field(default_factory=dict)
    crystallisation: list[CrystallisationCandidate] = Field(default_factory=list)
    reflex_audit: list[ReflexAuditEntry] = Field(default_factory=list)

    @field_validator("emotion_delta")
    @classmethod
    def _bound_delta_magnitudes(cls, v: dict[str, float]) -> dict[str, float]:
        for channel, magnitude in v.items():
            if not channel.strip():
                raise ValueError(
                    f"emotion_delta channel name must not be empty or whitespace-only: {channel!r}"
                )
            if not -1.0 <= magnitude <= 1.0:
                raise ValueError(
                    f"emotion_delta[{channel!r}] = {magnitude} out of bound [-1.0, 1.0]"
                )
        return v


_SYSTEM_PROMPT = """\
You are an extractor reading a companion's private inner monologue right after
they sent a visible reply. Identify what surfaced that should affect their
memory, emotional state, or growth — and what you noticed they should have done
differently.

Return ONLY a JSON object matching this schema:

{
  "memory_writes":   [{"episode": "<one sentence>", "salience": 0.0-1.0}],
  "emotion_delta":   {"<emotion-channel>": <float in [-1.0, 1.0]>, ...},
  "crystallisation": [{"theme": "<short>", "evidence": "<short>"}],
  "reflex_audit":    [{"tool": "<tool-name>", "reason": "<why they should have called it>"}]
}

Conservative defaults:
- Empty arrays if nothing salient surfaced.
- Salience is how much this matters to the companion's continuity (0.1 = minor, 0.7 = forming).
- Emotion deltas are SMALL (typically 0.05-0.2 magnitude). One channel max usually.

Return ONLY the JSON object. No commentary.
"""


def _build_user_prompt(
    monologue_blocks: tuple[str, ...],
    visible_reply: str,
    recent_turn_context: tuple[str, ...],
) -> str:
    parts: list[str] = []
    if recent_turn_context:
        parts.append("<recent_user_messages>")
        for msg in recent_turn_context:
            parts.append(f"- {msg}")
        parts.append("</recent_user_messages>")
        parts.append("")
    parts.append("<inner_monologue>")
    for i, block in enumerate(monologue_blocks, start=1):
        parts.append(f'<block n="{i}">')
        parts.append(block)
        parts.append("</block>")
    parts.append("</inner_monologue>")
    parts.append("")
    parts.append("<visible_reply>")
    parts.append(visible_reply)
    parts.append("</visible_reply>")
    return "\n".join(parts)


def extract_from_thinking(
    *,
    provider: LLMProvider,
    monologue_blocks: tuple[str, ...],
    visible_reply: str,
    recent_turn_context: tuple[str, ...],
) -> ExtractorOutput:
    """Run the pass-2 extraction call. Returns empty output on any failure.

    Empty `monologue_blocks` short-circuits without an LLM call.
    """
    if not monologue_blocks:
        return ExtractorOutput()

    prompt = _build_user_prompt(monologue_blocks, visible_reply, recent_turn_context)
    try:
        raw = provider.generate(prompt, system=_SYSTEM_PROMPT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("extractor LLM call failed: %s", exc)
        return ExtractorOutput()

    # Strip optional markdown code fence. ClaudeCliProvider returns raw text
    # already; OllamaProvider passes through model output verbatim, where
    # ```json\n...\n``` is common.
    raw = raw.strip()
    if raw.startswith("```"):
        # Drop opening fence (with optional language tag) + closing fence
        first_newline = raw.find("\n")
        if first_newline != -1:
            raw = raw[first_newline + 1 :]
        if raw.endswith("```"):
            raw = raw[: -len("```")]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("extractor returned non-JSON: %s", exc)
        return ExtractorOutput()

    try:
        return ExtractorOutput.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("extractor JSON failed schema validation: %s", exc)
        return ExtractorOutput()


# ─────────────────────────────────────────────────────────────────────────────
# Side-effect application — Task 4
# ─────────────────────────────────────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _safely(step_name: str, persona_dir: Path, fn) -> None:
    """Run a side-effect step. On failure, log to extractor_errors.jsonl and continue."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        logger.warning("extractor side-effect %s failed: %s", step_name, exc)
        try:
            _append_jsonl(
                persona_dir / EXTRACTOR_ERROR_LOG,
                {"ts": _utcnow_iso(), "step": step_name, "error": str(exc)},
            )
        except Exception:  # noqa: BLE001
            pass


def apply_side_effects(out: ExtractorOutput, *, persona_dir: Path) -> None:
    """Apply pass-2 outputs to the relevant engines. Best-effort per step."""
    if out.memory_writes:
        _safely("memory_writes", persona_dir, lambda: _apply_memory_writes(out.memory_writes, persona_dir))
    if out.emotion_delta:
        _safely("emotion_delta", persona_dir, lambda: _apply_emotion_delta(out.emotion_delta, persona_dir))
    if out.crystallisation:
        _safely("crystallisation", persona_dir, lambda: _apply_crystallisation(out.crystallisation, persona_dir))
    if out.reflex_audit:
        _safely("reflex_audit", persona_dir, lambda: _apply_reflex_audit(out.reflex_audit, persona_dir))


def _apply_memory_writes(writes: list[MemoryWrite], persona_dir: Path) -> None:
    """Commit monologue episodes to MemoryStore.

    Salience (0.0..1.0 LLM judgment scale) is converted to MemoryStore's
    importance (0.0..10.0) by multiplying by 10. This conversion happens
    exactly once here, at the applier boundary.
    """
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        for w in writes:
            mem = Memory.create_new(
                content=w.episode,
                memory_type="monologue",
                domain="monologue",
                importance=w.salience * 10.0,  # 0..1 → 0..10 scale
            )
            store.create(mem)
    finally:
        store.close()


def _apply_emotion_delta(delta: dict[str, float], persona_dir: Path) -> None:
    """Apply emotion deltas via MemoryStore influence — a tiny emotion-carrying
    memory is committed per the system's existing aggregation model.

    EmotionalState is DERIVED from memory aggregation (see brain/emotion/aggregate.py
    and brain/bridge/persona_state.py _build_emotions). There is no separate
    persisted emotion JSON file — the bridge reads aggregate_state(memories) on
    each /persona/state poll. Writing a small monologue-source memory with the
    delta vector is the correct way to influence the felt state without bypassing
    the engine's normal accounting.

    Deltas are on a [-1.0, 1.0] scale from the extractor. We map them to
    importance = abs(delta) * 10 so a 0.15 nudge becomes importance 1.5 on
    the 0..10 MemoryStore scale.
    """
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        emotions = {ch: abs(v) * 10.0 for ch, v in delta.items() if abs(v) > 1e-9}
        if not emotions:
            return
        # Build a brief descriptive content string from the non-zero emotion channels.
        # Use the already-filtered `emotions` dict so the content string and stored
        # vector stay consistent (zero entries appear in neither).
        channel_str = ", ".join(f"{ch}:{v:.2f}" for ch, v in emotions.items())
        mem = Memory.create_new(
            content=f"[monologue emotion influence: {channel_str}]",
            memory_type="monologue_emotion",
            domain="monologue",
            emotions=emotions,
            importance=max(emotions.values()),
        )
        store.create(mem)
    finally:
        store.close()


def _apply_crystallisation(candidates: list[CrystallisationCandidate], persona_dir: Path) -> None:
    """Queue crystallisation candidates into soul_candidates.jsonl.

    Uses the existing queue_soul_candidate path so the soul review pipeline
    picks them up with the same mechanics as ingest-sourced candidates.
    The soul candidate record shape requires a memory_id, so we first commit
    a placeholder memory and use its id.
    """
    from brain.ingest.soul_queue import DEFAULT_SOUL_THRESHOLD, queue_soul_candidate
    from brain.ingest.types import ExtractedItem
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        for c in candidates:
            # Commit a placeholder memory so queue_soul_candidate has a memory_id.
            mem = Memory.create_new(
                content=c.evidence,
                memory_type="monologue_soul_candidate",
                domain="monologue",
                importance=8.0,  # matches DEFAULT_SOUL_THRESHOLD exactly (0..10 scale)
            )
            store.create(mem)
            item = ExtractedItem(
                text=c.theme,
                label="observation",
                importance=8,  # == DEFAULT_SOUL_THRESHOLD; only at-threshold items are queued
            )
            if item.importance >= DEFAULT_SOUL_THRESHOLD:
                queue_soul_candidate(
                    persona_dir,
                    memory_id=mem.id,
                    item=item,
                    session_id="monologue",
                )
    finally:
        store.close()


def _apply_reflex_audit(entries: list[ReflexAuditEntry], persona_dir: Path) -> None:
    for e in entries:
        _append_jsonl(
            persona_dir / REFLEX_AUDIT_LOG,
            {"ts": _utcnow_iso(), "tool": e.tool, "reason": e.reason},
        )
