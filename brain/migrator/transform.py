"""OG → new Memory transformer. Permissive: skips malformed records with a reason."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brain.memory.store import Memory, _coerce_utc

# OG fields with direct first-class mapping in the new schema.
_FIRST_CLASS_OG_FIELDS = frozenset(
    {
        "id",
        "content",
        "memory_type",
        "domain",
        "created_at",
        "last_accessed",
        "tags",
        "importance",
        "emotions",
        "emotion_score",
        "active",
    }
)


@dataclass(frozen=True)
class SkippedMemory:
    """Record of a skipped-during-migration OG memory."""

    id: str
    reason: str  # short code: missing_content, non_numeric_emotion, ...
    field: str  # the specific field that failed, or "" if whole-record
    raw_snippet: str  # truncated human-readable excerpt of the original


def transform_memory(og: dict[str, Any]) -> tuple[Memory | None, SkippedMemory | None]:
    """Transform a single OG memory dict into a new Memory.

    Returns (memory, None) on success, (None, SkippedMemory) on skip.
    Never raises — all malformed input surfaces as a SkippedMemory.
    """
    og_id = og.get("id")
    if not og_id or not isinstance(og_id, str):
        return None, SkippedMemory(
            id=str(og_id or "<unknown>"),
            reason="missing_id",
            field="id",
            raw_snippet=_snippet(og),
        )

    content = og.get("content")
    if not content or not isinstance(content, str):
        return None, SkippedMemory(
            id=og_id,
            reason="missing_content",
            field="content",
            raw_snippet=_snippet(og),
        )

    emotions_raw = og.get("emotions") or {}
    if not isinstance(emotions_raw, dict):
        return None, SkippedMemory(
            id=og_id,
            reason="non_numeric_emotion",
            field="emotions",
            raw_snippet=_snippet(og),
        )
    for v in emotions_raw.values():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None, SkippedMemory(
                id=og_id,
                reason="non_numeric_emotion",
                field="emotions",
                raw_snippet=_snippet(og),
            )
    emotions = {k: float(v) for k, v in emotions_raw.items()}

    created_at_raw = og.get("created_at")
    if not created_at_raw or not isinstance(created_at_raw, str):
        return None, SkippedMemory(
            id=og_id,
            reason="unparseable_created_at",
            field="created_at",
            raw_snippet=_snippet(og),
        )
    try:
        created_at = _coerce_utc(created_at_raw)
    except ValueError:
        return None, SkippedMemory(
            id=og_id,
            reason="unparseable_created_at",
            field="created_at",
            raw_snippet=_snippet(og),
        )

    last_accessed_raw = og.get("last_accessed")
    last_accessed_at = None
    if last_accessed_raw and isinstance(last_accessed_raw, str):
        try:
            last_accessed_at = _coerce_utc(last_accessed_raw)
        except ValueError:
            last_accessed_at = None  # soft-skip this one field; don't skip the memory

    # Prefer OG's stored emotion_score; fall back to sum(emotions.values()).
    score_raw = og.get("emotion_score")
    if isinstance(score_raw, (int, float)) and not isinstance(score_raw, bool):
        score = float(score_raw)
    else:
        score = float(sum(emotions.values()))

    metadata = {k: v for k, v in og.items() if k not in _FIRST_CLASS_OG_FIELDS}

    mem = Memory(
        id=og_id,
        content=content,
        memory_type=str(og.get("memory_type") or "conversation"),
        domain=str(og.get("domain") or "us"),
        created_at=created_at,
        emotions=emotions,
        tags=list(og.get("tags") or []),
        importance=float(og.get("importance") or 0.0),
        score=score,
        last_accessed_at=last_accessed_at,
        active=bool(og.get("active", True)),
        protected=False,
        metadata=metadata,
    )
    return mem, None


def _snippet(og: dict[str, Any], max_len: int = 120) -> str:
    """Human-readable excerpt of the offending OG record, truncated."""
    content = og.get("content") or ""
    if isinstance(content, str):
        return content[:max_len]
    return repr(content)[:max_len]
