"""OG → new Memory transformer. Permissive: skips malformed records with a reason."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from brain.memory.store import Memory, _coerce_utc, clamp_importance
from brain.tools.impls._common import _calc_importance_from_emotions

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
    """Record of a skipped-during-migration OG memory.

    Reason codes set by transform_memory:
        missing_id, missing_content, non_numeric_emotion, unparseable_created_at.

    Reason codes set externally by run_migrate (brain.migrator.cli):
        duplicate_id.
    """

    id: str
    reason: str
    field: str  # the specific field that failed, or "" if whole-record
    raw_snippet: str  # truncated human-readable excerpt of the original


def transform_memory(
    og: dict[str, Any], checked: Callable[[Memory], Memory] | None = None
) -> tuple[Memory | None, SkippedMemory | None]:
    """Transform a single OG memory dict into a new Memory.

    Returns (memory, None) on success, (None, SkippedMemory) on skip.
    Never raises on malformed input: all of it surfaces as a SkippedMemory.

    `checked` is the store's check on a memory's feelings (`MemoryStore.checked`), or None for a caller that keeps the names it
    brings. When given, it runs once on the memory built here (a skipped row never reaches it), the returned memory carries the
    checked feelings, and a legacy row with no numeric importance gets its importance from those kept feelings. An exception the
    check raises (it can ask the bridge or take the vocabulary file's lock) is not caught here.
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

    # importance: only coerce real numbers. Strings like "high" would crash
    # float(); lists/dicts too. Protects the never-raises contract.
    # P3 retention rework, Change 1 (OG-0 backfill): a missing/malformed OG
    # importance no longer hard-defaults to 0.0 (which would mass-evict
    # migrated history at the new importance-driven decay baseline). Backfill
    # from emotions when present (reusing the OG add_memory auto-importance
    # bucket), else a moderate flat default — migrated history should not
    # start at the bottom of the scale. A numeric-but-oversized OG value
    # (e.g. 50.0) is clamped, same as every other create_new/from_stored_row
    # boundary (finding #4).
    #
    # #803 step 13: a legacy importance that is derived (no numeric value in the row) comes from the feelings the store's check KEEPS,
    # not from the raw ones, so a feeling the check drops, or two spellings it averages, leave no trace in it. The check is handed in
    # (`checked`, the store's own) and runs here, on the memory built below, so the importance is decided where it is made.
    importance_raw = og.get("importance")
    importance_given = isinstance(importance_raw, (int, float)) and not isinstance(importance_raw, bool)
    if importance_given:
        importance = clamp_importance(importance_raw)
    else:
        importance = _derived_importance(emotions)

    # tags: only accept a real list. list("mytag") would explode a string
    # into ['m','y','t','a','g']; list({"a":1}) would pull dict keys. Both
    # are silent corruption, so we degrade-to-empty just like last_accessed.
    tags_raw = og.get("tags")
    tags = list(tags_raw) if isinstance(tags_raw, list) else []

    mem = Memory.from_stored_row(
        id=og_id,
        content=content,
        memory_type=str(og.get("memory_type") or "conversation"),
        domain=str(og.get("domain") or "us"),
        created_at=created_at,
        emotions=emotions,
        tags=tags,
        importance=importance,
        score=score,
        last_accessed_at=last_accessed_at,
        active=bool(og.get("active", True)),
        metadata=metadata,
    )
    if checked is not None:
        mem = checked(mem)
        if not importance_given:
            mem = replace(mem, importance=_derived_importance(mem._stored_emotions))
    return mem, None


def _derived_importance(emotions: dict[str, float]) -> float:
    """The importance of a legacy row that has none: the OG bucket of its feelings, or a moderate 4.0 when it has none."""
    if emotions:
        return float(_calc_importance_from_emotions(emotions))
    return 4.0


def _snippet(og: dict[str, Any], max_len: int = 120) -> str:
    """Human-readable excerpt of the offending OG record, truncated."""
    content = og.get("content") or ""
    if isinstance(content, str):
        return content[:max_len]
    return repr(content)[:max_len]
