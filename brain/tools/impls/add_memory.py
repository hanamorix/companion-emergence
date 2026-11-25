"""add_memory tool implementation — gated write with Hebbian auto-linking."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from brain.emotion import vocabulary
from brain.emotion.engine_utils import memory_feelings
from brain.emotion.persona_loader import register_new_word
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.tools.impls._common import _write_gate_check

logger = logging.getLogger(__name__)


_MAX_HALF_LIFE_DAYS = 3650.0


def _valid_half_life(value: object) -> float | None:
    """A real finite number with 0 < v <= 3650, else None (the default applies). bool is not a number here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # The range first: comparing works on any int, while math.isfinite raises OverflowError on one too large for a float.
    if not 0 < value <= _MAX_HALF_LIFE_DAYS or not math.isfinite(value):
        return None
    return float(value)


def add_memory(
    content: str,
    memory_type: str,
    domain: str,
    emotions: dict[str, int],
    tags: list[str] | None = None,
    importance: int | None = None,
    emotion_half_lives: dict[str, float] | None = None,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Write a gated memory.

    Write gate (ported from OG nell_tools.py:add_memory):
      - emotion_score = sum(emotions.values())
      - effective_importance = importance if provided else auto-calculated
      - PASSES iff emotion_score >= 15 OR effective_importance >= 7
      - Else returns {"created": False, "reason": "..."} with below-threshold detail

    On gate pass:
      - Creates a Memory via Memory.create_new
      - Persists via store.create
      - Auto-Hebbian: finds top-3 related memories by text search, strengthens
        edges with delta=0.5 each
      - Returns {"created": True, "id": ..., "importance": ..., "auto_linked_to": [...]}

    Returns
    -------
    dict — either a rejection dict (created=False) or a success dict (created=True).
    """
    tags = list(tags or [])
    if not isinstance(emotion_half_lives, dict):
        emotion_half_lives = {}

    # A name already in the vocabulary skips the gate; a new word must pass it. Raw
    # spellings are checked before pooling, so refusals name what was actually written.
    kept: dict[str, float] = {}
    refused: list[str] = []
    for raw, value in emotions.items():
        if vocabulary.get(raw) is None and vocabulary.validate_new_name(raw) is None:
            refused.append(raw)
        else:
            kept[raw] = value
    refused.sort()
    emotions = vocabulary.pool_by_canonical_name(kept)

    passes, effective_importance, emotion_score, rejection_reason = _write_gate_check(
        emotions, importance
    )

    if not passes:
        return {
            "created": False,
            "reason": rejection_reason,
            "refused_emotions": refused,
        }

    canonical_half_lives = {
        vocabulary.canonical_name(raw): value
        for raw, value in emotion_half_lives.items()
        if isinstance(raw, str)
    }
    unregistered: list[str] = []
    for name in emotions:
        if vocabulary.get(name) is None:
            registered_as = register_new_word(
                persona_dir / "emotion_vocabulary.json",
                name,
                half_life_days=_valid_half_life(
                    emotion_half_lives.get(name, canonical_half_lives.get(name))
                ),
            )
            if registered_as is None:
                unregistered.append(name)
    unregistered.sort()
    if unregistered:
        # The memory still keeps these feelings; only the vocabulary entry is missing.
        logger.warning(
            "add_memory could not register new emotion word(s) %s in the vocabulary", unregistered
        )

    # Build the memory with importance as a float (Memory.create_new normalises
    # importance to score/10 if None, but we've already calculated it — pass
    # explicitly so the stored value reflects the gate-validated importance).
    float_emotions = {k: float(v) for k, v in emotions.items()}
    memory = Memory.create_new(
        content=content,
        memory_type=memory_type,
        domain=domain,
        emotions=float_emotions,
        tags=tags,
        importance=float(effective_importance),
        metadata={"tags": tags, "importance": effective_importance},
    )
    # She reaches this memory at the moment she makes it (emotion rework, Hana half, Task 6).
    memory.last_accessed_at = memory.created_at
    store.create(memory)

    # Body event hook: when an LLM-tagged memory commits with climax at or above
    # CLIMAX_THRESHOLD, auto-write a private journal_entry referencing this
    # originating memory. Fail-soft inside record_climax_event — never affects the
    # add_memory return value. Per spec docs/source-spec/2026-04-29-body-state-design.md §5.1.
    from brain.body.events import CLIMAX_THRESHOLD, record_climax_event

    # A memory created this instant has no age to fade: the pure read (no engine call, which could raise after the row is committed).
    if memory_feelings(memory, None, None, None).feelings.get("climax", 0.0) >= CLIMAX_THRESHOLD:
        record_climax_event(
            originating_memory=memory,
            store=store,
            persona_dir=persona_dir,
        )

    # Auto-Hebbian: find the 3 most text-related existing memories and
    # strengthen edges. We search on individual keywords from the content
    # (not the full phrase) so partial-match memories surface correctly.
    # Exclude the new memory itself.
    related_ids: list[str] = []
    auto_link_error: str | None = None
    try:
        # Extract up to 3 significant keywords from the content (skip short words)
        words = [w.strip(".,!?;:\"'") for w in content.lower().split()]
        keywords = [w for w in words if len(w) >= 4][:3]
        if not keywords:
            keywords = words[:3]

        seen_ids: set[str] = {memory.id}
        for kw in keywords:
            if len(related_ids) >= 3:
                break
            # bump=False: finding memories to link is not recalling them, this new one included.
            hits = store.search_text(kw, active_only=True, limit=4, bump=False)
            for rel in hits:
                if rel.id in seen_ids:
                    continue
                if len(related_ids) >= 3:
                    break
                hebbian.strengthen(memory.id, rel.id, delta=0.5)
                related_ids.append(rel.id)
                seen_ids.add(rel.id)
    except Exception as exc:
        # Hebbian strengthening is best-effort — don't fail the write, but do
        # surface the partial failure so callers and logs are not misled.
        auto_link_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "add_memory auto-link failed memory_id=%s err=%s", memory.id, auto_link_error
        )

    result = {
        "created": True,
        "id": memory.id,
        "importance": effective_importance,
        "auto_linked_to": related_ids,
        "refused_emotions": refused,
        "unregistered_emotions": unregistered,
    }
    if auto_link_error is not None:
        result["auto_link_error"] = auto_link_error
    return result
