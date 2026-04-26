"""add_memory tool implementation — gated write with Hebbian auto-linking."""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.tools.impls._common import _write_gate_check


def add_memory(
    content: str,
    memory_type: str,
    domain: str,
    emotions: dict[str, int],
    tags: list[str] | None = None,
    importance: int | None = None,
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

    passes, effective_importance, emotion_score, rejection_reason = _write_gate_check(
        emotions, importance
    )

    if not passes:
        return {
            "created": False,
            "reason": rejection_reason,
        }

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
    store.create(memory)

    # Auto-Hebbian: find the 3 most text-related existing memories and
    # strengthen edges. We search on individual keywords from the content
    # (not the full phrase) so partial-match memories surface correctly.
    # Exclude the new memory itself.
    related_ids: list[str] = []
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
            hits = store.search_text(kw, active_only=True, limit=4)
            for rel in hits:
                if rel.id in seen_ids:
                    continue
                if len(related_ids) >= 3:
                    break
                hebbian.strengthen(memory.id, rel.id, delta=0.5)
                related_ids.append(rel.id)
                seen_ids.add(rel.id)
    except Exception:
        # Hebbian strengthening is best-effort — don't fail the write.
        pass

    return {
        "created": True,
        "id": memory.id,
        "importance": effective_importance,
        "auto_linked_to": related_ids,
    }
