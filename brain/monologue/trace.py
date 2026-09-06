"""Tier 2 — persist a monologue as a retained `monologue_trace` memory.

The verbatim first-person drift is stored as a normal MemoryStore memory so
the existing forgetting engine ages it: FADE rewrites content→tombstone
summary (sharp→blurred), LOSE forgets it with grief + graveyard. Seeding the
trace with the current emotional aggregate means a thought formed in a charged
moment carries more salience (emotion is the heaviest forgetting weight) and
persists longer than flat idle drift. Importance now also feeds the
forgetting decay lever (P3 retention rework, Change 2) — kept modest here
because traces are high-volume.
"""
from __future__ import annotations

from brain.emotion.aggregate import aggregate_state
from brain.memory.store import Memory, MemoryStore

MONOLOGUE_TRACE_TYPE = "monologue_trace"
MONOLOGUE_DOMAIN = "monologue"
_TRACE_IMPORTANCE_FLOOR = 0.2
_TRACE_IMPORTANCE_CAP = 3.0
_TRACE_IMPORTANCE_SCALE = 0.3  # peak emotion intensity (0..10) * scale, then clamped


def _trace_importance(emotions: dict[str, float]) -> float:
    """Derive a per-trace importance from the aggregate's peak emotion
    intensity (P3 retention rework, Change 1 — replaces the flat 0.3
    constant so traces differentiate by how charged the moment was).
    Floored/capped modest: traces are high-volume, so even a peak trace
    should stay well below a deliberate write like a journal entry."""
    peak = max(emotions.values(), default=0.0)
    return min(_TRACE_IMPORTANCE_CAP, max(_TRACE_IMPORTANCE_FLOOR, peak * _TRACE_IMPORTANCE_SCALE))


def write_trace_memory(store: MemoryStore, monologue: str) -> str:
    """Persist `monologue` verbatim as a monologue_trace memory (state=active).
    Returns the new memory id."""
    emotions = dict(aggregate_state(store.list_active()).emotions)
    mem = Memory.create_new(
        content=monologue,
        memory_type=MONOLOGUE_TRACE_TYPE,
        domain=MONOLOGUE_DOMAIN,
        emotions=emotions,
        importance=_trace_importance(emotions),
    )
    from brain.memory.pending import route_write

    return route_write(store, mem, source="monologue_trace")
