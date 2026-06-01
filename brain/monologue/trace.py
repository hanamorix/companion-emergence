"""Tier 2 — persist a monologue as a retained `monologue_trace` memory.

The verbatim first-person drift is stored as a normal MemoryStore memory so
the existing forgetting engine ages it: FADE rewrites content→tombstone
summary (sharp→blurred), LOSE forgets it with grief + graveyard. Seeding the
trace with the current emotional aggregate means a thought formed in a charged
moment carries more salience (emotion is the heaviest forgetting weight) and
persists longer than flat idle drift.
"""
from __future__ import annotations

from brain.emotion.aggregate import aggregate_state
from brain.memory.store import Memory, MemoryStore

MONOLOGUE_TRACE_TYPE = "monologue_trace"
MONOLOGUE_DOMAIN = "monologue"
_TRACE_IMPORTANCE = 0.3  # calibration; not part of the forgetting salience formula


def write_trace_memory(store: MemoryStore, monologue: str) -> str:
    """Persist `monologue` verbatim as a monologue_trace memory (state=active).
    Returns the new memory id."""
    emotions = dict(aggregate_state(store.list_active()).emotions)
    mem = Memory.create_new(
        content=monologue,
        memory_type=MONOLOGUE_TRACE_TYPE,
        domain=MONOLOGUE_DOMAIN,
        emotions=emotions,
        importance=_TRACE_IMPORTANCE,
    )
    return store.create(mem)
