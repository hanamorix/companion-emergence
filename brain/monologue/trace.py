"""Tier 2 — persist a monologue as a retained `monologue_trace` memory.

The trace is stored as a normal MemoryStore memory so the existing forgetting
engine ages it: FADE rewrites content→tombstone summary (sharp→blurred), LOSE
forgets it with grief + graveyard. With the emotion switch off it is the
verbatim first-person drift, seeded with the current emotional aggregate, so a
thought formed in a charged moment carries more salience (emotion is the
heaviest forgetting weight) and persists longer than flat idle drift. With the
engine live (emotion rework B1 PR 6) it is the note: the feed digest, seeded
with the engine's strongest short-average and long-average feelings; the raw
drift still reaches pass 2 from the capture's return value. Importance now
also feeds the forgetting decay lever (P3 retention rework, Change 2) — kept
modest here because traces are high-volume.
"""
from __future__ import annotations

import logging
from pathlib import Path

from brain.emotion.aggregate import aggregate_state
from brain.memory.store import Memory, MemoryStore

logger = logging.getLogger(__name__)

MONOLOGUE_TRACE_TYPE = "monologue_trace"
MONOLOGUE_DOMAIN = "monologue"
_TRACE_IMPORTANCE_FLOOR = 0.2
_TRACE_IMPORTANCE_CAP = 3.0
_TRACE_IMPORTANCE_SCALE = 0.3  # peak emotion intensity (0..10) * scale, then clamped


def _trace_importance(emotions: dict[str, float]) -> float:
    """Derive a per-trace importance from the peak of the trace's own feelings
    (the aggregate with the switch off, the engine's strongest feelings once
    live) (P3 retention rework, Change 1 — replaces the flat 0.3
    constant so traces differentiate by how charged the moment was).
    Floored/capped modest: traces are high-volume, so even a peak trace
    should stay well below a deliberate write like a journal entry."""
    peak = max(emotions.values(), default=0.0)
    return min(_TRACE_IMPORTANCE_CAP, max(_TRACE_IMPORTANCE_FLOOR, peak * _TRACE_IMPORTANCE_SCALE))


def _is_live(persona_dir: Path | None) -> bool:
    """Whether the emotion engine is live for this persona; a failure to tell counts as off, so the thought is never
    lost.

    Shared only by this writer's two halves (capture and the trace write), so they use one answer. Other writers check
    `engine_for(persona_dir).is_live()` inline (B1 design decision 3: no shared gate helper); don't import this one.
    """
    if persona_dir is None:
        return False
    try:
        from brain.emotion.engine import engine_for

        return engine_for(persona_dir).is_live()
    except Exception:  # noqa: BLE001
        logger.warning("monologue trace: could not tell whether the engine is live; storing today's trace", exc_info=True)
        return False


def write_trace_memory(
    store: MemoryStore,
    monologue: str,
    *,
    feed_digest: str,
    persona_dir: Path | None,
    live: bool | None = None,
) -> str:
    """Persist a monologue_trace memory (state=active) and return its id.

    With the emotion engine live (emotion rework B1 PR 6) the trace is the note: its content is `feed_digest`. Otherwise
    it is today's trace, the verbatim monologue. `live` is the caller's one answer to "is the engine live"; when it is
    None this asks the switch itself.
    """
    if persona_dir is None:
        live = False  # no persona, no engine to read: the whole trace is today's
    elif live is None:
        live = _is_live(persona_dir)
    if live:
        from brain.emotion.engine import engine_for

        emotions = _strongest_feelings(engine_for(persona_dir))
    else:
        emotions = dict(aggregate_state(store.list_active()).emotions)
    mem = Memory.create_new(
        content=feed_digest if live else monologue,
        memory_type=MONOLOGUE_TRACE_TYPE,
        domain=MONOLOGUE_DOMAIN,
        emotions=emotions,
        importance=_trace_importance(emotions),
    )
    from brain.memory.pending import route_write

    return route_write(store, mem, source="monologue_trace")


def _strongest_feelings(eng) -> dict[str, float]:
    """The engine's strongest short-average feeling and strongest long-average feeling, as of its last drain. One key
    with the larger value when both are the same feeling; ties go to the alphabetically first name. A read that fails
    gives {}, so the note is still stored."""
    try:
        readings = eng.read()
    except Exception:  # noqa: BLE001
        logger.warning("monologue trace: engine read failed; storing no feelings", exc_info=True)
        return {}
    feelings: dict[str, float] = {}
    for field in ("short", "long"):
        felt = [n for n in readings if getattr(readings[n], field) > 0]
        if not felt:
            continue
        name = min(felt, key=lambda n: (-getattr(readings[n], field), n))
        feelings[name] = max(getattr(readings[name], field), feelings.get(name, 0.0))
    return feelings
