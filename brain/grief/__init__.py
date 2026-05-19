"""brain.grief — affective layer over forgetting + narrative-memory loss.

Public surface — three entry points wired into the brain at fault-isolated
call sites:

    handle_drop(memory=, salience_at_drop=, persona_dir=, store=)
        Called inline by forgetting.__init__.run_pass after hard_delete.

    handle_arc_close(arc=, persona_dir=, store=)
        Called inline by narrative_memory.__init__ when an open arc transitions
        to closed.

    handle_recall_touch(touched_ids=, graveyard_entries=, persona_dir=, store=,
                        lived_age_hours_now=, triggering_arc_id=None)
        Called by chat.prompt._build_recall_block, tools.dispatch (after
        recall_forgotten resolves), and narrative_memory membership refresh.

Spec: docs/superpowers/specs/2026-05-19-grief-design.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from brain.grief import breadcrumb, policy
from brain.grief.prompt import render_grief_block
from brain.grief.recall import handle_recall_touch
from brain.memory.store import Memory, MemoryStore

log = logging.getLogger(__name__)

__all__ = [
    "handle_drop",
    "handle_arc_close",
    "handle_recall_touch",
    "render_grief_block",
]


def _dominant_non_grief_emotion(emotions: dict[str, float]) -> tuple[str, float] | None:
    candidates = [
        (name, float(val))
        for name, val in emotions.items()
        if name != "memory_grief" and isinstance(val, (int, float)) and float(val) > 0.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda kv: kv[1])


def handle_drop(
    *,
    memory: Memory,
    salience_at_drop: float,
    persona_dir: Path,
    store: MemoryStore,
) -> None:
    """Write a drop-time grief breadcrumb if intensity >= THRESHOLD.

    Called after the graveyard write + hard_delete in forgetting.__init__.run_pass.
    Fault-isolated by caller — this function does NOT raise on internal errors;
    it logs and returns.
    """
    try:
        # emotions are stored on a 0-10 scale; pass raw value — compute_drop_intensity
        # clamps its result to [0, 10], so out-of-range inputs are safe.
        emotion_max = max(memory.emotions.values()) if memory.emotions else 0.0
        intensity = breadcrumb.compute_drop_intensity(
            emotion_at_ingest_max=emotion_max, salience_at_drop=salience_at_drop
        )
        if intensity < policy.THRESHOLD:
            return
        residue = _dominant_non_grief_emotion(dict(memory.emotions or {}))
        breadcrumb.write_breadcrumb(
            store=store,
            intensity=intensity,
            subtype="drop",
            referent_type="memory",
            referent_id=memory.id,
            content=breadcrumb.drop_phrase(memory.content or ""),
            residue_emotion=residue,
        )
    except Exception:
        log.exception("grief.handle_drop failed for memory_id=%s", getattr(memory, "id", "?"))


def handle_arc_close(
    *,
    arc: Any,
    persona_dir: Path,
    store: MemoryStore,
) -> None:
    """Write an arc-close grief breadcrumb if intensity >= THRESHOLD.

    Expects the arc to carry two fields populated at close time by
    narrative_memory:
      - max_member_emotion_normalised: float in [0, 1]
      - dominant_non_grief_emotion: tuple[str, float] | None (intensity 0-10)

    See Phase 7 — narrative_memory wiring populates these on the closed
    arc before calling. Older arcs missing these fields degrade safely
    (no breadcrumb written).
    """
    try:
        max_member = float(getattr(arc, "max_member_emotion_normalised", None) or 0.0)
        intensity = breadcrumb.compute_arc_close_intensity(arc_max_member_emotion=max_member)
        if intensity < policy.THRESHOLD:
            return
        residue = getattr(arc, "dominant_non_grief_emotion", None)
        breadcrumb.write_breadcrumb(
            store=store,
            intensity=intensity,
            subtype="arc_close",
            referent_type="arc",
            referent_id=getattr(arc, "id", "?"),
            content=breadcrumb.arc_close_phrase(getattr(arc, "title", "")),
            residue_emotion=residue,
        )
    except Exception:
        log.exception("grief.handle_arc_close failed for arc_id=%s", getattr(arc, "id", "?"))
