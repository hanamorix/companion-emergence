"""brain.maker.wiring — the three wire-backs of a completed making."""
from __future__ import annotations

import logging

from brain.maker.maker import Making

logger = logging.getLogger(__name__)

# base + per-source accent (mirrors the W8 reach_emotion map shape)
_BASE = {"satisfaction": 0.12, "tenderness": 0.08}
_SOURCE_ACCENT = {
    "grief": {"tenderness": 0.10},
    "joy": {"satisfaction": 0.10},
    "dream": {"wonder": 0.10},
    "soul": {"pride": 0.10},
}


def making_emotion_delta(making: Making, *, dominant_source: str) -> dict[str, float]:
    """Small vocab-filtered emotion delta for having made. Decay-subordinate."""
    from brain.chat.extractor import _filter_to_registered
    raw = dict(_BASE)
    for name, v in _SOURCE_ACCENT.get(dominant_source, {}).items():
        raw[name] = raw.get(name, 0.0) + v
    return _filter_to_registered(raw)


def write_making_memory(store, making: Making, *, emotions: dict[str, float]) -> None:
    """Episodic memory of the ACT of making (distinct from the artifact)."""
    from brain.memory.store import Memory
    content = f"I made something — \"{making.title}\" ({making.type}). It came from what's been moving in me."
    mem = Memory.create_new(
        content=content, memory_type="making", domain="interior",
        tags=["making", making.disposition], emotions=emotions or None,
    )
    try:
        store.create(mem)
    except Exception:
        logger.exception("maker: act-memory write failed")
