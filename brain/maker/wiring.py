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
