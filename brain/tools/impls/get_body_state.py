"""get_body_state tool implementation — STUB until body-state module lands."""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def get_body_state(
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """STUB: body-state module not yet ported.

    OG read dall_body_state.json (energy, comfort, arousal, days_since_contact).
    New framework defers body-state module entirely; chat engine works without it.

    Returns reasonable defaults so boot() composition doesn't break.
    """
    return {
        "loaded": False,
        "energy": 5,
        "comfort": 5,
        "arousal": 0,
        "days_since_contact": 0.0,
        "voice_state": "default",
        "note": "Body-state module pending",
    }
