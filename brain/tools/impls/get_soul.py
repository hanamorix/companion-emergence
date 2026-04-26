"""get_soul tool implementation — STUB until SP-5."""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def get_soul(
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """STUB: soul module not yet ported (lands in SP-5).

    Returns empty crystallizations list so boot() composition works.
    """
    return {
        "crystallizations": [],
        "loaded": False,
        "note": "Soul module pending (SP-5)",
    }
