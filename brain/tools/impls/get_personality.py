"""get_personality tool implementation — STUB until SP-6."""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def get_personality(
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """STUB: personality loader not yet ported.

    OG read nell_personality.json. New framework doesn't have a personality
    module yet (likely lands as part of SP-6 voice.md or a future per-persona
    personality file).

    Returns a minimal placeholder so boot() can still compose without crashing.
    """
    return {
        "loaded": False,
        "note": "Personality module pending — voice.md (SP-6) will likely subsume this",
    }
