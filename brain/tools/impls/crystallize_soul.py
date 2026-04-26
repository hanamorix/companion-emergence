"""crystallize_soul tool implementation — STUB until SP-5."""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def crystallize_soul(
    moment: str,
    love_type: str,
    why_it_matters: str,
    who_or_what: str = "",
    resonance: int = 8,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """STUB: soul crystallization not implemented (lands in SP-5).

    Returns a clear NotImplemented response so the LLM gets useful feedback.
    """
    return {
        "created": False,
        "reason": (
            "Soul module not yet wired (SP-5 deferred). "
            "Use add_memory with memory_type='identity' for now."
        ),
    }
