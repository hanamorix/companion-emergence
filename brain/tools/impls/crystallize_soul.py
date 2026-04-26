"""crystallize_soul tool implementation — SP-5 real impl.

Direct crystallization path — bypasses the candidate-review pipeline.
Used when the chat engine's LLM decides a moment is soul-worthy and acts
on it directly (no queue, no later review). Validates love_type and resonance
before writing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
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
    """Directly crystallize a moment into the soul.

    Validates love_type against LOVE_TYPES. Validates resonance 1-10.
    Creates a Crystallization with permanent=True and writes it to
    crystallizations.db.

    Returns
    -------
    {"created": True, "id": id, "love_type": ..., "resonance": ...}
        on success.
    {"created": False, "reason": "..."}
        on validation failure.
    """
    from brain.soul.crystallization import Crystallization
    from brain.soul.love_types import LOVE_TYPES
    from brain.soul.store import SoulStore

    # Validate love_type
    if love_type not in LOVE_TYPES:
        valid = ", ".join(LOVE_TYPES.keys())
        return {
            "created": False,
            "reason": f"unknown love_type {love_type!r}; valid: {valid}",
        }

    # Validate resonance
    try:
        resonance_int = int(resonance)
    except (TypeError, ValueError):
        return {
            "created": False,
            "reason": f"resonance must be int 1-10, got {resonance!r}",
        }
    if not (1 <= resonance_int <= 10):
        return {
            "created": False,
            "reason": f"resonance out of range: {resonance_int}",
        }

    c = Crystallization(
        id=str(uuid.uuid4()),
        moment=moment,
        love_type=love_type,
        why_it_matters=why_it_matters,
        crystallized_at=datetime.now(UTC),
        who_or_what=who_or_what,
        resonance=resonance_int,
        permanent=True,
    )

    soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        soul_store.create(c)
    finally:
        soul_store.close()

    return {
        "created": True,
        "id": c.id,
        "love_type": love_type,
        "resonance": resonance_int,
    }
