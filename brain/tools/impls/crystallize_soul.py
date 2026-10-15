"""crystallize_soul tool implementation — SP-5 real impl.

Direct crystallization path — bypasses the candidate-review pipeline.
Used when the chat engine's LLM decides a moment is soul-worthy and acts
on it directly (no queue, no later review). Validates love_type and resonance
before writing.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def crystallize_soul(
    moment: str,
    love_type: str,
    why_it_matters: str,
    who_or_what: str = "",
    resonance: int = 8,
    source_memory_id: Any = None,
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

    result = {
        "created": True,
        "id": c.id,
        "love_type": love_type,
        "resonance": resonance_int,
    }
    soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        soul_store.create(c)
        _link_source_memory(soul_store, store, c.id, source_memory_id, result)
    finally:
        soul_store.close()

    return result


def _link_source_memory(
    soul_store: Any,
    store: MemoryStore,
    crystallization_id: str,
    source_memory_id: Any,
    result: dict,
) -> None:
    """Protect the memory that prompted this crystallisation from forgetting (#371)."""
    wanted = source_memory_id.strip() if isinstance(source_memory_id, str) else ""
    if not wanted:
        return
    # The crystallisation is already committed and permanent: nothing about an optional id may
    # undo it. Any failure here degrades to "saved, not linked" and is logged, never raised.
    try:
        memory = store.get(wanted, bump=False)
        if memory is not None:
            soul_store.link_source_memory(crystallization_id, wanted)
            result["source_memory_linked"] = True
            if memory.state == "fading":
                result["note"] = _FADING_NOTE
            return
        logger.warning(
            "crystallize_soul: no memory %s to link crystallization %s to; it was saved without a link",
            wanted,
            crystallization_id,
        )
    except Exception:
        logger.warning(
            "crystallize_soul: could not link crystallization %s to memory %s; it was saved without a link",
            crystallization_id,
            wanted,
            exc_info=True,
        )
    _mark_not_linked(result)


# One note for every "saved but not linked" outcome (missing memory, failed lookup, failed link write)
# so the model is never invited to retry the crystallisation or to search for the memory.
_NOT_LINKED_NOTE = (
    "The crystallisation was saved, but no memory was linked. "
    "Do not repeat it and do not search for the memory."
)


# The exemption stops the forgetting pass touching a memory; it does not un-fade one (#372).
_FADING_NOTE = (
    "That memory is already fading. It is now protected from being lost, "
    "but it is not restored: it stays faded."
)


def _mark_not_linked(result: dict) -> None:
    result["source_memory_linked"] = False
    result["note"] = _NOT_LINKED_NOTE
