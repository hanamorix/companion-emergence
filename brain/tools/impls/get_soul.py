"""get_soul tool implementation — SP-5 real impl."""

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
    """Return the persona's active crystallizations.

    Opens a SoulStore against the persona's crystallizations.db, reads
    all active (non-revoked) crystallizations, and returns them as a list
    of dicts. The store is closed in a finally block.
    """
    from brain.soul.store import SoulStore

    soul_db_path = persona_dir / "crystallizations.db"
    soul_store = SoulStore(str(soul_db_path))
    try:
        active = soul_store.list_active()
    finally:
        soul_store.close()

    return {
        "loaded": True,
        "count": len(active),
        "crystallizations": [c.to_dict() for c in active],
    }
