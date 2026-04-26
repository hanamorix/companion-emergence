"""SP-4 COMMIT stage — direct write to MemoryStore + auto-Hebbian linking.

This bypasses the add_memory importance write-gate. The extraction LLM
has already provided an importance signal; the ingest pipeline's own
threshold is the gate, not a per-call gate in this module.

Memory type  = ExtractedItem.label  (one of VALID_LABELS)
Domain       = "brain"              (the conversation's own domain)
Tags         = ["auto_ingest", "conversation", label]
Importance   = item.importance / 10.0  (normalized to the store's 0..10.0 float scale)

After creation, auto-Hebbian: search the store for the top-3 related
memories (by keyword overlap) and strengthen each pair.
"""

from __future__ import annotations

import logging

from brain.ingest.types import ExtractedItem
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore

logger = logging.getLogger(__name__)

_AUTO_HEBBIAN_LIMIT = 4  # fetch 4, skip self → up to 3 edges


def commit_item(
    item: ExtractedItem,
    *,
    session_id: str,
    store: MemoryStore,
    hebbian: HebbianMatrix,
) -> str | None:
    """Create a Memory from an ExtractedItem and write it to the store.

    Does NOT go through any add_memory write-gate. The extraction LLM's
    importance signal is trusted directly.

    Auto-Hebbian wiring:
        After writing, call store.search_text(item.text, limit=4) to find
        the most textually related existing memories. Exclude the new
        memory's own id, then strengthen(new_id, related_id, delta=0.5)
        for each — up to 3 edges.

    Returns the new memory's id, or None if creation fails.
    """
    try:
        memory = Memory.create_new(
            content=item.text,
            memory_type=item.label,
            domain="brain",
            tags=["auto_ingest", "conversation", item.label],
            importance=float(item.importance),
            metadata={"source_summary": f"conversation:{session_id}"},
        )
        new_id = store.create(memory)
    except Exception as exc:  # noqa: BLE001
        logger.warning("commit_item: store.create failed: %s", exc)
        return None

    # Auto-Hebbian: find related memories and strengthen connections.
    # We use the longest meaningful word from the text as a keyword query so
    # that store.search_text (a LIKE substring match) has a realistic chance of
    # hitting overlapping memories without requiring the full phrase to match.
    try:
        words = [w for w in item.text.split() if len(w) > 4]
        keyword = words[0] if words else item.text[:20]
        related = store.search_text(keyword, active_only=True, limit=_AUTO_HEBBIAN_LIMIT)
        linked = 0
        for candidate in related:
            if candidate.id == new_id:
                continue
            hebbian.strengthen(new_id, candidate.id, delta=0.5)
            linked += 1
            if linked >= 3:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("commit_item: auto-hebbian failed for %s: %s", new_id, exc)

    return new_id
