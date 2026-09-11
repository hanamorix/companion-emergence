"""read_full_memory tool implementation.

The deliberate-read companion to the snippet-then-read flow (P2): recall and
search_memories surface truncated snippets + ids without bumping recall_count;
the model pulls the few it actually wants in full through this tool. A full
read opens the memory in full, so it routes through the ONE ``open_memory`` door
(#231 consolidation) — the same door the passive full-render sites use. As a
deliberate open it bumps ``recall_count`` (+1.0) AND touches ``last_accessed_at``
(the honest "I engaged with this" signal that feeds the forgetting pass) and,
per reconciliation C, now ALSO enqueues the memory for reappraisal — "if a
memory gets opened, it goes back into the reappraisal queue, doesn't matter how".
"""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.recall_open import open_memory
from brain.memory.store import MemoryStore
from brain.tools.impls._common import _mem_to_result


def read_full_memory(
    memory_id: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Return the full (untruncated) memory body for ``memory_id``.

    Fetches the row (no bump) for the existence check + result, then routes the
    open through ``open_memory(deliberate=True)`` — which applies the same +1.0
    recall_count and ``last_accessed_at`` touch as before AND enqueues a
    reappraisal (reconciliation C). Returns the full ``_mem_to_result`` dict, or
    ``{"error": "not found", "id": memory_id}`` when the id is unknown.

    ``hebbian`` is injected by dispatch but unused here.
    """
    mem = store.get(memory_id, bump=False)
    if mem is None:
        return {"error": "not found", "id": memory_id}
    open_memory(mem, store=store, persona_dir=persona_dir, deliberate=True, seen=None)
    return _mem_to_result(mem)
