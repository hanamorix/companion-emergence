"""read_full_memory tool implementation.

The deliberate-read companion to the snippet-then-read flow (P2): recall and
search_memories surface truncated snippets + ids without bumping recall_count;
the model pulls the few it actually wants in full through this tool. Only this
full read bumps ``recall_count`` (via ``store.get()``) — the honest
"I engaged with this" signal that feeds the forgetting pass.
"""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
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

    Calls ``store.get()`` — which bumps ``recall_count`` + ``last_accessed_at``
    (the deliberate-engagement bump). Returns the full ``_mem_to_result`` dict,
    or ``{"error": "not found", "id": memory_id}`` when the id is unknown.

    ``hebbian``/``persona_dir`` are injected by dispatch but unused here.
    """
    mem = store.get(memory_id)
    if mem is None:
        return {"error": "not found", "id": memory_id}
    return _mem_to_result(mem)
