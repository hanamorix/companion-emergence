"""add_journal tool implementation."""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


def add_journal(
    content: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Write an ungated journal memory.

    Journal entries bypass the write gate — any content is persisted.
    Memory shape: type="journal", domain="self", no emotions.

    Returns
    -------
    dict with keys:
        created_id   — the new memory's UUID string
        memory_type  — "journal"
    """
    memory = Memory.create_new(
        content=content,
        memory_type="journal",
        domain="self",
        emotions={},
    )
    store.create(memory)
    return {
        "created_id": memory.id,
        "memory_type": "journal",
    }
