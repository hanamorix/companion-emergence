"""The memory substrate — SQLite-backed store + embeddings + Hebbian.

Three sub-modules, each with a single responsibility:
- store: Memory dataclass + MemoryStore (SQLite-backed CRUD)
- embeddings: provider abstraction + content-hash cache
- hebbian: connection matrix + spreading activation

See spec Section 4.1 for the file-tree and Section 10.1 for the SQLite
data-layer decision (replaces OG's JSON/numpy files).
"""

from brain.memory.store import Memory, MemoryStore

__all__ = ["Memory", "MemoryStore"]
