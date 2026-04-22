"""The memory substrate — SQLite-backed store + embeddings + Hebbian + search.

Four sub-modules, each with a single responsibility:
- store: Memory dataclass + MemoryStore (SQLite-backed CRUD)
- embeddings: provider abstraction + content-hash cache
- hebbian: connection matrix + spreading activation
- search: semantic + emotional + temporal + spreading queries

See spec Section 4.1 for the file-tree and Section 10.1 for the SQLite
data-layer decision (replaces OG's JSON/numpy files).
"""

from brain.memory.store import Memory

__all__ = ["Memory"]
