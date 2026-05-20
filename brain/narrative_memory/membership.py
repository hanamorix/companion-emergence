"""Hebbian-OR-embedding membership test (spec §4).

Pure functions. Two duck-typed protocol views (`HebbianView`,
`EmbeddingsView`) keep this module test-friendly without depending on
the real HebbianMatrix / EmbeddingIndex classes.

`centroid_for` is memoised per pass via a caller-supplied cache dict
keyed on arc.id, thrown away after the pass.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from brain.memory.embeddings import cosine_similarity
from brain.narrative_memory.arc import Arc

# Module-top tunables (spec §4 + §2 policy.py — duplicated here for fast read
# at the membership.py call site without importing policy).
MEMBER_HEBBIAN_THRESHOLD: float = 3.0
MEMBER_EMBEDDING_THRESHOLD: float = 0.6


class HebbianView(Protocol):
    def weight(self, a: str, b: str) -> float: ...


class EmbeddingsView(Protocol):
    def get(self, memory_id: str) -> np.ndarray | None: ...


class _MemoryLike(Protocol):
    id: str


def is_candidate(
    memory: _MemoryLike,
    arc: Arc,
    *,
    hebbian: HebbianView,
    embeddings: EmbeddingsView,
    centroid_cache: dict[str, np.ndarray | None],
) -> tuple[bool, str | None]:
    """Return (True, via) if memory should join arc; (False, None) otherwise.

    Hebbian path runs first (cheap pairwise lookup). On miss, embedding
    path computes cosine to arc centroid (cached per pass).
    """
    # Hebbian path
    for member in arc.members:
        if hebbian.weight(memory.id, member.memory_id) >= MEMBER_HEBBIAN_THRESHOLD:
            return True, "hebbian"

    # Embedding path
    centroid = centroid_for(arc, embeddings=embeddings, cache=centroid_cache)
    if centroid is None:
        return False, None
    vec = embeddings.get(memory.id)
    if vec is None:
        return False, None
    if cosine_similarity(vec, centroid) >= MEMBER_EMBEDDING_THRESHOLD:
        return True, "embedding"
    return False, None


def centroid_for(
    arc: Arc,
    *,
    embeddings: EmbeddingsView,
    cache: dict[str, np.ndarray | None],
) -> np.ndarray | None:
    """Mean of arc-member embedding vectors, cached per arc per pass.

    Members with missing vectors are skipped from the mean. If ALL
    members are missing, returns None (caller falls back to hebbian-only).
    Single-member arcs return the seed memory's vector directly.
    """
    if arc.id in cache:
        return cache[arc.id]

    vectors: list[np.ndarray] = []
    for member in arc.members:
        vec = embeddings.get(member.memory_id)
        if vec is not None:
            vectors.append(vec)

    if not vectors:
        cache[arc.id] = None
        return None

    if len(vectors) == 1:
        cache[arc.id] = vectors[0]
        return vectors[0]

    centroid = np.mean(np.stack(vectors), axis=0)
    cache[arc.id] = centroid
    return centroid
