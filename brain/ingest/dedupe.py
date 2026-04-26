"""SP-4 DEDUPE stage — cosine-similarity check against existing embeddings.

Design decision: dedupe is opt-in. When embeddings is None (the default in
the pipeline signature), this function returns False unconditionally and the
item passes through. This keeps SP-4 self-contained — the pipeline runs even
without embedding infrastructure.

When embeddings IS provided, we compute the candidate's embedding and compare
it against every cached vector. If any similarity >= threshold, the item is a
duplicate and should be skipped.

The EmbeddingCache API exposes:
  get_or_compute(content) -> np.ndarray   — compute + cache the vector
  count() -> int                          — number of cached vectors

Iterating all vectors requires querying the underlying SQLite table directly
via the cache's internal connection. We do this defensively — if anything
fails, we return False (let the item through) rather than crashing the pipeline.
"""

from __future__ import annotations

import logging

import numpy as np

from brain.memory.embeddings import EmbeddingCache, cosine_similarity
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

DEFAULT_DEDUP_THRESHOLD = 0.88


def is_duplicate(
    text: str,
    *,
    store: MemoryStore,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    embeddings: EmbeddingCache | None = None,
) -> bool:
    """Cosine-similarity check against the persona's existing embeddings.

    If ``embeddings`` is None or has no cached entries: return False —
    dedupe is impossible, let the item through.

    Otherwise:
      1. Compute (or retrieve from cache) the embedding for ``text``.
      2. Iterate all stored (content_hash, vector) pairs.
      3. Return True if max cosine similarity >= threshold.

    Any exception during the process is caught and logged; we return False
    on failure (safe default — at worst we commit a near-duplicate).
    """
    if embeddings is None:
        return False

    try:
        # Snapshot existing rows BEFORE computing the candidate embedding so we
        # don't accidentally compare the text against itself if get_or_compute
        # adds it to the cache during this call.
        existing_rows = embeddings._conn.execute(  # noqa: SLF001
            "SELECT vector, dim FROM embedding_cache"
        ).fetchall()

        if not existing_rows:
            return False

        candidate = embeddings.get_or_compute(text)

        max_sim = 0.0
        for vec_bytes, dim in existing_rows:
            stored_vec = np.frombuffer(vec_bytes, dtype=np.float32).copy().reshape(dim)
            sim = cosine_similarity(candidate, stored_vec)
            if sim > max_sim:
                max_sim = sim
            if max_sim >= threshold:
                return True

        return max_sim >= threshold

    except Exception as exc:  # noqa: BLE001
        logger.warning("is_duplicate: error during similarity check, letting item through: %s", exc)
        return False
