"""SP-4 DEDUPE stage — cosine-similarity check against existing memories.

Sources vectors off the memories row / warm matrix (F1 #259).

  - EXISTING vectors: every active, currently-embedded row in `store`'s
    memories.db, via the process-wide warm matrix
    (`build_embedding_matrix(store.db_path).snapshot()`). Pre-migration /
    pre-backfill most rows have `embedding IS NULL`, so this set can be
    sparse — dedup then simply finds fewer semantic near-dups, the same
    graceful degradation semantic recall has while the backfill catches up.
    We deliberately do NOT embed-on-read to fill gaps here.
  - CANDIDATE vector: the transient, not-yet-committed `text` is embedded
    DIRECTLY via `build_embedding_provider().embed(text)` — there is no
    cache row to write (and therefore nothing to undo if the item's commit
    later fails).

Both the matrix and the provider key off the SAME current model id
(`model_tier.model_for_tier(TIER_EMBEDDING)`), so a row left over from a
prior/different embedding model is never loaded into the comparison — the
matrix's own build query already filters to the current model id.

Any exception during the process (matrix build, provider embed, etc.) is
caught and logged; we return False on failure (safe default — at worst we
commit a near-duplicate) rather than crashing the pipeline.
"""

from __future__ import annotations

import logging

from brain.memory import embeddings as embeddings_mod
from brain.memory.embedding_matrix import build_embedding_matrix
from brain.memory.embeddings import cosine_similarity
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

DEFAULT_DEDUP_THRESHOLD = 0.88


def is_duplicate(
    text: str,
    *,
    store: MemoryStore,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> bool:
    """Cosine-similarity check against the persona's existing row-embedded memories.

    1. Snapshot the warm matrix's currently-embedded vectors for this store.
       If none exist yet (cold-start / pre-backfill), return False.
    2. Embed ``text`` directly through the production provider (no cache
       row written).
    3. Return True if max cosine similarity against the snapshot >= threshold.

    Any exception during the process is caught and logged; we return False
    on failure (safe default — at worst we commit a near-duplicate).
    """
    try:
        existing = build_embedding_matrix(store.db_path).snapshot()
        if not existing:
            return False

        # Looked up via the MODULE (not a bare imported name) so a test's
        # monkeypatch on `embeddings.build_embedding_provider` is honored —
        # mirrors `MemoryStore.embed_row`'s identical dynamic lookup.
        candidate = embeddings_mod.build_embedding_provider().embed(text).astype("float32")

        max_sim = 0.0
        for stored_vec in existing.values():
            sim = cosine_similarity(candidate, stored_vec)
            if sim > max_sim:
                max_sim = sim
            if max_sim >= threshold:
                return True

        return max_sim >= threshold

    except Exception as exc:  # noqa: BLE001
        logger.warning("is_duplicate: error during similarity check, letting item through: %s", exc)
        return False
