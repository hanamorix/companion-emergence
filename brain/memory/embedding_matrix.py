"""Warm in-process vector matrix over `memories.embedding` (F1, #259 step 2).

A process-level `{memory_id -> np.ndarray(float32, 384-dim)}` map, sourced
from the `embedding` column F1 added to the `memories` table (see
`brain/memory/store.py`'s `_SCHEMA`). This is the read-side cache a recall
consumer will eventually query directly instead of the old content-hash
`embeddings.db` join — but THIS module is additive only: nothing in this
increment wires it into recall, backfill, dedupe, or clustering. It exists
so the next increment has a warm-matrix primitive ready to consume.

Build cost: the matrix is built LAZILY, on first access, never at import or
construction time — building at startup would pay a cost on every process
boot (including ones that never touch semantic recall) and would race an
empty/mid-migration DB. The build query is deliberately lean:

    SELECT id, embedding FROM memories WHERE active = 1 AND embedding IS NOT NULL

not `store.list_active()` / `_row_to_memory` — decoding every JSON field
(emotions, tags, metadata) on every row just to throw it away and keep the
vector would be pure waste at corpus scale.

Concurrency (mirrors the double-checked-locking pattern in
`brain/memory/embeddings.py`'s `_provider_cache_lock`): a single
`threading.RLock` guards the live `{id: vector}` dict reference itself, but
the lock is held only for the O(1)/O(batch-of-1) operations — a dict read, a
single-item put, a single-item evict, or the final reference swap after a
rebuild. The (possibly slow) SQLite scan that a lazy build or a full rebuild
performs runs OUTSIDE the lock, into a fresh local dict, which is then
swapped into `self._vectors` atomically under a brief lock acquisition. A
concurrent reader (e.g. a recall-path lookup on the request thread) can
therefore never be blocked behind a full-batch build/rebuild (a
backfill/fade/delete tick on the supervisor thread) for longer than a single
dict assignment — see the module docstring's "never holding the lock during
the (possibly slow) build" requirement in the F1 spec, section 2.

`model_id` tracks which embedding model produced the vectors currently held
(set at construction, updated by `rebuild(model_id=...)`). Nothing in this
module compares it against the live provider's model id yet — that
comparison, and the decision to trigger a rebuild on a model-id swap or
process restart, is a later increment's job. `rebuild()` is exposed now so
that wiring has something to call.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np


class EmbeddingMatrix:
    """Warm `{memory_id: np.ndarray}` cache over one persona's `memories.db`.

    Not thread-affine: `get`/`snapshot` (reads), `put`/`evict` (per-item
    writes), and `rebuild` (full reload) may all be called from different
    threads. See module docstring for the locking discipline.
    """

    def __init__(self, db_path: str | Path, *, model_id: str) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._vectors: dict[str, np.ndarray] = {}
        self._built = False
        # The model_id the CURRENTLY HELD vectors were built under. Set at
        # construction; updated by rebuild(model_id=...) on a model swap.
        self.model_id = model_id

    # -- lifecycle -----------------------------------------------------

    def ensure_built(self) -> None:
        """Build the matrix if it hasn't been built yet. Idempotent and
        safe to call from multiple threads — a race on the very first
        access may cause more than one thread to run the (read-only) scan
        redundantly, but each computes the same result off-lock and only
        the final swap is locked, so no torn or partial state is ever
        observable by a reader.
        """
        if self._built:
            return
        vectors = self._load_from_db()
        with self._lock:
            if not self._built:
                self._vectors = vectors
                self._built = True

    def rebuild(self, *, model_id: str | None = None) -> None:
        """Full rebuild: reload every active embedded row from disk into a
        NEW dict, then atomically swap it in. Used on a model-id swap
        (pass the new `model_id`) or a process restart (call with no
        argument to reload under the same model_id).

        The slow part (the DB scan) happens BEFORE the lock is acquired —
        readers keep being served the OLD matrix for the full duration of
        the scan and only block for the instant of the reference swap.
        """
        vectors = self._load_from_db()
        with self._lock:
            self._vectors = vectors
            self._built = True
            if model_id is not None:
                self.model_id = model_id

    def _load_from_db(self) -> dict[str, np.ndarray]:
        """The lean lazy-build query. Opens and closes its own short-lived
        connection rather than sharing a `MemoryStore`'s connection — this
        runs on whatever thread triggers a build/rebuild (request thread
        for the first lazy access, supervisor thread for a maintenance
        rebuild), and sqlite3 connections are not safe to share across
        threads without extra coordination the rest of this module doesn't
        need.
        """
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE active = 1 AND embedding IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        return {
            row[0]: np.frombuffer(row[1], dtype=np.float32).copy() for row in rows
        }

    # -- reads -----------------------------------------------------------

    def get(self, memory_id: str) -> np.ndarray | None:
        """The vector for one memory id, or None if absent/not embedded.
        Triggers a lazy build on first call. Returns a COPY — callers may
        mutate their own result freely without corrupting the matrix.
        """
        self.ensure_built()
        with self._lock:
            vec = self._vectors.get(memory_id)
        return None if vec is None else vec.copy()

    def snapshot(self) -> dict[str, np.ndarray]:
        """A shallow-copied `{id: vector}` mapping snapshot — safe for a
        caller to iterate over (e.g. to brute-force a cosine scan for
        candidate reads) without holding the matrix's lock and without the
        risk of the underlying dict changing size mid-iteration. Triggers a
        lazy build on first call.
        """
        self.ensure_built()
        with self._lock:
            return dict(self._vectors)

    def __len__(self) -> int:
        self.ensure_built()
        with self._lock:
            return len(self._vectors)

    def __contains__(self, memory_id: str) -> bool:
        self.ensure_built()
        with self._lock:
            return memory_id in self._vectors

    # -- per-item writes ---------------------------------------------------

    def put(self, memory_id: str, vector: np.ndarray) -> None:
        """Insert/update ONE entry. Short, per-item lock hold — never call
        this in a loop while holding an outer lock of your own across many
        items; the whole point is that each call here is independently
        cheap for a concurrent reader to wait behind.

        Deliberately does NOT mark the matrix as built: a `put` that lands
        before the first lazy build (e.g. an embed-on-write for the very
        first memory in a fresh process) must not make a later
        `ensure_built()` skip loading the rest of the corpus. The row is
        already durable in `memories` by the time a caller puts it here, so
        the eventual lazy/rebuild load picks it up regardless — this is
        purely a same-process warm-cache update, never the sole record.
        """
        vec = np.asarray(vector, dtype=np.float32).copy()
        with self._lock:
            self._vectors[memory_id] = vec

    def evict(self, memory_id: str) -> None:
        """Remove ONE entry (e.g. on fade-without-reembed or hard_delete).
        A no-op if the id isn't present. Does not affect the built flag —
        see `put`'s docstring for why."""
        with self._lock:
            self._vectors.pop(memory_id, None)
