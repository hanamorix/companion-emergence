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

    SELECT id, embedding FROM memories
    WHERE active = 1 AND embedding IS NOT NULL AND embedding_model_id = ?

not `store.list_active()` / `_row_to_memory` — decoding every JSON field
(emotions, tags, metadata) on every row just to throw it away and keep the
vector would be pure waste at corpus scale. The `embedding_model_id = ?`
filter loads ONLY the current model's vectors: during a model swap a rebuild
must not pick up rows still stamped with the old model and re-serve them
under the new model's label.

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
dict assignment.

Reconciling concurrent writes across a build (the lost-update / resurrection
hazard): because `_load_from_db()` runs OFF-lock, a `put`/`evict` can land
after the SELECT snapshotted the DB but before the swap. A wholesale
`self._vectors = loaded` would silently discard that write — a `put` would
be lost (recall degrades to lexical for a row whose vector is durably on
disk) and an `evict` would be undone (a deleted memory resurrected in the
matrix). To prevent that, a build records every mutation that lands while it
is in flight into a PENDING OVERLAY (`{id: vector-or-TOMBSTONE}`); at swap
time it applies the overlay ON TOP of the freshly-loaded dict — puts win,
tombstones delete — before publishing. `put`/`evict` update BOTH the live
matrix (so concurrent readers see the change immediately) AND the overlay
(so the swap cannot lose it). A depth counter lets a lazy build and a
rebuild overlap without corrupting the overlay: it is opened when the first
build starts recording and cleared only when the last in-flight build
finishes.

`model_id` tracks which embedding model produced the vectors currently held
(set at construction, updated by `rebuild(model_id=...)`). Nothing in this
module compares it against the live provider's model id yet — that
comparison, and the decision to trigger a rebuild on a model-id swap or
process restart, is a later increment's job. `rebuild()` is exposed now so
that wiring has something to call.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Retrieval embeddings are 384-dim float32 (see the backfill/provider). A
# blob is 384 * 4 = 1536 bytes; anything else is a corrupt/short row that
# must be skipped rather than allowed to crash the whole build.
_EXPECTED_DIM = 384
_EXPECTED_BYTES = _EXPECTED_DIM * 4

# Sentinel recorded in the pending overlay to mean "this id was evicted while
# a build was in flight" — distinct from an absent key (no mutation) and from
# a vector (a put). At swap time a tombstone deletes the id from the freshly
# loaded dict so a concurrent evict is never undone by the rebuild.
_TOMBSTONE = object()


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
        # Reconciliation state for the off-lock build window (see module
        # docstring). `_pending` records mutations that land while a build is
        # in flight; `_build_depth` counts overlapping builds so the overlay
        # survives until the LAST one swaps.
        self._pending: dict[str, object] = {}
        self._build_depth = 0

    # -- lifecycle -----------------------------------------------------

    def ensure_built(self) -> None:
        """Build the matrix if it hasn't been built yet. Idempotent and
        safe to call from multiple threads — a race on the very first
        access may cause more than one thread to run the (read-only) scan
        redundantly, but each computes the same result off-lock, any
        mutation that lands during the scan is captured by the pending
        overlay and re-applied at swap time, and only the final swap is
        locked, so no torn or partial state is ever observable by a reader.
        """
        self._run_build(model_id=None, force=False)

    def rebuild(self, *, model_id: str | None = None) -> None:
        """Full rebuild: reload every active embedded row from disk into a
        NEW dict, then atomically swap it in. Used on a model-id swap
        (pass the new `model_id`) or a process restart (call with no
        argument to reload under the same model_id).

        The slow part (the DB scan) happens BEFORE the lock is acquired —
        readers keep being served the OLD matrix for the full duration of
        the scan and only block for the instant of the reference swap. Any
        `put`/`evict` that lands during the scan is recorded in the pending
        overlay and re-applied on top of the freshly-loaded dict at swap
        time, so a concurrent write is never lost and a concurrent evict is
        never resurrected.
        """
        self._run_build(model_id=model_id, force=True)

    def _run_build(self, *, model_id: str | None, force: bool) -> None:
        """Shared build/rebuild engine.

        `force=False` (lazy build) returns early if the matrix is already
        built. `force=True` (rebuild) always reloads. The DB scan runs
        off-lock; the pending overlay opened here captures every concurrent
        mutation and is applied at swap time. `_build_depth` lets a lazy
        build and a rebuild overlap safely — the overlay is cleared only
        when the last in-flight build completes.
        """
        with self._lock:
            if not force and self._built:
                return
            self._build_depth += 1
            target_model = model_id if model_id is not None else self.model_id

        try:
            loaded = self._load_from_db(target_model)
        except BaseException:
            with self._lock:
                self._build_depth -= 1
                if self._build_depth == 0:
                    self._pending = {}
            raise

        with self._lock:
            # Apply the overlay of writes that landed during the off-lock
            # scan ON TOP of the freshly loaded dict: puts win over the
            # stale DB snapshot, tombstones delete resurrected rows.
            for mem_id, entry in self._pending.items():
                if entry is _TOMBSTONE:
                    loaded.pop(mem_id, None)
                else:
                    loaded[mem_id] = entry  # type: ignore[assignment]
            self._vectors = loaded
            self._built = True
            if model_id is not None:
                self.model_id = model_id
            self._build_depth -= 1
            if self._build_depth == 0:
                self._pending = {}

    def _load_from_db(self, model_id: str) -> dict[str, np.ndarray]:
        """The lean lazy-build query. Opens and closes its own short-lived
        connection rather than sharing a `MemoryStore`'s connection — this
        runs on whatever thread triggers a build/rebuild (request thread
        for the first lazy access, supervisor thread for a maintenance
        rebuild), and sqlite3 connections are not safe to share across
        threads without extra coordination the rest of this module doesn't
        need.

        Filters to the target `model_id` so a rebuild during a model swap
        loads only the current model's vectors. Decodes each row
        defensively: a corrupt or wrong-length blob is skipped-and-logged,
        never allowed to throw and kill the whole build (which would
        propagate into recall via the request-thread lazy build).
        """
        conn = sqlite3.connect(str(self._db_path))
        try:
            # Mirror MemoryStore's 5s busy_timeout (WAL is already on the
            # DB) so a concurrent writer does not surface "database is
            # locked" on the default busy_timeout=0.
            conn.execute("PRAGMA busy_timeout = 5000")
            rows = conn.execute(
                "SELECT id, embedding FROM memories"
                " WHERE active = 1 AND embedding IS NOT NULL"
                " AND embedding_model_id = ?",
                (model_id,),
            ).fetchall()
        finally:
            conn.close()

        result: dict[str, np.ndarray] = {}
        for row in rows:
            mem_id, blob = row[0], row[1]
            if blob is None or len(blob) != _EXPECTED_BYTES:
                logger.warning(
                    "embedding_matrix: skipping row %s with bad embedding blob"
                    " (expected %d bytes, got %s)",
                    mem_id,
                    _EXPECTED_BYTES,
                    "None" if blob is None else len(blob),
                )
                continue
            try:
                vec = np.frombuffer(blob, dtype=np.float32).copy()
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "embedding_matrix: skipping row %s, undecodable embedding: %s",
                    mem_id,
                    exc,
                )
                continue
            if vec.shape != (_EXPECTED_DIM,):
                logger.warning(
                    "embedding_matrix: skipping row %s with wrong embedding dim"
                    " (expected %d, got %d)",
                    mem_id,
                    _EXPECTED_DIM,
                    vec.shape[0],
                )
                continue
            result[mem_id] = vec
        return result

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

        If a build is in flight, the entry is ALSO recorded in the pending
        overlay so the build's swap re-applies it instead of discarding it
        (the lost-update fix).

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
            if self._build_depth > 0:
                self._pending[memory_id] = vec

    def evict(self, memory_id: str) -> None:
        """Remove ONE entry (e.g. on fade-without-reembed or hard_delete).
        A no-op on the live matrix if the id isn't present. Does not affect
        the built flag — see `put`'s docstring for why.

        If a build is in flight, a TOMBSTONE is recorded in the pending
        overlay so the build's swap re-applies the deletion instead of
        resurrecting the row from the stale DB snapshot (the resurrection
        fix)."""
        with self._lock:
            self._vectors.pop(memory_id, None)
            if self._build_depth > 0:
                self._pending[memory_id] = _TOMBSTONE
