"""Numpy-only memory-vector clustering — Stage 5 (#157) of the local
semantic-retrieval build (companion-emergence).

Folds #157 into the semantic-retrieval PR per the spec's Resolved decision 2
(``~/.claude/plans/memory-dream-rework-semantic-retrieval-brief.md``): cluster
the persona's cached embedding vectors and expose cluster membership as a
MACHINE-USABLE retrieval tag (an integer cluster id, not a human-readable
topic label).

HARD CONSTRAINT (spec decision 2, re-confirmed in
``hunts/semantic-retrieval/plan.md`` Stage 5): cluster data lives in its OWN
content-hash-keyed side table (``MemoryClusterStore`` below), co-located in
``embeddings.db`` alongside ``embedding_cache`` (both are content-hash-keyed,
vector-derived data — the plan's own recommendation) but as separate tables.
Cluster data is NEVER a column on the `memories` table in `memories.db` — that
bolt-on is the one design that would collide with a future emotions
row-migration; the side-table design keeps this work disjoint from `memories`,
`hebbian.db`, and any emotions cleanup.

Scoping: every row carries the `model_id` that produced the clustered vectors
(mirrors `embedding_cache`'s own `(content_hash, model_id)` scoping — see that
module's docstring). `cluster_for`/`cluster_tag_for_content` only ever return
a row whose `model_id` matches the caller's current embedding provider, so a
stale row is never served as if valid. Two ways staleness is avoided, mirroring
`EmbeddingCache.get_or_compute`'s own INSERT-OR-REPLACE behavior exactly:
content NOT YET reclustered under a new model has no row scoped to that
model_id at all (invisible until the next pass computes one); content THAT HAS
been reclustered has its row's `model_id`/`cluster_id` overwritten in place
(`content_hash` is the row's natural key — the same text hashes identically
regardless of which model embedded it, so there is only ever one row per
content_hash, pointing at whichever model most recently clustered it) — a
lookup still scoped to the OLD model_id on that content_hash then correctly
returns nothing, not a mix of old and new.

Runs as a periodic BATCH pass (own persisted supervisor cadence — see
`_run_clustering_tick` in `brain/bridge/supervisor.py`), never on the message
hot path — same posture as Stage 2's idle-chipped embedding backfill, though
the mechanics differ: this is a single full recompute over the ENTIRE
model-scoped vector set per firing (no row cap — a from-scratch k-means pass
needs the whole candidate pool at once to produce meaningful clusters, and a
fixed-size cap would silently and permanently starve any row embedded after
the cap was first reached), not a chipped per-row batch. Clustering is an
off-hot-path background job on a ~6h supervisor cadence and companion corpora
are bounded in size, so a full recompute every pass is affordable.

Graceful with sparse data: below `MIN_VECTORS_TO_CLUSTER` cached vectors, a
pass is a clean no-op (`ClusteringPassResult.ran=False`) — no crash, no
degenerate single-cluster write. Idempotent/resumable: `run_clustering_pass`
is a pure recompute-and-upsert against the current `embedding_cache` snapshot;
re-running it (including after a kill mid-write) always converges to a
membership consistent with the last COMPLETED pass, never a mix of two passes
— `MemoryClusterStore.replace_pass` writes memberships + centroids inside one
transaction, committed once, so a kill mid-write leaves the table at its
PREVIOUS consistent state (SQLite rolls back an uncommitted transaction on
next open), not a half-applied one.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain.memory.embeddings import EmbeddingCache, hash_content

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables — documented defaults (plan/spec don't pin these; see build report
# for the reasoning), not derived from any measurement.
# ---------------------------------------------------------------------------

# Below this many embedded vectors, clustering is skipped cleanly: too few
# points for a k-means pass to produce anything meaningful, and the "graceful
# with sparse data" constraint explicitly calls for a clean skip rather than a
# garbage single-cluster result. A fresh/small persona simply has no cluster
# tags yet, same posture as the embedding backfill's own warm-up story.
MIN_VECTORS_TO_CLUSTER = 8

# Bounds on k (number of clusters), applied after the sqrt(n/2) heuristic
# below. K_MIN=2 because "1 cluster" is not clustering. K_MAX=50 keeps a
# single pass's compute bounded even against a very large corpus.
K_MIN = 2
K_MAX = 50

DEFAULT_SEED = 0
DEFAULT_MAX_ITER = 100


def choose_k(n: int) -> int:
    """Default k heuristic: k ~= sqrt(n / 2), bounded to [K_MIN, K_MAX] and
    never >= n (k-means requires k <= n; k == n is degenerate — one point per
    cluster — so k is also capped at n - 1 whenever that is >= K_MIN).

    Documented default per the build task's own instruction ("pick the
    default and report it... so I can confirm it with the spec owner") — the
    spec does not pin a choice-of-k rule.
    """
    if n < 1:
        raise ValueError("choose_k requires n >= 1")
    if n == 1:
        return 1  # degenerate: a single vector can't form >=2 clusters
    raw = round(math.sqrt(n / 2))
    k = max(K_MIN, min(K_MAX, raw))
    return min(k, n - 1)


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between every row of `a` (n, d) and every
    row of `b` (k, d) -> (n, k). Standard expansion, avoids an (n, k, d)
    temporary."""
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)


def _kmeans_plus_plus_init(vectors: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: first centroid uniform-random, each subsequent
    centroid drawn with probability proportional to its squared distance from
    the nearest already-chosen centroid. Deterministic given `rng` (a seeded
    Generator) and the input vectors' order.
    """
    n = vectors.shape[0]
    first = int(rng.integers(0, n))
    chosen = [first]
    closest_sq = np.full(n, np.inf)

    for _ in range(1, k):
        last = vectors[chosen[-1]]
        diff = vectors - last
        d2 = np.sum(diff * diff, axis=1)
        closest_sq = np.minimum(closest_sq, d2)
        total = float(closest_sq.sum())
        if total <= 0.0:
            # Every remaining point coincides with an already-chosen
            # centroid (e.g. duplicate vectors) — fall back to uniform
            # choice so seeding still terminates deterministically.
            idx = int(rng.integers(0, n))
        else:
            probs = closest_sq / total
            idx = int(rng.choice(n, p=probs))
        chosen.append(idx)

    return vectors[np.asarray(chosen)].copy()


def kmeans(
    vectors: np.ndarray,
    k: int,
    *,
    seed: int = DEFAULT_SEED,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic-seeded k-means (Lloyd's algorithm, k-means++ init).

    numpy-only — no scikit-learn/scipy dependency (hard constraint: no new
    heavy clustering library). Given the same `vectors` (in the same order),
    `k`, and `seed`, this always produces the same `labels`/`centroids` —
    k-means++ init draws from a seeded `np.random.default_rng(seed)`, and the
    Lloyd's-algorithm update loop that follows is itself fully deterministic
    (no randomness once centroids are seeded).

    Returns `(labels, centroids)`: `labels` is an `(n,)` int array in
    `[0, k)`; `centroids` is a `(k, dim)` float64 array (the mean of each
    cluster's current members — an empty cluster, however unlikely after
    k-means++ seeding, simply keeps its previous centroid rather than being
    reseeded, so the loop always terminates).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n = vectors.shape[0]
    if k > n:
        raise ValueError(f"k ({k}) must be <= number of vectors ({n})")

    vectors = np.asarray(vectors, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centroids = _kmeans_plus_plus_init(vectors, k, rng)
    labels = np.full(n, -1, dtype=np.int64)

    for _ in range(max_iter):
        dists = _pairwise_sq_dists(vectors, centroids)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centroids = centroids.copy()
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                new_centroids[c] = vectors[mask].mean(axis=0)
            # else: empty cluster keeps its previous centroid (no reseed) —
            # a documented, deterministic tie-break rather than a special
            # case that could loop forever chasing a perfectly balanced k.
        centroids = new_centroids

    return labels, centroids


class MemoryClusterStore:
    """Content-hash-keyed side table for memory cluster membership.

    OWN table (`memory_clusters` + `memory_cluster_centroids`), never a
    column on `memories` — see module docstring's hard constraint. Mirrors
    `EmbeddingCache`'s schema/pragma/ALTER-guard shape so the two side tables
    read as one family, but is its own class with its own connection (a
    genuinely separate side table, not a method bolted onto EmbeddingCache).

    `memory_clusters` holds one row per `content_hash`: the cluster id that
    content currently belongs to, and the `model_id` of the embedding that
    produced it. `memory_cluster_centroids` holds each cluster's centroid
    vector, keyed by `(model_id, cluster_id)` — kept for future retrieval use
    (e.g. "nearest cluster to a query vector") though Stage 5 itself only
    needs to expose membership.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS memory_clusters (
        content_hash TEXT PRIMARY KEY,
        model_id TEXT NOT NULL DEFAULT '',
        cluster_id INTEGER NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_memory_clusters_model
        ON memory_clusters(model_id);
    CREATE TABLE IF NOT EXISTS memory_cluster_centroids (
        model_id TEXT NOT NULL,
        cluster_id INTEGER NOT NULL,
        centroid BLOB NOT NULL,
        dim INTEGER NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (model_id, cluster_id)
    );
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        # WAL + busy_timeout mirrors EmbeddingCache — the supervisor's
        # periodic clustering tick and any concurrent reader (e.g. a future
        # retrieval-path query) share this file.
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def cluster_for(self, content_hash: str, *, model_id: str) -> int | None:
        """The cluster id for `content_hash`, scoped to `model_id`. `None`
        when unclustered OR when the only row on file was written under a
        DIFFERENT model_id (stale — never served)."""
        row = self._conn.execute(
            "SELECT cluster_id FROM memory_clusters WHERE content_hash = ? AND model_id = ?",
            (content_hash, model_id),
        ).fetchone()
        return int(row[0]) if row is not None else None

    def cluster_for_content(self, content: str, *, model_id: str) -> int | None:
        """Convenience wrapper: hash `content` the same way `embedding_cache`
        does, then look up its cluster tag."""
        return self.cluster_for(hash_content(content), model_id=model_id)

    def centroids(self, *, model_id: str) -> dict[int, np.ndarray]:
        """`{cluster_id: centroid_vector}` for `model_id`'s current clusters."""
        rows = self._conn.execute(
            "SELECT cluster_id, centroid, dim FROM memory_cluster_centroids WHERE model_id = ?",
            (model_id,),
        ).fetchall()
        return {
            int(cid): np.frombuffer(blob, dtype=np.float32).copy().reshape(dim)
            for cid, blob, dim in rows
        }

    def count(self, *, model_id: str | None = None) -> int:
        """Number of clustered rows, optionally scoped to `model_id`."""
        if model_id is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM memory_clusters").fetchone()[0])
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM memory_clusters WHERE model_id = ?", (model_id,)
            ).fetchone()[0]
        )

    def replace_pass(
        self,
        memberships: dict[str, int],
        centroids: np.ndarray,
        *,
        model_id: str,
    ) -> None:
        """Atomically replace `model_id`'s cluster memberships + centroids
        with the result of one clustering pass.

        ONE transaction, ONE commit — a process kill mid-write leaves this
        table exactly as it was after the LAST successfully committed pass
        (SQLite rolls back an uncommitted transaction on next open), never a
        mix of old and new memberships/centroids. This is what makes
        `run_clustering_pass` idempotent/resumable: re-running it after a
        kill just redoes the whole (cheap, local, numpy-only) computation and
        writes it in one more atomic replace.

        `memory_clusters` is WHOLESALE-REPLACED for `model_id`, exactly
        mirroring `memory_cluster_centroids` below: every existing row for
        this `model_id` is deleted, then this pass's `memberships` are
        (re)inserted. A content_hash present in a PRIOR pass but absent from
        `memberships` (e.g. its pool composition changed — it fell out of the
        embedding cache via `EmbeddingCache.evict()`, or simply wasn't part
        of this pass's candidate pool) ends up with NO row at all, not a
        dangling one: `cluster_for()`/`cluster_tag_for_memory()` then
        correctly return `None` for it instead of a stale `cluster_id` that
        points at a centroid this pass just deleted. (An earlier version
        upserted memberships without ever deleting — asymmetric against the
        centroid table's delete-then-reinsert below — so a content_hash that
        dropped out of the pool kept its old `cluster_id` pointing at a
        centroid row that no longer existed.) A content_hash present in BOTH
        the prior and current pass still gets its `model_id`/`cluster_id`
        overwritten in place via the reinsert (`content_hash` is the PRIMARY
        KEY — one row per content, always pointing at whichever model most
        recently clustered it, exactly mirroring
        `EmbeddingCache.get_or_compute`'s own INSERT-OR-REPLACE-by-
        content_hash behavior on a model swap): a lookup still scoped to that
        content's PRIOR model_id then correctly finds nothing, rather than a
        stale tag.
        """
        self._conn.execute(
            "DELETE FROM memory_clusters WHERE model_id = ?", (model_id,)
        )
        self._conn.executemany(
            "INSERT INTO memory_clusters (content_hash, model_id, cluster_id, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(content_hash) DO UPDATE SET "
            "model_id = excluded.model_id, "
            "cluster_id = excluded.cluster_id, "
            "updated_at = excluded.updated_at",
            [(content_hash, model_id, cluster_id) for content_hash, cluster_id in memberships.items()],
        )
        self._conn.execute(
            "DELETE FROM memory_cluster_centroids WHERE model_id = ?", (model_id,)
        )
        self._conn.executemany(
            "INSERT INTO memory_cluster_centroids (model_id, cluster_id, centroid, dim) "
            "VALUES (?, ?, ?, ?)",
            [
                (model_id, i, centroid.astype(np.float32).tobytes(), centroid.shape[0])
                for i, centroid in enumerate(centroids)
            ],
        )
        self._conn.commit()


@dataclass(frozen=True)
class ClusteringPassResult:
    """What one clustering pass accomplished — for logging/tests."""

    ran: bool  # False iff skipped (sparse-data floor not met)
    n_vectors: int  # vectors considered this pass
    k: int  # number of clusters used (0 when ran=False)
    reason: str  # "ok" | "sparse-skip"


def run_clustering_pass(
    embeddings: EmbeddingCache,
    cluster_store: MemoryClusterStore,
    *,
    seed: int = DEFAULT_SEED,
    min_vectors: int = MIN_VECTORS_TO_CLUSTER,
) -> ClusteringPassResult:
    """Run one full clustering pass over `embeddings`' currently cached
    vectors (scoped to `embeddings.model_id`) and upsert the result into
    `cluster_store`.

    Off the message hot path by construction — callers only ever invoke this
    from a periodic background tick (see `_run_clustering_tick` in
    `brain/bridge/supervisor.py`), never from a chat-turn code path.

    Considers the ENTIRE model-scoped vector set every pass — no row cap.
    An earlier version applied a fixed `LIMIT` (`MAX_VECTORS_PER_PASS`) with
    no `ORDER BY`, which SQLite serves in insertion order: once a persona's
    embedding_cache exceeded that cap, the SAME first-N-inserted rows were
    returned every pass and every later-embedded row was silently and
    PERMANENTLY excluded from clustering. Removed rather than replaced with a
    rotating/sampled bound: this is an off-hot-path background job on a ~6h
    supervisor cadence and companion corpora are bounded in size, so a full
    recompute every pass is affordable.

    Sparse-data floor: fewer than `min_vectors` cached vectors -> clean no-op
    (`ClusteringPassResult(ran=False, reason="sparse-skip")`), no write, no
    crash. Idempotent: re-running with the same cached vectors + `seed`
    reproduces the same memberships/centroids and upserts them again (a
    strict no-op in effect, since `replace_pass` overwrites with identical
    values) — safe to call every cadence firing indefinitely.
    """
    model_id = embeddings.model_id
    pairs = embeddings.all_hashes_and_vectors()
    n = len(pairs)

    if n < min_vectors:
        logger.info(
            "clustering: skipping pass (%d cached vectors < floor %d) model_id=%s",
            n,
            min_vectors,
            model_id,
        )
        return ClusteringPassResult(ran=False, n_vectors=n, k=0, reason="sparse-skip")

    hashes = [content_hash for content_hash, _ in pairs]
    vectors = np.stack([vector for _, vector in pairs])
    k = choose_k(n)
    labels, centroids = kmeans(vectors, k, seed=seed)

    memberships = {h: int(label) for h, label in zip(hashes, labels, strict=True)}
    cluster_store.replace_pass(memberships, centroids, model_id=model_id)

    logger.info(
        "clustering: pass complete n_vectors=%d k=%d model_id=%s", n, k, model_id
    )
    return ClusteringPassResult(ran=True, n_vectors=n, k=k, reason="ok")


def cluster_tag_for_memory(
    memory_id: str,
    *,
    store,  # brain.memory.store.MemoryStore — untyped to avoid a hard import here
    embeddings: EmbeddingCache,
    cluster_store: MemoryClusterStore,
) -> int | None:
    """The query accessor Stage 3+ retrieval can later use: given a memory
    id, resolve its content (WITHOUT bumping recall — this is a metadata
    lookup, not a surfacing event, same `bump=False` care as the
    `_EmbeddingsByMemoryId` precedent in `brain/bridge/supervisor.py`), hash
    it, and look up its cluster tag scoped to `embeddings`' current
    model_id.

    Returns `None` when the memory doesn't exist, hasn't been embedded yet,
    hasn't been clustered yet, or was only clustered under a prior model_id
    (never serves a stale tag). NOT wired into the retrieval path itself —
    that is a separate, later stage; this only makes the tag queryable.
    """
    memory = store.get(memory_id, bump=False)
    if memory is None:
        return None
    return cluster_store.cluster_for_content(memory.content, model_id=embeddings.model_id)
