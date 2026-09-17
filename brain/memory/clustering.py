"""Numpy-only memory-vector clustering — Stage 5 (#157) of the local
semantic-retrieval build (companion-emergence).

Folds #157 into the semantic-retrieval PR per the spec's Resolved decision 2
(``~/.claude/plans/memory-dream-rework-semantic-retrieval-brief.md``): cluster
the persona's cached embedding vectors and expose cluster membership as a
MACHINE-USABLE retrieval tag (an integer cluster id, not a human-readable
topic label).

STORAGE (F1 #259 increment 4 — supersedes the OLD side-table design below):
the per-memory cluster tag lives as `cluster_id`/`cluster_model_id` columns
directly on the `memories` row, keyed by memory id, and per-cluster centroids
live in the `cluster_centroids` table — both inside `memories.db` (see
`brain/memory/store.py`'s `_SCHEMA`). This reverses the module's original
"NEVER a column on `memories`" hard constraint: that constraint turned out to
be an assistant-minted coordination heuristic that Roy never actually ruled
on (struck per the semantic-retrieval ledger's Q5 / the 2026-09-15
postmortem), and the invariant it was blocking — per-memory attributes live
on the memory row keyed by id (I2) — applies here same as anywhere else.
`run_clustering_pass` now sources its vectors from the warm `EmbeddingMatrix`
(`brain.memory.embedding_matrix.build_embedding_matrix(store.db_path)
.snapshot()`, itself backed by `memories.embedding`/`embedding_model_id`)
instead of `EmbeddingCache.all_hashes_and_vectors()`, and writes memberships
+ centroids via `MemoryStore.set_cluster_memberships` instead of
`MemoryClusterStore.replace_pass`. Identity key changes from content-hash to
memory-id as a result: two byte-identical memories no longer share one
cluster tag, each gets its own (intended, mirrors the same change embed-on-
write already made for the embedding column itself).

The OLD content-hash-keyed side table this module used to also define
(`MemoryClusterStore`, co-located in `embeddings.db` alongside
`embedding_cache`) has been dead since increment 4 and is REMOVED in the F1
#259 increment 8 code teardown, along with `embeddings.db`'s other code
paths — this module now only exposes the row/table-based storage described
above.

Scoping: still per-`model_id`, just implemented on the new storage.
`MemoryStore.set_cluster_memberships` wholesale-replaces every row tagged
with the target `model_id` each pass (mirroring `replace_pass`'s own
delete-then-reinsert symmetry — see that method's docstring), and
`cluster_tag_for_memory` only ever returns a `cluster_id` whose row-level
`cluster_model_id` matches the caller's current embedding model_id, so a
stale tag from a prior model is never served as if valid.

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
is a pure recompute-and-upsert against the current warm-matrix snapshot;
re-running it (including after a kill mid-write) always converges to a
membership consistent with the last COMPLETED pass, never a mix of two passes
— `MemoryStore.set_cluster_memberships` writes memberships + centroids inside
one transaction, committed once, so a kill mid-write leaves the row/table at
their PREVIOUS consistent state (SQLite rolls back an uncommitted transaction
on next open), not a half-applied one.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

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


@dataclass(frozen=True)
class ClusteringPassResult:
    """What one clustering pass accomplished — for logging/tests."""

    ran: bool  # False iff skipped (sparse-data floor not met)
    n_vectors: int  # vectors considered this pass
    k: int  # number of clusters used (0 when ran=False)
    reason: str  # "ok" | "sparse-skip"


def run_clustering_pass(
    store,  # brain.memory.store.MemoryStore — untyped to avoid a hard import here
    *,
    seed: int = DEFAULT_SEED,
    min_vectors: int = MIN_VECTORS_TO_CLUSTER,
) -> ClusteringPassResult:
    """Run one full clustering pass over `store`'s currently embedded
    vectors and upsert the result onto the `memories` row + the
    `cluster_centroids` table (F1 #259 increment 4).

    Vectors come from the warm `EmbeddingMatrix` for `store.db_path`
    (`brain.memory.embedding_matrix.build_embedding_matrix(store.db_path)
    .snapshot()` -> `{memory_id: vector}`), itself backed by the
    `memories.embedding`/`embedding_model_id` columns — NOT
    `EmbeddingCache.all_hashes_and_vectors()`/`embeddings.db` anymore. The
    matrix's own `model_id` (the model its currently-held vectors were
    embedded under) is what this pass's `cluster_id` writes get tagged
    with, so a tag always describes the model that actually produced the
    vectors clustered to produce it.

    The k-means algorithm itself (`choose_k`/`kmeans` above) is unchanged —
    only the vector source and the write destination moved.

    Dormant asymmetry (harmless today): writes are tagged with the warm
    matrix's own `model_id` (singleton-cached at first build — see
    `EmbeddingMatrix`), not a fresh `model_tier.model_for_tier(TIER_EMBEDDING)`
    lookup like other call sites use. Fine as long as nothing hot-reloads the
    embedding model mid-process (true today); it becomes a landmine only if
    that ever changes, since the matrix wouldn't notice the swap on its own.

    Off the message hot path by construction — callers only ever invoke this
    from a periodic background tick (see `_run_clustering_tick` in
    `brain/bridge/supervisor.py`), never from a chat-turn code path.

    Considers the ENTIRE model-scoped vector set every pass — no row cap.
    An earlier version applied a fixed `LIMIT` (`MAX_VECTORS_PER_PASS`) with
    no `ORDER BY` against `embedding_cache`, which SQLite serves in
    insertion order: once a persona's cache exceeded that cap, the SAME
    first-N-inserted rows were returned every pass and every later-embedded
    row was silently and PERMANENTLY excluded from clustering. Removed
    rather than replaced with a rotating/sampled bound: this is an
    off-hot-path background job on a ~6h supervisor cadence and companion
    corpora are bounded in size, so a full recompute every pass is
    affordable — the warm-matrix `snapshot()` this pass now reads has no
    such cap either.

    Sparse-data floor: fewer than `min_vectors` embedded vectors -> clean
    no-op (`ClusteringPassResult(ran=False, reason="sparse-skip")`), no
    write, no crash. Idempotent: re-running with the same embedded vectors +
    `seed` reproduces the same memberships/centroids and upserts them again
    (a strict no-op in effect, since `set_cluster_memberships` overwrites
    with identical values) — safe to call every cadence firing indefinitely.
    """
    from brain.memory.embedding_matrix import build_embedding_matrix

    matrix = build_embedding_matrix(store.db_path)
    vectors_by_id = matrix.snapshot()
    model_id = matrix.model_id
    n = len(vectors_by_id)

    if n < min_vectors:
        logger.info(
            "clustering: skipping pass (%d embedded vectors < floor %d) model_id=%s",
            n,
            min_vectors,
            model_id,
        )
        return ClusteringPassResult(ran=False, n_vectors=n, k=0, reason="sparse-skip")

    memory_ids = list(vectors_by_id.keys())
    vectors = np.stack([vectors_by_id[mid] for mid in memory_ids])
    k = choose_k(n)
    labels, centroids = kmeans(vectors, k, seed=seed)

    memberships = {
        mid: int(label) for mid, label in zip(memory_ids, labels, strict=True)
    }
    store.set_cluster_memberships(memberships, centroids, model_id=model_id)

    logger.info(
        "clustering: pass complete n_vectors=%d k=%d model_id=%s", n, k, model_id
    )
    return ClusteringPassResult(ran=True, n_vectors=n, k=k, reason="ok")


def cluster_tag_for_memory(
    memory_id: str,
    *,
    store,  # brain.memory.store.MemoryStore — untyped to avoid a hard import here
) -> int | None:
    """The query accessor Stage 3+ retrieval can later use: given a memory
    id, read its `cluster_id`/`cluster_model_id` straight off the `memories`
    row (`MemoryStore.get_cluster_id` — a raw row read, no recall bump, same
    `bump=False` care as the `_EmbeddingsByMemoryId` precedent in
    `brain/bridge/supervisor.py`) and return the tag only if it was written
    under the caller's CURRENT embedding model_id
    (`model_tier.model_for_tier(TIER_EMBEDDING)`, looked up via the module
    so a test's monkeypatch on `model_for_tier` is honored, mirroring every
    other dynamic model_id lookup in this codebase).

    Returns `None` when the memory doesn't exist, hasn't been clustered yet,
    or was only clustered under a prior model_id (never serves a stale tag).
    NOT wired into the retrieval path itself — that is a separate, later
    stage; this only makes the tag queryable.
    """
    from brain.bridge import model_tier

    row = store.get_cluster_id(memory_id)
    if row is None:
        return None
    cluster_id, cluster_model_id = row
    current_model_id = model_tier.model_for_tier(model_tier.TIER_EMBEDDING)
    if cluster_model_id != current_model_id:
        return None
    return cluster_id
