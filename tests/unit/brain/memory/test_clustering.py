"""Tests for brain.memory.clustering — Stage 5 (#157) of the local
semantic-retrieval build: numpy-only k-means over cached embedding vectors,
exposed as a machine-usable retrieval tag via a content-hash-keyed side
table.

Covers the build task's acceptance bar: stable membership on a seeded
fixture, content-hash+model_id-keyed side table, model_id swap recomputes
rather than serving stale tags, sparse-data skip, idempotent re-run, and the
hard "never a `memories`-table column" guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brain.memory.clustering import (
    K_MAX,
    K_MIN,
    MIN_VECTORS_TO_CLUSTER,
    MemoryClusterStore,
    choose_k,
    cluster_tag_for_memory,
    kmeans,
    run_clustering_pass,
)
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.store import Memory, MemoryStore

# ---------------------------------------------------------------------------
# choose_k heuristic
# ---------------------------------------------------------------------------


def test_choose_k_is_bounded_and_below_n() -> None:
    for n in (1, 2, 8, 50, 500, 5000):
        k = choose_k(n)
        assert 1 <= k <= n
        assert K_MIN <= k or k == 1  # k==1 only for the degenerate n<=2 case
        assert k <= K_MAX


def test_choose_k_matches_sqrt_n_over_2_heuristic_in_the_middle_range() -> None:
    """Sanity: for n where the heuristic isn't clipped by either bound, k is
    exactly round(sqrt(n/2)) — the documented default."""
    n = 200  # round(sqrt(100)) == 10, well inside [K_MIN, K_MAX]
    assert choose_k(n) == 10


def test_choose_k_never_reaches_or_exceeds_n() -> None:
    """k == n would be one point per cluster — degenerate, explicitly
    disallowed (see module docstring)."""
    for n in range(1, 30):
        assert choose_k(n) < n or n == 1


def test_choose_k_raises_on_non_positive_n() -> None:
    with pytest.raises(ValueError):
        choose_k(0)


# ---------------------------------------------------------------------------
# kmeans — deterministic seeded numpy k-means
# ---------------------------------------------------------------------------


def _three_blob_fixture(seed: int = 1234) -> np.ndarray:
    """12 points in 3 well-separated 4-D blobs (4 points each), small noise.
    A fixed synthetic fixture — same call always returns the same array."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0, 0.0],
            [0.0, 50.0, 0.0, 0.0],
        ]
    )
    points = []
    for c in centers:
        points.append(c + rng.normal(scale=0.5, size=(4, 4)))
    return np.vstack(points)


def test_kmeans_produces_stable_membership_on_a_seeded_fixture() -> None:
    """Same vectors + same seed -> identical labels/centroids across two
    independent calls. This is the load-bearing determinism guarantee the
    build task calls out explicitly."""
    vectors = _three_blob_fixture()
    labels_a, centroids_a = kmeans(vectors, k=3, seed=42)
    labels_b, centroids_b = kmeans(vectors, k=3, seed=42)

    np.testing.assert_array_equal(labels_a, labels_b)
    np.testing.assert_allclose(centroids_a, centroids_b)


def test_kmeans_separates_well_separated_blobs_correctly() -> None:
    """Not just stable — actually correct: each 4-point blob ends up in its
    own cluster (all points within a blob share a label; different blobs get
    different labels)."""
    vectors = _three_blob_fixture()
    labels, _centroids = kmeans(vectors, k=3, seed=7)

    blob_labels = [set(labels[i : i + 4]) for i in (0, 4, 8)]
    for bl in blob_labels:
        assert len(bl) == 1  # every point in a blob shares one label
    assert len({next(iter(bl)) for bl in blob_labels}) == 3  # 3 distinct labels


def test_kmeans_labels_and_centroids_shapes() -> None:
    vectors = _three_blob_fixture()
    labels, centroids = kmeans(vectors, k=3, seed=0)
    assert labels.shape == (12,)
    assert centroids.shape == (3, 4)
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}


def test_kmeans_rejects_k_greater_than_n() -> None:
    vectors = _three_blob_fixture()
    with pytest.raises(ValueError):
        kmeans(vectors, k=13, seed=0)


def test_kmeans_rejects_k_less_than_one() -> None:
    vectors = _three_blob_fixture()
    with pytest.raises(ValueError):
        kmeans(vectors, k=0, seed=0)


def test_kmeans_k_equals_1_puts_everything_in_one_cluster() -> None:
    vectors = _three_blob_fixture()
    labels, centroids = kmeans(vectors, k=1, seed=0)
    assert set(np.unique(labels).tolist()) == {0}
    assert centroids.shape == (1, 4)


# ---------------------------------------------------------------------------
# MemoryClusterStore — content-hash + model_id keyed side table
# ---------------------------------------------------------------------------


def test_store_schema_never_touches_memories_table(tmp_path: Path) -> None:
    """Hard constraint guard: opening a MemoryClusterStore (even against the
    SAME file a MemoryStore could use) never creates/touches a `memories`
    table or a `cluster_id` column anywhere. The store lives in its own
    file/table pair; this also documents that its schema is fully disjoint
    from MemoryStore's."""
    db_path = tmp_path / "embeddings.db"
    cluster_store = MemoryClusterStore(db_path)
    try:
        tables = {
            row[0]
            for row in cluster_store._conn.execute(  # noqa: SLF001 — test-only introspection
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "memories" not in tables
        assert {"memory_clusters", "memory_cluster_centroids"} <= tables

        cols = {
            row[1]
            for row in cluster_store._conn.execute(  # noqa: SLF001
                "PRAGMA table_info(memory_clusters)"
            ).fetchall()
        }
        assert "cluster_id" in cols  # on the SIDE table, exactly as designed
    finally:
        cluster_store.close()

    # And a real MemoryStore's own `memories` table schema is untouched by
    # any of this (same assertion the plan's Stage 5 verification calls for,
    # run against the actual memories.db this persona would use).
    store = MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)
    try:
        cols = {
            row[1]
            for row in store._conn.execute("PRAGMA table_info(memories)").fetchall()  # noqa: SLF001
        }
        assert "cluster_id" not in cols
    finally:
        store.close()


def test_cluster_for_content_queryable_by_content_hash(tmp_path: Path) -> None:
    store = MemoryClusterStore(tmp_path / "embeddings.db")
    try:
        store.replace_pass(
            {"hash-a": 0, "hash-b": 1}, np.zeros((2, 4)), model_id="model-x"
        )
        assert store.cluster_for("hash-a", model_id="model-x") == 0
        assert store.cluster_for("hash-b", model_id="model-x") == 1
        assert store.cluster_for("hash-c", model_id="model-x") is None  # never written
    finally:
        store.close()


def test_cluster_for_content_hashes_the_same_way_as_embedding_cache(tmp_path: Path) -> None:
    """cluster_for_content must key rows identically to how EmbeddingCache
    hashes content, so a memory's content resolves to the same row either
    way."""
    from brain.memory.embeddings import hash_content

    store = MemoryClusterStore(tmp_path / "embeddings.db")
    try:
        h = hash_content("some memory content")
        store.replace_pass({h: 5}, np.zeros((1, 4)), model_id="model-x")
        assert store.cluster_for_content("some memory content", model_id="model-x") == 5
    finally:
        store.close()


def test_model_id_scoping_a_different_model_never_sees_the_others_rows(tmp_path: Path) -> None:
    """Hard invariant: cluster tags are model_id-scoped exactly like
    embedding_cache. A row written under one model_id must be invisible to a
    lookup scoped to a different model_id."""
    store = MemoryClusterStore(tmp_path / "embeddings.db")
    try:
        store.replace_pass({"hash-a": 0}, np.zeros((1, 4)), model_id="model-old")
        assert store.cluster_for("hash-a", model_id="model-old") == 0
        assert store.cluster_for("hash-a", model_id="model-new") is None
    finally:
        store.close()


def test_model_swap_recomputes_rather_than_serving_stale_tags(tmp_path: Path) -> None:
    """A model swap must not silently serve a cluster tag computed under a
    stale model's vector space — it must recompute (or stay absent) under the
    new model_id."""
    cache_old = EmbeddingCache(tmp_path / "embeddings.db", FakeEmbeddingProvider(dim=16))
    cluster_store = MemoryClusterStore(tmp_path / "embeddings.db")
    try:
        texts = [f"memory content number {i} long enough to embed" for i in range(10)]
        for t in texts:
            cache_old.get_or_compute(t)
        result_old = run_clustering_pass(cache_old, cluster_store, seed=1)
        assert result_old.ran is True
        old_model_id = cache_old.model_id
        for t in texts:
            assert cluster_store.cluster_for_content(t, model_id=old_model_id) is not None

        # Swap to a different model (different dim -> different model_id,
        # same as the real FastEmbedProvider dim-migration story).
        cache_new = EmbeddingCache(tmp_path / "embeddings.db", FakeEmbeddingProvider(dim=32))
        new_model_id = cache_new.model_id
        assert new_model_id != old_model_id

        # Before the new model has embedded/clustered anything, old tags must
        # NEVER be served under the new model_id.
        for t in texts:
            assert cluster_store.cluster_for_content(t, model_id=new_model_id) is None

        for t in texts:
            cache_new.get_or_compute(t)
        result_new = run_clustering_pass(cache_new, cluster_store, seed=1)
        assert result_new.ran is True
        for t in texts:
            assert cluster_store.cluster_for_content(t, model_id=new_model_id) is not None
        # The row for each content_hash is now overwritten to point at the
        # NEW model (content_hash is the row's natural key — the same text
        # hashes identically regardless of which model embedded it, exactly
        # mirroring EmbeddingCache.get_or_compute's own INSERT-OR-REPLACE
        # behavior on a model swap). A query still scoped to the OLD
        # model_id now correctly returns None — the targeted invalidation
        # the hard invariant calls for: never serve a stale tag.
        for t in texts:
            assert cluster_store.cluster_for_content(t, model_id=old_model_id) is None
        cache_new.close()
    finally:
        cache_old.close()
        cluster_store.close()


def test_replace_pass_is_atomic_a_failed_write_leaves_prior_state_intact(
    tmp_path: Path,
) -> None:
    """Simulates a crash mid-write: replace_pass must not leave a partial
    (memberships-without-centroids) state behind. `None` in place of a
    centroid ndarray raises a genuine ``AttributeError`` on ``.astype(...)``
    partway through the SECOND (centroids) write, AFTER the memberships
    upsert already ran — but both writes share ONE uncommitted SQLite
    transaction, so rolling back after the failure undoes the memberships
    write too. The table must end up at exactly whatever the LAST
    successfully COMMITTED pass left it at — never a mix of old and new."""
    db_path = tmp_path / "embeddings.db"
    store = MemoryClusterStore(db_path)
    store.replace_pass({"hash-a": 0}, np.zeros((1, 4)), model_id="m")
    assert store.cluster_for("hash-a", model_id="m") == 0

    bad_centroids = [None]  # blows up inside replace_pass's centroid loop
    with pytest.raises(AttributeError):
        store.replace_pass({"hash-a": 1, "hash-b": 2}, bad_centroids, model_id="m")
    store._conn.rollback()  # noqa: SLF001 — undo the uncommitted partial transaction
    store.close()

    # A fresh connection against the same file confirms durability: the
    # PRIOR committed pass (hash-a -> 0) is intact; the failed pass never
    # landed (hash-b was never written; hash-a was never overwritten).
    reopened = MemoryClusterStore(db_path)
    try:
        assert reopened.cluster_for("hash-a", model_id="m") == 0
        assert reopened.cluster_for("hash-b", model_id="m") is None
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# run_clustering_pass — sparse-data skip + idempotency
# ---------------------------------------------------------------------------


def _seed_embedding_cache(cache: EmbeddingCache, n: int) -> list[str]:
    texts = [f"memory content number {i} long enough to embed cleanly" for i in range(n)]
    for t in texts:
        cache.get_or_compute(t)
    return texts


def test_sparse_data_skips_cleanly_no_crash_no_write(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.db", FakeEmbeddingProvider(dim=8))
    cluster_store = MemoryClusterStore(tmp_path / "embeddings.db")
    try:
        _seed_embedding_cache(cache, MIN_VECTORS_TO_CLUSTER - 1)
        result = run_clustering_pass(cache, cluster_store)
        assert result.ran is False
        assert result.reason == "sparse-skip"
        assert cluster_store.count() == 0
    finally:
        cache.close()
        cluster_store.close()


def test_at_the_floor_clustering_runs(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.db", FakeEmbeddingProvider(dim=8))
    cluster_store = MemoryClusterStore(tmp_path / "embeddings.db")
    try:
        _seed_embedding_cache(cache, MIN_VECTORS_TO_CLUSTER)
        result = run_clustering_pass(cache, cluster_store)
        assert result.ran is True
        assert result.reason == "ok"
        assert result.n_vectors == MIN_VECTORS_TO_CLUSTER
        assert cluster_store.count(model_id=cache.model_id) == MIN_VECTORS_TO_CLUSTER
    finally:
        cache.close()
        cluster_store.close()


def test_empty_cache_skips_cleanly(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.db", FakeEmbeddingProvider(dim=8))
    cluster_store = MemoryClusterStore(tmp_path / "embeddings.db")
    try:
        result = run_clustering_pass(cache, cluster_store)
        assert result.ran is False
        assert result.n_vectors == 0
    finally:
        cache.close()
        cluster_store.close()


def test_idempotent_rerun_same_corpus_same_seed_converges(tmp_path: Path) -> None:
    """Re-running the pass against an unchanged cache (default seed) leaves
    every content's cluster tag unchanged, and the row count doesn't grow."""
    cache = EmbeddingCache(tmp_path / "embeddings.db", FakeEmbeddingProvider(dim=8))
    cluster_store = MemoryClusterStore(tmp_path / "embeddings.db")
    try:
        texts = _seed_embedding_cache(cache, 20)
        result1 = run_clustering_pass(cache, cluster_store)
        tags_after_1 = {t: cluster_store.cluster_for_content(t, model_id=cache.model_id) for t in texts}

        result2 = run_clustering_pass(cache, cluster_store)
        tags_after_2 = {t: cluster_store.cluster_for_content(t, model_id=cache.model_id) for t in texts}

        assert result1.ran and result2.ran
        assert tags_after_1 == tags_after_2
        assert cluster_store.count(model_id=cache.model_id) == 20  # no growth, no dupes
    finally:
        cache.close()
        cluster_store.close()


def test_kill_mid_pass_then_rerun_leaves_consistent_state(tmp_path: Path) -> None:
    """Simulates a process kill between two ticks: a fresh
    EmbeddingCache/MemoryClusterStore pair opened against the same on-disk
    files after an interrupted-looking prior pass still converges to a
    consistent, fully-clustered state on the next call — same resumability
    posture as the Stage 2 embedding backfill, even though this module
    recomputes wholesale rather than chipping incrementally."""
    db_path = tmp_path / "embeddings.db"
    cache_a = EmbeddingCache(db_path, FakeEmbeddingProvider(dim=8))
    texts = _seed_embedding_cache(cache_a, 15)
    cluster_store_a = MemoryClusterStore(db_path)
    run_clustering_pass(cache_a, cluster_store_a)
    cache_a.close()
    cluster_store_a.close()  # simulates the process dying right after commit

    # Fresh "process" reopens the same files and runs again.
    cache_b = EmbeddingCache(db_path, FakeEmbeddingProvider(dim=8))
    cluster_store_b = MemoryClusterStore(db_path)
    try:
        result = run_clustering_pass(cache_b, cluster_store_b)
        assert result.ran is True
        for t in texts:
            assert cluster_store_b.cluster_for_content(t, model_id=cache_b.model_id) is not None
        assert cluster_store_b.count(model_id=cache_b.model_id) == 15
    finally:
        cache_b.close()
        cluster_store_b.close()


# ---------------------------------------------------------------------------
# cluster_tag_for_memory — the query accessor retrieval will later use
# ---------------------------------------------------------------------------


def test_cluster_tag_for_memory_end_to_end(tmp_path: Path) -> None:
    memories_db = tmp_path / "memories.db"
    embeddings_db = tmp_path / "embeddings.db"

    store = MemoryStore(str(memories_db), integrity_check=False)
    cache = EmbeddingCache(embeddings_db, FakeEmbeddingProvider(dim=8))
    cluster_store = MemoryClusterStore(embeddings_db)
    try:
        made_ids = []
        for i in range(MIN_VECTORS_TO_CLUSTER):
            m = Memory.create_new(
                content=f"memory content number {i} long enough to embed",
                memory_type="conversation",
                domain="us",
            )
            store.create(m)
            made_ids.append(m.id)
            cache.get_or_compute(m.content)

        result = run_clustering_pass(cache, cluster_store)
        assert result.ran is True

        for mid in made_ids:
            tag = cluster_tag_for_memory(
                mid, store=store, embeddings=cache, cluster_store=cluster_store
            )
            assert tag is not None
            assert isinstance(tag, int)
    finally:
        store.close()
        cache.close()
        cluster_store.close()


def test_cluster_tag_for_memory_returns_none_for_unknown_memory(tmp_path: Path) -> None:
    memories_db = tmp_path / "memories.db"
    embeddings_db = tmp_path / "embeddings.db"
    store = MemoryStore(str(memories_db), integrity_check=False)
    cache = EmbeddingCache(embeddings_db, FakeEmbeddingProvider(dim=8))
    cluster_store = MemoryClusterStore(embeddings_db)
    try:
        tag = cluster_tag_for_memory(
            "nonexistent-id", store=store, embeddings=cache, cluster_store=cluster_store
        )
        assert tag is None
    finally:
        store.close()
        cache.close()
        cluster_store.close()


def test_cluster_tag_for_memory_does_not_bump_recall(tmp_path: Path) -> None:
    """Mirrors the `_EmbeddingsByMemoryId` precedent's own trap (Part A of
    the build plan): a candidate-pool/metadata lookup must never inflate
    recall_count just from being consulted."""
    memories_db = tmp_path / "memories.db"
    embeddings_db = tmp_path / "embeddings.db"
    store = MemoryStore(str(memories_db), integrity_check=False)
    cache = EmbeddingCache(embeddings_db, FakeEmbeddingProvider(dim=8))
    cluster_store = MemoryClusterStore(embeddings_db)
    try:
        m = Memory.create_new(
            content="a memory whose recall count must not move",
            memory_type="conversation",
            domain="us",
        )
        store.create(m)
        cache.get_or_compute(m.content)
        before = store.get(m.id, bump=False)
        assert before is not None
        recall_before = before.recall_count

        cluster_tag_for_memory(m.id, store=store, embeddings=cache, cluster_store=cluster_store)

        after = store.get(m.id, bump=False)
        assert after is not None
        assert after.recall_count == recall_before
    finally:
        store.close()
        cache.close()
        cluster_store.close()


# ---------------------------------------------------------------------------
# No new heavy dependency (numpy-only) sanity
# ---------------------------------------------------------------------------


def test_clustering_module_does_not_import_sklearn_or_torch() -> None:
    """Static guard for the hard 'numpy-only k-means' constraint: the module
    must not IMPORT scikit-learn/scipy/torch (checks actual import
    statements via the AST, not prose — the module's own docstring
    legitimately mentions these names when explaining what it deliberately
    avoids)."""
    import ast

    import brain.memory.clustering as clustering_mod

    src = Path(clustering_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"sklearn", "scipy", "torch"}
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint(banned)
