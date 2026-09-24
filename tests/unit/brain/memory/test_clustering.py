"""Tests for brain.memory.clustering — Stage 5 (#157) of the local
semantic-retrieval build: numpy-only k-means over embedded memory vectors,
exposed as a machine-usable retrieval tag.

Covers the build task's acceptance bar: stable membership on a seeded
fixture, model_id-scoped tags, a model_id swap recomputes rather than
serving stale tags, sparse-data skip, idempotent re-run.

F1 (#259) increment 4: `run_clustering_pass`/`cluster_tag_for_memory` source
vectors from the warm `EmbeddingMatrix` over `memories.db` and write
`cluster_id`/`cluster_model_id` onto the `memories` row + the
`cluster_centroids` table — NOT the old content-hash-keyed
`MemoryClusterStore`/`embeddings.db` side table, which was dead since
increment 4 and is REMOVED (class + its own tests) in the increment 8 code
teardown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brain.bridge import model_tier
from brain.memory.clustering import (
    K_MAX,
    K_MIN,
    MIN_VECTORS_TO_CLUSTER,
    choose_k,
    cluster_tag_for_memory,
    kmeans,
    run_clustering_pass,
)
from brain.memory.embedding_matrix import _reset_embedding_matrix_cache, build_embedding_matrix
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
# run_clustering_pass / cluster_tag_for_memory — F1 #259 increment 4: sourced
# from the warm EmbeddingMatrix over memories.db, written onto the memories
# row (`cluster_id`/`cluster_model_id`) + the `cluster_centroids` table.
# ---------------------------------------------------------------------------

_TEST_MODEL_ID = "fake-clustering-test-model"


def _align_embedding_tier(monkeypatch: pytest.MonkeyPatch, model_id: str = _TEST_MODEL_ID) -> None:
    """`run_clustering_pass` (via the warm `EmbeddingMatrix`) and
    `cluster_tag_for_memory` both derive their model_id from
    `model_tier.model_for_tier(TIER_EMBEDDING)` — NOT from any embedding
    provider a test might otherwise construct. Align the two so the
    matrix's lazy-build filter (and `cluster_tag_for_memory`'s model_id
    comparison) actually match rows seeded under `model_id` — same
    convention as `test_semantic_recall.py`'s `_align_embedding_tier`."""
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, model_id)


def _seed_embedded_rows(
    store: MemoryStore,
    n: int,
    *,
    model_id: str = _TEST_MODEL_ID,
    label: str = "row",
) -> list[str]:
    """Create `n` active memory rows, each with a deterministic 384-dim
    vector written directly onto `embedding`/`embedding_model_id`. 384 is
    just this fixture's own consistent choice — the warm matrix no longer
    requires any particular dim (#259 inc7 red-team F1: `_load_from_db`
    decodes each row to its own stored byte-length, gated only on the
    `embedding_model_id` filter, not a dimension constant) — but every row
    here must still share ONE dim with each other, since `run_clustering_
    pass`'s `np.stack` over the matrix snapshot requires a uniform shape.
    Bypasses a real embedding provider entirely, same convention as
    `test_semantic_recall.py`'s `_seed_row_vector`. Returns the created ids
    in insertion order."""
    ids: list[str] = []
    for i in range(n):
        m = Memory.create_new(
            content=f"{label} memory content number {i}",
            memory_type="conversation",
            domain="us",
        )
        store.create(m)
        vec = np.full(384, float(i), dtype=np.float32)
        store._conn.execute(  # noqa: SLF001
            "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
            (vec.tobytes(), model_id, m.id),
        )
        ids.append(m.id)
    store._conn.commit()  # noqa: SLF001
    return ids


def _bulk_seed_embedded_rows(
    store: MemoryStore, n: int, *, model_id: str = _TEST_MODEL_ID, label: str = "bulk"
) -> list[str]:
    """Bulk-insert `n` embedded memory rows via a single `executemany` + one
    commit — bypasses `Memory.create_new`/`store.create()`'s per-row commit,
    used only for corpora too large to seed one row at a time within a
    reasonable test runtime (mirrors this module's pre-F1
    `_bulk_seed_cache_directly` helper, ported onto the row schema)."""
    import uuid as _uuid
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime

    now = _datetime.now(_UTC).isoformat()
    ids: list[str] = []
    rows = []
    for i in range(n):
        mem_id = str(_uuid.uuid4())
        ids.append(mem_id)
        vec = np.full(384, float(i), dtype=np.float32)
        rows.append(
            (
                mem_id,
                f"{label} bulk-seeded memory number {i}",
                "conversation",
                "us",
                "{}",
                "[]",
                now,
                1,
                vec.tobytes(),
                model_id,
            )
        )
    store._conn.executemany(  # noqa: SLF001
        "INSERT INTO memories (id, content, memory_type, domain, emotions_json,"
        " tags_json, created_at, active, embedding, embedding_model_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    store._conn.commit()  # noqa: SLF001
    return ids


def test_sparse_data_skips_cleanly_no_crash_no_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _align_embedding_tier(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        ids = _seed_embedded_rows(store, MIN_VECTORS_TO_CLUSTER - 1)
        result = run_clustering_pass(store)
        assert result.ran is False
        assert result.reason == "sparse-skip"
        for mid in ids:
            assert store.get_cluster_id(mid) is None
        n_centroids = store._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM cluster_centroids"
        ).fetchone()[0]
        assert n_centroids == 0
    finally:
        store.close()


def test_at_the_floor_clustering_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _align_embedding_tier(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        ids = _seed_embedded_rows(store, MIN_VECTORS_TO_CLUSTER)
        result = run_clustering_pass(store)
        assert result.ran is True
        assert result.reason == "ok"
        assert result.n_vectors == MIN_VECTORS_TO_CLUSTER
        for mid in ids:
            row = store.get_cluster_id(mid)
            assert row is not None
            cluster_id, cluster_model_id = row
            assert cluster_model_id == _TEST_MODEL_ID
            assert isinstance(cluster_id, int)
    finally:
        store.close()


def test_empty_matrix_skips_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _align_embedding_tier(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        result = run_clustering_pass(store)
        assert result.ran is False
        assert result.n_vectors == 0
    finally:
        store.close()


def test_idempotent_rerun_same_corpus_same_seed_converges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running the pass against an unchanged corpus (default seed) leaves
    every memory's cluster tag unchanged, and no centroid-row growth."""
    _align_embedding_tier(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        ids = _seed_embedded_rows(store, 20)
        result1 = run_clustering_pass(store)
        tags_after_1 = {mid: store.get_cluster_id(mid) for mid in ids}

        result2 = run_clustering_pass(store)
        tags_after_2 = {mid: store.get_cluster_id(mid) for mid in ids}

        assert result1.ran and result2.ran
        assert tags_after_1 == tags_after_2
        n_centroids = store._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM cluster_centroids WHERE model_id = ?",
            (_TEST_MODEL_ID,),
        ).fetchone()[0]
        assert n_centroids == result1.k  # no growth, no dupes
    finally:
        store.close()


def test_kill_mid_pass_then_rerun_leaves_consistent_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulates a process kill between two ticks: a fresh MemoryStore (and
    a fresh warm matrix — the process-wide singleton cache is explicitly
    reset mid-test to simulate a genuinely new process, since a real restart
    would never carry the old matrix over) opened against the same on-disk
    file after an interrupted-looking prior pass still converges to a
    consistent, fully-clustered state on the next call — same resumability
    posture as the Stage 2 embedding backfill, even though this module
    recomputes wholesale rather than chipping incrementally."""
    _align_embedding_tier(monkeypatch)
    db_path = tmp_path / "memories.db"
    store_a = MemoryStore(db_path, integrity_check=False)
    ids = _seed_embedded_rows(store_a, 15)
    run_clustering_pass(store_a)
    store_a.close()  # simulates the process dying right after commit

    _reset_embedding_matrix_cache()  # simulate a fresh process: no warm matrix carried over

    store_b = MemoryStore(db_path, integrity_check=False)
    try:
        result = run_clustering_pass(store_b)
        assert result.ran is True
        for mid in ids:
            assert store_b.get_cluster_id(mid) is not None
        n_centroids = store_b._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM cluster_centroids WHERE model_id = ?",
            (_TEST_MODEL_ID,),
        ).fetchone()[0]
        assert n_centroids == result.k
    finally:
        store_b.close()


def test_run_clustering_pass_drops_stale_membership_for_evicted_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end regression for the dangling/stale cluster_id bug: a memory
    whose embedding is evicted between two passes under the same model_id
    (fade/hard_delete/a failed re-embed all clear the row + evict the
    warm-matrix entry the same way) must lose its cluster membership
    entirely (cluster_id -> NULL), never keep a stale cluster_id pointing at
    a centroid this pass deleted."""
    _align_embedding_tier(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        # Seed one more than the sparse-data floor so eviction still leaves
        # enough vectors for pass 2 to actually run (not sparse-skip).
        ids = _seed_embedded_rows(store, MIN_VECTORS_TO_CLUSTER + 1)
        result1 = run_clustering_pass(store)
        assert result1.ran is True

        evicted_id = ids[0]
        assert store.get_cluster_id(evicted_id) is not None

        # Simulate the row's embedding being evicted — clear the row column
        # AND the warm-matrix entry directly, exactly what
        # MemoryStore.fade()/hard_delete() do in production.
        store._conn.execute(  # noqa: SLF001
            "UPDATE memories SET embedding = NULL, embedding_model_id = NULL WHERE id = ?",
            (evicted_id,),
        )
        store._conn.commit()  # noqa: SLF001
        build_embedding_matrix(store.db_path).evict(evicted_id)

        result2 = run_clustering_pass(store)
        assert result2.ran is True
        assert result2.n_vectors == MIN_VECTORS_TO_CLUSTER

        # The evicted memory's membership must be gone, not stale.
        assert store.get_cluster_id(evicted_id) is None

        # Every remaining membership under this model_id has a live centroid.
        for mid in ids[1:]:
            row = store.get_cluster_id(mid)
            assert row is not None
            cluster_id, _cluster_model_id = row
            live = store._conn.execute(  # noqa: SLF001
                "SELECT 1 FROM cluster_centroids WHERE model_id = ? AND cluster_id = ?",
                (_TEST_MODEL_ID, cluster_id),
            ).fetchone()
            assert live is not None
    finally:
        store.close()


def test_model_swap_recomputes_rather_than_serving_stale_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model swap must not silently serve a cluster tag computed under a
    stale model's vector space — it must recompute (or stay absent) under
    the new model_id."""
    old_model_id = "fake-clustering-model-old"
    new_model_id = "fake-clustering-model-new"
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        _align_embedding_tier(monkeypatch, old_model_id)
        old_ids = _seed_embedded_rows(store, 10, model_id=old_model_id, label="row-v1")
        result_old = run_clustering_pass(store)
        assert result_old.ran is True
        for mid in old_ids:
            row = store.get_cluster_id(mid)
            assert row is not None
            assert row[1] == old_model_id
            assert cluster_tag_for_memory(mid, store=store) is not None

        # Swap the active embedding tier to a new model — a genuinely new
        # process would build a fresh matrix filtered to the new model_id;
        # reset the singleton to simulate that rather than relying on this
        # process's already-built (old-model) matrix.
        _reset_embedding_matrix_cache()
        _align_embedding_tier(monkeypatch, new_model_id)

        # Before the new model has embedded/clustered anything, old tags
        # must NEVER be served under the new model_id.
        for mid in old_ids:
            assert cluster_tag_for_memory(mid, store=store) is None

        new_ids = _seed_embedded_rows(store, 10, model_id=new_model_id, label="row-v2")
        result_new = run_clustering_pass(store)
        assert result_new.ran is True
        for mid in new_ids:
            assert cluster_tag_for_memory(mid, store=store) is not None

        # The OLD rows are untouched by the new-model pass (their
        # cluster_model_id is still old_model_id) — a lookup still scoped to
        # the CURRENT (new) model correctly returns None for them, the
        # targeted invalidation the hard invariant calls for: never serve a
        # stale tag.
        for mid in old_ids:
            row = store.get_cluster_id(mid)
            assert row is not None
            assert row[1] == old_model_id  # untouched, not overwritten
            assert cluster_tag_for_memory(mid, store=store) is None
    finally:
        store.close()


def test_clustering_includes_rows_beyond_the_old_fixed_window_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the fixed-window-LIMIT-with-no-ORDER-BY starvation bug
    (see module history): an earlier version's vector source applied a fixed
    `LIMIT` with no `ORDER BY`, so once the corpus exceeded that cap, only
    the same first-N rows were ever clustered and every later row was
    silently and PERMANENTLY excluded. The warm-matrix snapshot this pass
    now reads has no LIMIT at all — confirm the full corpus, including rows
    well past the old cap, is included."""
    _align_embedding_tier(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        old_cap = 5000
        total = old_cap + 50
        ids = _bulk_seed_embedded_rows(store, total, label="starvation-regression")

        result = run_clustering_pass(store)

        assert result.ran is True
        assert result.n_vectors == total  # not silently capped at the old 5000

        # Rows inserted well past the old fixed window must be clustered too.
        for mid in ids[-10:]:
            assert store.get_cluster_id(mid) is not None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# cluster_tag_for_memory — the query accessor retrieval will later use
# ---------------------------------------------------------------------------


def test_cluster_tag_for_memory_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _align_embedding_tier(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        ids = _seed_embedded_rows(store, MIN_VECTORS_TO_CLUSTER)
        result = run_clustering_pass(store)
        assert result.ran is True

        for mid in ids:
            tag = cluster_tag_for_memory(mid, store=store)
            assert tag is not None
            assert isinstance(tag, int)
    finally:
        store.close()


def test_cluster_tag_for_memory_returns_none_for_unknown_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _align_embedding_tier(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        tag = cluster_tag_for_memory("nonexistent-id", store=store)
        assert tag is None
    finally:
        store.close()


def test_cluster_tag_for_memory_returns_none_for_unclustered_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A memory that exists and is even embedded, but hasn't been through a
    clustering pass yet, has no `cluster_id` -> `None`, not an error."""
    _align_embedding_tier(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        ids = _seed_embedded_rows(store, 1)
        assert cluster_tag_for_memory(ids[0], store=store) is None
    finally:
        store.close()


def test_cluster_tag_for_memory_does_not_bump_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors the `_EmbeddingsByMemoryId` precedent's own trap (Part A of
    the build plan): a candidate-pool/metadata lookup must never inflate
    recall_count just from being consulted. `cluster_tag_for_memory` reads
    via `MemoryStore.get_cluster_id` — a raw row SELECT, not `store.get()` —
    so this asserts that stays true end to end."""
    _align_embedding_tier(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        ids = _seed_embedded_rows(store, MIN_VECTORS_TO_CLUSTER)
        run_clustering_pass(store)
        mid = ids[0]
        before = store.get(mid, bump=False)
        assert before is not None
        recall_before = before.recall_count

        cluster_tag_for_memory(mid, store=store)

        after = store.get(mid, bump=False)
        assert after is not None
        assert after.recall_count == recall_before
    finally:
        store.close()


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
