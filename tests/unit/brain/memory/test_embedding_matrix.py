"""Tests for brain.memory.embedding_matrix.EmbeddingMatrix (F1 #259 step 2).

Additive-only module — nothing here wires it into recall/backfill/dedupe;
these tests exercise the matrix in isolation against a seeded memories.db.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from brain.memory.embedding_matrix import EmbeddingMatrix
from brain.memory.store import MemoryStore

DIM = 384


def _vec(seed: float) -> np.ndarray:
    """A deterministic, distinguishable float32 vector for test fixtures."""
    return (np.full(DIM, seed, dtype=np.float32))


def _seed_db(db_path: Path, rows: list[tuple[str, np.ndarray | None, bool]]) -> None:
    """Create a memories.db (via MemoryStore, for the real schema) and seed
    rows directly with SQL — `rows` is (id, vector_or_None, active)."""
    store = MemoryStore(db_path)
    now = "2026-01-01T00:00:00+00:00"
    for mem_id, vec, active in rows:
        blob = vec.tobytes() if vec is not None else None
        store._conn.execute(
            "INSERT INTO memories (id, content, memory_type, domain, emotions_json,"
            " tags_json, created_at, active, embedding, embedding_model_id)"
            " VALUES (?, 'body', 'conversation', 'us', '{}', '[]', ?, ?, ?, ?)",
            (mem_id, now, int(active), blob, "fake-model-v1" if vec is not None else None),
        )
    store._conn.commit()
    store.close()


@pytest.fixture
def seeded_db(tmp_path) -> Path:
    db_path = tmp_path / "memories.db"
    _seed_db(
        db_path,
        [
            ("mem-active-1", _vec(0.1), True),
            ("mem-active-2", _vec(0.2), True),
            ("mem-inactive", _vec(0.3), False),  # active=0 — must be excluded
            ("mem-no-embedding", None, True),  # embedding IS NULL — must be excluded
        ],
    )
    return db_path


@pytest.fixture
def coherence_db(tmp_path) -> Path:
    """A memories.db with 12 active, embedded rows (mem-co-0 .. mem-co-11),
    every one durably holding _vec(1.0) under fake-model-v1. Used by the
    concurrency-coherence tests, which put/evict against ids that REALLY
    exist in the DB so a rebuild's reload is a genuine adversary: a lost put
    reverts to the DB's _vec(1.0), a resurrected row reappears from disk."""
    db_path = tmp_path / "memories.db"
    _seed_db(db_path, [(f"mem-co-{i}", _vec(1.0), True) for i in range(12)])
    return db_path


# ---------------------------------------------------------------------------
# Lazy build
# ---------------------------------------------------------------------------


def test_matrix_does_not_build_at_construction(seeded_db) -> None:
    """Building must be lazy — constructing the matrix touches no DB
    connection until a public accessor is called."""
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    assert matrix._built is False


def test_lazy_build_loads_only_active_embedded_rows(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    snap = matrix.snapshot()
    assert set(snap.keys()) == {"mem-active-1", "mem-active-2"}
    assert matrix._built is True


def test_lazy_build_runs_exactly_once_across_repeated_calls(seeded_db, monkeypatch) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    calls = []
    original = matrix._load_from_db

    def counting_load(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(matrix, "_load_from_db", counting_load)
    matrix.get("mem-active-1")
    matrix.get("mem-active-2")
    matrix.snapshot()
    assert len(calls) == 1


def test_get_returns_correct_vector(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    got = matrix.get("mem-active-1")
    assert got is not None
    assert got.dtype == np.float32
    assert got.shape == (DIM,)
    np.testing.assert_array_equal(got, _vec(0.1))


def test_get_missing_id_returns_none(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    assert matrix.get("does-not-exist") is None
    assert matrix.get("mem-inactive") is None
    assert matrix.get("mem-no-embedding") is None


def test_get_returns_a_copy_not_an_alias(seeded_db) -> None:
    """Mutating a returned vector must not corrupt the matrix's internal
    state — get() must hand back a copy."""
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    got = matrix.get("mem-active-1")
    got[0] = 999.0
    got_again = matrix.get("mem-active-1")
    assert got_again[0] == pytest.approx(0.1)


def test_snapshot_is_independent_of_later_puts(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    snap = matrix.snapshot()
    matrix.put("mem-active-1", _vec(42.0))
    assert snap["mem-active-1"][0] == pytest.approx(0.1)  # the OLD snapshot, untouched


# ---------------------------------------------------------------------------
# Per-item put / evict
# ---------------------------------------------------------------------------


def test_put_adds_new_entry(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    matrix.ensure_built()
    matrix.put("mem-new", _vec(5.0))
    assert "mem-new" in matrix
    got = matrix.get("mem-new")
    np.testing.assert_array_equal(got, _vec(5.0))


def test_put_updates_existing_entry(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    matrix.ensure_built()  # already-built matrix — put() must now win over DB truth
    matrix.put("mem-active-1", _vec(77.0))
    got = matrix.get("mem-active-1")
    np.testing.assert_array_equal(got, _vec(77.0))


def test_evict_removes_entry(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    matrix.ensure_built()
    assert "mem-active-1" in matrix
    matrix.evict("mem-active-1")
    assert "mem-active-1" not in matrix
    assert matrix.get("mem-active-1") is None


def test_evict_missing_id_is_a_no_op(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    matrix.evict("never-existed")  # must not raise


def test_put_before_first_lazy_build_does_not_short_circuit_it(seeded_db) -> None:
    """Regression guard: a put() landing before the matrix has ever been
    built must NOT mark it built-with-one-entry — the next access must
    still load the full corpus from disk (the row is already durable in
    the DB by the time anything calls put(), so the lazy load simply
    re-discovers it; skipping the load would silently drop every other
    row forever)."""
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    matrix.put("mem-active-1", _vec(0.1))  # BEFORE any ensure_built/get/snapshot
    assert matrix._built is False
    snap = matrix.snapshot()
    assert set(snap.keys()) == {"mem-active-1", "mem-active-2"}


# ---------------------------------------------------------------------------
# Full rebuild — atomic swap
# ---------------------------------------------------------------------------


def test_rebuild_picks_up_new_rows_written_directly_to_db(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    matrix.ensure_built()
    assert "mem-active-3" not in matrix

    # Simulate an out-of-band write (e.g. a migration/backfill tick in
    # another process) landing directly in the DB.
    conn = sqlite3.connect(str(seeded_db))
    conn.execute(
        "INSERT INTO memories (id, content, memory_type, domain, emotions_json,"
        " tags_json, created_at, active, embedding, embedding_model_id)"
        " VALUES ('mem-active-3', 'x', 'conversation', 'us', '{}', '[]',"
        " '2026-01-01T00:00:00+00:00', 1, ?, 'fake-model-v1')",
        (_vec(0.9).tobytes(),),
    )
    conn.commit()
    conn.close()

    matrix.rebuild()
    assert "mem-active-3" in matrix
    np.testing.assert_array_equal(matrix.get("mem-active-3"), _vec(0.9))


def test_rebuild_updates_model_id_when_passed(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    assert matrix.model_id == "fake-model-v1"
    matrix.rebuild(model_id="fake-model-v2")
    assert matrix.model_id == "fake-model-v2"


def test_rebuild_without_model_id_keeps_current_model_id(seeded_db) -> None:
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    matrix.rebuild()
    assert matrix.model_id == "fake-model-v1"


def test_rebuild_is_a_reference_swap_readers_never_see_a_half_built_dict(
    seeded_db, monkeypatch
) -> None:
    """The slow DB scan must happen BEFORE the lock is taken — assert the
    matrix still serves the OLD full snapshot while a rebuild's (stubbed
    slow) scan is in flight."""
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    matrix.ensure_built()
    old_keys = set(matrix.snapshot().keys())

    real_load = matrix._load_from_db
    started = threading.Event()
    release = threading.Event()

    def slow_load(*args, **kwargs):
        started.set()
        release.wait(timeout=5)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(matrix, "_load_from_db", slow_load)

    rebuild_thread = threading.Thread(target=matrix.rebuild)
    rebuild_thread.start()
    assert started.wait(timeout=5)

    # Rebuild's scan is blocked mid-flight (still outside the lock) — a
    # concurrent reader must be served the OLD matrix immediately, not wait.
    t0 = time.monotonic()
    snap_during = matrix.snapshot()
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, "reader blocked behind an in-flight rebuild's DB scan"
    assert set(snap_during.keys()) == old_keys

    release.set()
    rebuild_thread.join(timeout=5)
    assert not rebuild_thread.is_alive()


# ---------------------------------------------------------------------------
# Acceptance criterion 3: warm-matrix stays COHERENT under concurrent
# rebuild + put/evict — no lost update, no resurrected row, no stale vector.
# ---------------------------------------------------------------------------


def test_rebuild_reconciles_concurrent_put_and_evict(coherence_db, monkeypatch) -> None:
    """DETERMINISTIC coherence proof (fails on the pre-fix wholesale-swap).

    A rebuild's slow DB scan is pinned mid-flight (its off-lock window held
    open by an event). While it is blocked, three mutations land against ids
    that DO exist in the DB:

      * put("mem-co-0", 2.0)   — the DB still holds 1.0 for this id
      * put("mem-co-new", 3.0) — a brand-new id with no DB row at all
      * evict("mem-co-1")      — the DB still holds a row for this id

    A wholesale `self._vectors = loaded` at swap time (the pre-fix code)
    discards all three: mem-co-0 reverts to the DB's 1.0 (lost update),
    mem-co-new vanishes (lost update), mem-co-1 comes back from disk
    (resurrection). The pending-overlay reconcile re-applies them on top of
    the freshly-loaded dict, so every mutation survives the rebuild."""
    matrix = EmbeddingMatrix(coherence_db, model_id="fake-model-v1")
    matrix.ensure_built()
    # Baseline: every id currently reads its DB value.
    np.testing.assert_array_equal(matrix.get("mem-co-0"), _vec(1.0))

    real_load = matrix._load_from_db
    started = threading.Event()
    release = threading.Event()

    def slow_load(*args, **kwargs):
        started.set()
        release.wait(timeout=5)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(matrix, "_load_from_db", slow_load)

    rebuild_thread = threading.Thread(target=matrix.rebuild)
    rebuild_thread.start()
    # The rebuild has entered its off-lock scan; the pending overlay is now
    # recording. Land the mutations INSIDE this window.
    assert started.wait(timeout=5)

    matrix.put("mem-co-0", _vec(2.0))
    matrix.put("mem-co-new", _vec(3.0))
    matrix.evict("mem-co-1")

    release.set()
    rebuild_thread.join(timeout=5)
    assert not rebuild_thread.is_alive()

    # Lost-update guard: the concurrent put must WIN over the stale reload.
    np.testing.assert_array_equal(matrix.get("mem-co-0"), _vec(2.0))
    # Lost-update guard for an id with no DB backing yet: must not vanish.
    got_new = matrix.get("mem-co-new")
    assert got_new is not None
    np.testing.assert_array_equal(got_new, _vec(3.0))
    # Resurrection guard: the concurrent evict must hold, not be undone.
    assert "mem-co-1" not in matrix
    assert matrix.get("mem-co-1") is None
    # Untouched ids load their correct DB value — not stale, not torn.
    np.testing.assert_array_equal(matrix.get("mem-co-2"), _vec(1.0))


def test_concurrent_reads_survive_writer_puts_and_rebuild(coherence_db) -> None:
    """Stress/robustness arm: a recall-style reader thread hammers
    get()/snapshot() while a supervisor-style writer thread does many
    per-item puts/evicts interleaved with full rebuilds, all against ids
    that REALLY exist in the DB. Assert: no exception, no torn read (every
    vector handed back is a full, correctly shaped/typed float32 array), and
    — after the storm settles on a deterministic final state written AFTER
    the last rebuild — membership/value coherence: every keeper holds its
    final value, every deleted id is absent."""
    matrix = EmbeddingMatrix(coherence_db, model_id="fake-model-v1")
    matrix.ensure_built()

    keeper_ids = [f"mem-co-{i}" for i in range(6)]
    delete_ids = [f"mem-co-{i}" for i in range(6, 12)]
    stop = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                snap = matrix.snapshot()
                assert isinstance(snap, dict)
                for vec in snap.values():
                    assert vec.dtype == np.float32
                    assert vec.shape == (DIM,)
                for mem_id in keeper_ids:
                    got = matrix.get(mem_id)
                    if got is not None:
                        assert got.dtype == np.float32
                        assert got.shape == (DIM,)
        except BaseException as exc:  # noqa: BLE001 - captured for the main thread to re-raise
            errors.append(exc)

    def writer() -> None:
        try:
            for round_ in range(50):
                for i, mem_id in enumerate(keeper_ids):
                    matrix.put(mem_id, _vec(float(i) + round_ * 0.01))
                for mem_id in delete_ids:
                    matrix.evict(mem_id)
                if round_ % 10 == 0:
                    # A rebuild here would revert the in-memory puts to the
                    # DB's _vec(1.0) and resurrect the evicted rows if the
                    # overlay reconcile were absent.
                    matrix.rebuild()
            # Deterministic final state, written AFTER the last rebuild so the
            # post-storm assertions are race-free.
            for i, mem_id in enumerate(keeper_ids):
                matrix.put(mem_id, _vec(100.0 + i))
            for mem_id in delete_ids:
                matrix.evict(mem_id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    reader_threads = [threading.Thread(target=reader) for _ in range(3)]
    writer_thread = threading.Thread(target=writer)

    for t in reader_threads:
        t.start()
    writer_thread.start()
    writer_thread.join(timeout=30)
    stop.set()
    for t in reader_threads:
        t.join(timeout=5)

    assert not writer_thread.is_alive()
    assert not any(t.is_alive() for t in reader_threads)
    assert errors == []

    for i, mem_id in enumerate(keeper_ids):
        np.testing.assert_array_equal(matrix.get(mem_id), _vec(100.0 + i))
    for mem_id in delete_ids:
        assert mem_id not in matrix


# ---------------------------------------------------------------------------
# Finding 3: per-row embedding_model_id filter (no stale-model vectors)
# ---------------------------------------------------------------------------


def _insert_row(
    db_path: Path,
    mem_id: str,
    blob: bytes | None,
    model_id: str | None,
    *,
    active: bool = True,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO memories (id, content, memory_type, domain, emotions_json,"
            " tags_json, created_at, active, embedding, embedding_model_id)"
            " VALUES (?, 'body', 'conversation', 'us', '{}', '[]',"
            " '2026-01-01T00:00:00+00:00', ?, ?, ?)",
            (mem_id, int(active), blob, model_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_build_loads_only_the_target_model_id(seeded_db) -> None:
    """A row stamped with a DIFFERENT embedding_model_id must NOT be loaded
    under the current model's label — otherwise a rebuild during a model
    swap serves stale-model vectors under the new model_id."""
    _insert_row(seeded_db, "mem-other-model", _vec(0.5).tobytes(), "fake-model-v2")
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    snap = matrix.snapshot()
    assert "mem-other-model" not in snap
    assert set(snap.keys()) == {"mem-active-1", "mem-active-2"}


def test_rebuild_onto_new_model_id_loads_only_that_models_rows(seeded_db) -> None:
    """After a model swap, rebuild(model_id=v2) loads ONLY the v2 rows — the
    old v1 vectors are dropped, not re-served under the v2 label."""
    _insert_row(seeded_db, "mem-v2-a", _vec(0.7).tobytes(), "fake-model-v2")
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    matrix.ensure_built()
    assert set(matrix.snapshot().keys()) == {"mem-active-1", "mem-active-2"}

    matrix.rebuild(model_id="fake-model-v2")
    assert matrix.model_id == "fake-model-v2"
    assert set(matrix.snapshot().keys()) == {"mem-v2-a"}


# ---------------------------------------------------------------------------
# Finding 4: per-row decode guard (one bad blob must not kill the build)
# ---------------------------------------------------------------------------


def test_corrupt_embedding_blob_is_skipped_not_fatal(seeded_db) -> None:
    """A short/corrupt embedding blob on one row must be skipped-and-logged,
    not throw and take down the whole build (which would propagate into
    recall via the request-thread lazy build)."""
    _insert_row(seeded_db, "mem-corrupt", b"\x00\x01\x02", "fake-model-v1")
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    snap = matrix.snapshot()  # must not raise
    assert "mem-corrupt" not in snap
    # The good rows still loaded.
    assert set(snap.keys()) == {"mem-active-1", "mem-active-2"}


def test_wrong_dim_embedding_blob_is_skipped(seeded_db) -> None:
    """A blob whose byte length is a clean float32 multiple but the wrong
    dimension (not 384) is skipped, not served as a malformed vector."""
    wrong = np.full(128, 0.5, dtype=np.float32).tobytes()  # 128-dim, not 384
    _insert_row(seeded_db, "mem-wrong-dim", wrong, "fake-model-v1")
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    snap = matrix.snapshot()
    assert "mem-wrong-dim" not in snap
    assert set(snap.keys()) == {"mem-active-1", "mem-active-2"}


# ---------------------------------------------------------------------------
# build_embedding_matrix — process-wide singleton (F1 #259 step 0)
# ---------------------------------------------------------------------------


def test_build_embedding_matrix_returns_the_same_instance_for_the_same_path(seeded_db) -> None:
    """Two callers asking for the matrix over the SAME db file must get the
    SAME object — a `put()` from one caller (e.g. embed-on-write) must be
    visible to every other caller reading that file (e.g. a recall-path
    `matrix.get(...)`), which only holds if they share one instance."""
    from brain.memory.embedding_matrix import build_embedding_matrix

    m1 = build_embedding_matrix(seeded_db)
    m2 = build_embedding_matrix(seeded_db)
    assert m1 is m2


def test_build_embedding_matrix_normalizes_path_before_keying(seeded_db: Path) -> None:
    """FIX 2 (increment-2 cold red-team, LOW): two differently-SPELLED paths
    to the SAME underlying file must resolve to the SAME cache key. Keying
    by raw `str(db_path)` alone would split them into two independent
    matrices over one physical memories.db — a `put()`/`evict()` from a
    caller that arrived via one spelling would be silently invisible to a
    reader that arrived via the other. `str(Path(db_path).resolve())`
    collapses the `..` traversal here, so both spellings land on one
    instance."""
    from brain.memory.embedding_matrix import build_embedding_matrix

    aliased_path = seeded_db.parent / "nonexistent_subdir" / ".." / seeded_db.name
    assert str(aliased_path) != str(seeded_db)  # genuinely different spelling ...
    assert aliased_path.resolve() == seeded_db.resolve()  # ... of the same physical file

    m1 = build_embedding_matrix(seeded_db)
    m2 = build_embedding_matrix(aliased_path)
    assert m1 is m2


def test_build_embedding_matrix_returns_different_instances_for_different_paths(
    tmp_path: Path,
) -> None:
    """Keyed by `str(db_path)`, not persona_dir or a bare singleton — two
    different memories.db files must never share a matrix instance."""
    from brain.memory.embedding_matrix import build_embedding_matrix

    db_a = tmp_path / "a" / "memories.db"
    db_b = tmp_path / "b" / "memories.db"
    db_a.parent.mkdir()
    db_b.parent.mkdir()
    _seed_db(db_a, [("mem-a", _vec(0.1), True)])
    _seed_db(db_b, [("mem-b", _vec(0.2), True)])

    m_a = build_embedding_matrix(db_a)
    m_b = build_embedding_matrix(db_b)
    assert m_a is not m_b


def test_build_embedding_matrix_model_id_comes_from_model_tier(
    seeded_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model id is sourced from `model_tier.model_for_tier(TIER_EMBEDDING)`
    (I7 — model ids live in model_tier.py, never hardcoded) — looked up
    through the `model_tier` MODULE so a test monkeypatching
    `model_tier.TIER_MODEL` is honored, mirroring
    `build_embedding_provider`'s own dynamic `model_for_tier` lookup."""
    from brain.bridge import model_tier
    from brain.memory.embedding_matrix import build_embedding_matrix

    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, "a-test-model-id")
    matrix = build_embedding_matrix(seeded_db)
    assert matrix.model_id == "a-test-model-id"


def test_reset_embedding_matrix_cache_clears_the_singleton(seeded_db) -> None:
    """The test-only reset hook (wired into `tests/conftest.py`'s autouse
    fixture) must make the NEXT `build_embedding_matrix()` call for a given
    path construct a fresh instance rather than returning the old one — the
    same contract `embeddings._reset_embedding_provider_cache` gives its own
    cache."""
    from brain.memory.embedding_matrix import _reset_embedding_matrix_cache, build_embedding_matrix

    m1 = build_embedding_matrix(seeded_db)
    _reset_embedding_matrix_cache()
    m2 = build_embedding_matrix(seeded_db)
    assert m1 is not m2


def test_build_embedding_matrix_double_checked_locking_constructs_once(
    seeded_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent first-time callers for the SAME path must all converge on
    ONE constructed EmbeddingMatrix instance — mirrors
    `build_embedding_provider`'s own race-to-construct coverage."""
    import brain.memory.embedding_matrix as embedding_matrix_mod

    real_init = embedding_matrix_mod.EmbeddingMatrix.__init__
    construct_count = {"n": 0}

    def counting_init(self, *args, **kwargs):
        construct_count["n"] += 1
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(embedding_matrix_mod.EmbeddingMatrix, "__init__", counting_init)

    results: list[embedding_matrix_mod.EmbeddingMatrix] = []
    errors: list[BaseException] = []

    def _build() -> None:
        try:
            results.append(embedding_matrix_mod.build_embedding_matrix(seeded_db))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_build) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len({id(r) for r in results}) == 1, "every thread must get the SAME matrix instance"
    assert construct_count["n"] == 1, "EmbeddingMatrix must be constructed exactly once"
