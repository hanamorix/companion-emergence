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

    def counting_load():
        calls.append(1)
        return original()

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

    def slow_load():
        started.set()
        release.wait(timeout=5)
        return real_load()

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
# Acceptance criterion 3: concurrent reader vs writer (put-storm + rebuild)
# ---------------------------------------------------------------------------


def test_concurrent_reads_survive_writer_puts_and_rebuild(seeded_db) -> None:
    """A recall-style reader thread hammers get()/snapshot() while a
    supervisor-style writer thread does many per-item puts interleaved with
    full rebuilds. Assert: no exception, and no torn read — every vector
    handed back (from either snapshot() or get()) is a full, correctly
    shaped/typed float32 array, never a partial/short buffer or a
    half-built matrix, for the whole duration of the concurrent run."""
    matrix = EmbeddingMatrix(seeded_db, model_id="fake-model-v1")
    matrix.ensure_built()

    ids = [f"mem-writer-{i}" for i in range(20)]
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
                for mem_id in ids:
                    got = matrix.get(mem_id)
                    if got is not None:
                        assert got.dtype == np.float32
                        assert got.shape == (DIM,)
        except BaseException as exc:  # noqa: BLE001 - captured for the main thread to re-raise
            errors.append(exc)

    def writer() -> None:
        try:
            for round_ in range(50):
                for i, mem_id in enumerate(ids):
                    matrix.put(mem_id, _vec(float(i) + round_ * 0.01))
                if round_ % 10 == 0:
                    matrix.rebuild()
            for mem_id in ids:
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
