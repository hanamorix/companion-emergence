"""FTS5 shadow-index sync + health-check rebuild + multi-term OR (P2 — C1/C2/C15/C23).

Each oracle is written to FAIL against the named known-bad state:
  - C1/C15 fail if the sync triggers are removed (FTS drifts from `memories`).
  - C2 fails if the FTS health check's rebuild is absent (FTS stays empty).
  - C23 fails against a bare-term (implicit-AND) MATCH (zero rows for a disjoint
    multi-term query).

RAM-spike-fix INC-3 (spec §6c): the self-heal in C2 used to run implicitly on
EVERY `MemoryStore()` open (`_boot_fts_backstop`, retired); it now runs only
via an explicit `db_health.run_fts_health_check_once(db_path)` call (the
bridge lifespan's job in production) — so this test calls it directly rather
than relying on a bare re-open. See `test_db_health.py` for the health-check
module's own criteria (C34/C37).
"""

from __future__ import annotations

from pathlib import Path

from brain.memory import db_health
from brain.memory.store import Memory, MemoryStore


def _mem(content: str) -> Memory:
    return Memory.create_new(content=content, memory_type="event", domain="d")


def _fts_ids(store: MemoryStore, term: str) -> set[str]:
    rows = store._conn.execute(
        "SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
        "WHERE memories_fts MATCH ?",
        (term,),
    ).fetchall()
    return {r[0] for r in rows}


def _like_ids(store: MemoryStore, term: str) -> set[str]:
    # active_only=False + bump=False → an unfiltered substring scan, the LIKE
    # baseline the FTS MATCH set must equal.
    return {m.id for m in store.search_text(term, active_only=False, bump=False)}


def test_c1_fts_stays_in_sync_on_insert_update_delete() -> None:
    store = MemoryStore(":memory:")
    a = _mem("apple pie recipe")
    b = _mem("banana bread loaf")
    c = _mem("apple tart tatin")
    for m in (a, b, c):
        store.create(m)

    # INSERT synced: MATCH == LIKE.
    assert _fts_ids(store, "apple") == _like_ids(store, "apple") == {a.id, c.id}

    # UPDATE(content) synced.
    store.update(b.id, content="apple crumble warm")
    assert _fts_ids(store, "apple") == _like_ids(store, "apple") == {a.id, b.id, c.id}

    # DELETE synced.
    store.hard_delete(a.id)
    assert _fts_ids(store, "apple") == _like_ids(store, "apple") == {b.id, c.id}


def test_c2_fts_self_heals_on_health_check_no_memory_lost(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    db_health._reset_for_tests()
    store = MemoryStore(db)
    for i in range(3):
        store.create(_mem(f"apple number {i}"))
    assert _fts_ids(store, "apple")  # non-empty before

    # Deliberately clear the shadow index (external-content delete-all) —
    # this also drops mem_rows != fts_rows out of sync (count_mismatch).
    store._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('delete-all')")
    store._conn.commit()
    assert not _fts_ids(store, "apple")  # empty now
    mem_count = store.count(active_only=False)
    store.close()

    # The explicit health check (the bridge lifespan's call in production)
    # detects the count mismatch and rebuilds — a bare re-open no longer
    # self-heals (INC-3: the check moved out of MemoryStore.__init__).
    db_health.run_fts_health_check_once(db)

    store2 = MemoryStore(db)
    try:
        assert _fts_ids(store2, "apple"), "health-check rebuild should re-seed the FTS index"
        assert store2.count(active_only=False) == mem_count, "no memory lost across rebuild"
    finally:
        store2.close()


def test_c15_fts_consistency_under_interleaved_write() -> None:
    store = MemoryStore(":memory:")
    a = _mem("cat on a mat")
    store.create(a)
    assert _fts_ids(store, "cat") == {a.id}

    # Inject a write between reads — the second read sees it atomically.
    b = _mem("cat by the window")
    store.create(b)
    assert _fts_ids(store, "cat") == {a.id, b.id}

    # A deleted row is absent from the next read.
    store.hard_delete(a.id)
    assert _fts_ids(store, "cat") == {b.id}


def test_c23_multi_term_query_ors_terms_on_search_fts_scored() -> None:
    store = MemoryStore(":memory:")
    m1 = _mem("henryk drinks coffee")
    m2 = _mem("her preferences about tea")
    store.create(m1)
    store.create(m2)

    # Disjoint terms (never co-occur) must return the UNION, not the empty
    # AND-intersection a bare-term MATCH would give.
    scored = store.search_fts_scored("henryk preferences")
    ids = {m.id for m, _ in scored}
    assert ids == {m1.id, m2.id}


def test_c23_empty_query_returns_no_match() -> None:
    store = MemoryStore(":memory:")
    store.create(_mem("henryk drinks coffee"))
    # All tokens dropped (too short) → no MATCH issued, empty result.
    assert store.search_fts_scored("a an") == []
