"""Tests for brain.works.store — SQLite index with FTS5 search."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain import works
from brain.works.store import WorksStore


def _w(
    *,
    work_id: str = "abc123def456",
    title: str = "Sample work",
    type_: str = "story",
    created_at: datetime | None = None,
    session_id: str | None = "session-x",
    word_count: int = 100,
    summary: str | None = "Sample summary",
) -> works.Work:
    return works.Work(
        id=work_id,
        title=title,
        type=type_,
        created_at=created_at or datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC),
        session_id=session_id,
        word_count=word_count,
        summary=summary,
    )


def test_store_creates_db_and_schema_version(tmp_path: Path) -> None:
    db_path = tmp_path / "works.db"
    store = WorksStore(db_path)
    assert db_path.exists()
    assert store.schema_version() == 1


def test_store_insert_and_get(tmp_path: Path) -> None:
    store = WorksStore(tmp_path / "works.db")
    w = _w()
    store.insert(w, content="Once upon a lighthouse.")
    fetched = store.get(w.id)
    assert fetched == w


def test_store_get_missing_returns_none(tmp_path: Path) -> None:
    store = WorksStore(tmp_path / "works.db")
    assert store.get("doesnotexist") is None


def test_store_insert_dedupes_on_id(tmp_path: Path) -> None:
    """Inserting the same id twice is idempotent; second insert is a no-op."""
    store = WorksStore(tmp_path / "works.db")
    w = _w()
    store.insert(w, content="content")
    store.insert(w, content="content")  # second insert
    assert store.list_recent(limit=10) == [w]


def test_store_list_recent_orders_by_created_at_desc(tmp_path: Path) -> None:
    store = WorksStore(tmp_path / "works.db")
    older = _w(work_id="111111111111", title="older", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    newer = _w(work_id="222222222222", title="newer", created_at=datetime(2026, 5, 1, tzinfo=UTC))
    store.insert(older, content="older content")
    store.insert(newer, content="newer content")
    assert store.list_recent(limit=10) == [newer, older]


def test_store_list_recent_filters_by_type(tmp_path: Path) -> None:
    store = WorksStore(tmp_path / "works.db")
    s = _w(work_id="111111111111", title="a story", type_="story")
    c = _w(work_id="222222222222", title="some code", type_="code")
    i = _w(work_id="333333333333", title="an idea", type_="idea")
    for w in (s, c, i):
        store.insert(w, content=w.title)

    stories_only = store.list_recent(limit=10, type="story")
    assert {w.id for w in stories_only} == {"111111111111"}

    codes_only = store.list_recent(limit=10, type="code")
    assert {w.id for w in codes_only} == {"222222222222"}


def test_store_list_recent_respects_limit(tmp_path: Path) -> None:
    store = WorksStore(tmp_path / "works.db")
    for i in range(5):
        store.insert(
            _w(work_id=f"{i:012d}", title=f"work {i}", created_at=datetime(2026, 5, i + 1, tzinfo=UTC)),
            content=f"content {i}",
        )
    assert len(store.list_recent(limit=3)) == 3


def test_store_search_finds_matches_by_title(tmp_path: Path) -> None:
    store = WorksStore(tmp_path / "works.db")
    w1 = _w(work_id="111111111111", title="The Lighthouse Keeper's Daughter")
    w2 = _w(work_id="222222222222", title="A Different Story")
    store.insert(w1, content="content one")
    store.insert(w2, content="content two")

    matches = store.search("lighthouse", limit=10)
    assert {w.id for w in matches} == {"111111111111"}


def test_store_search_finds_matches_by_content(tmp_path: Path) -> None:
    store = WorksStore(tmp_path / "works.db")
    w1 = _w(work_id="111111111111", title="A note", summary="x")
    store.insert(w1, content="The thing about chrysanthemums in autumn.")

    matches = store.search("chrysanthemums", limit=10)
    assert {w.id for w in matches} == {"111111111111"}


def test_store_search_finds_matches_by_summary(tmp_path: Path) -> None:
    store = WorksStore(tmp_path / "works.db")
    w1 = _w(work_id="111111111111", title="Untitled", summary="A meditation on solitude.")
    store.insert(w1, content="body")
    matches = store.search("solitude", limit=10)
    assert {w.id for w in matches} == {"111111111111"}


def test_store_search_with_type_filter(tmp_path: Path) -> None:
    store = WorksStore(tmp_path / "works.db")
    w1 = _w(work_id="111111111111", title="lighthouse story", type_="story")
    w2 = _w(work_id="222222222222", title="lighthouse code", type_="code")
    store.insert(w1, content="story body about lighthouses")
    store.insert(w2, content="code body about lighthouses")

    matches = store.search("lighthouse", type="story", limit=10)
    assert {w.id for w in matches} == {"111111111111"}


def test_store_search_returns_empty_on_no_match(tmp_path: Path) -> None:
    store = WorksStore(tmp_path / "works.db")
    store.insert(_w(), content="something else")
    assert store.search("zzzzzzzz", limit=10) == []


def test_search_returns_empty_on_malformed_fts5_query(tmp_path: Path) -> None:
    """Malformed FTS5 queries (unbalanced quotes, bare operators, etc.) must
    return [] not raise sqlite3.OperationalError. The LLM emitting a bad query
    should see no matches, not a stack trace."""
    store = WorksStore(tmp_path / "works.db")
    w = _w(work_id="111111111111", title="The Lighthouse")
    store.insert(w, content="content")

    # Each of these is malformed FTS5 syntax that previously raised
    # sqlite3.OperationalError ("unterminated string", "fts5: syntax error",
    # "unknown special query: ...").
    for malformed in [
        'lighthouse"',     # unbalanced quote
        'AND',             # bare operator
        '"lighthouse',     # unclosed quote
        'NEAR(',           # malformed near
        '*lighthouse',     # leading wildcard
        '(lighthouse',     # unbalanced paren
    ]:
        result = store.search(malformed, limit=10)
        assert result == [], f"query {malformed!r} should return [] (got {result!r})"


def test_list_recent_returns_empty_on_operational_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symmetric to test_search_returns_empty_on_malformed_fts5_query: a
    SQLite OperationalError on list_recent must not propagate to the bridge
    HTTP layer as 500. Returns [] so the caller sees "no rows" instead of
    a stack trace.

    Reproduces I-3 from the 2026-05-05 follow-up audit: the original I-2 fix
    only protected search() — list_recent + get were left uncaught."""
    import sqlite3
    store = WorksStore(tmp_path / "works.db")
    w = _w(work_id="111111111111", title="A")
    store.insert(w, content="content")

    real_connect = store._connect

    class _RaisingConn:
        def __init__(self, conn): self._conn = conn
        def __enter__(self): return self
        def __exit__(self, *a): self._conn.__exit__(*a)
        def execute(self, *a, **kw):
            raise sqlite3.OperationalError("simulated DB corruption")

    monkeypatch.setattr(store, "_connect", lambda: _RaisingConn(real_connect()))
    assert store.list_recent(limit=5) == []


def test_get_returns_none_on_operational_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symmetric to list_recent above: get() returns None on
    sqlite3.OperationalError instead of propagating. Mirrors how the
    function already handles the missing-row case."""
    import sqlite3
    store = WorksStore(tmp_path / "works.db")
    w = _w(work_id="222222222222", title="B")
    store.insert(w, content="content")

    real_connect = store._connect

    class _RaisingConn:
        def __init__(self, conn): self._conn = conn
        def __enter__(self): return self
        def __exit__(self, *a): self._conn.__exit__(*a)
        def execute(self, *a, **kw):
            raise sqlite3.OperationalError("simulated DB corruption")

    monkeypatch.setattr(store, "_connect", lambda: _RaisingConn(real_connect()))
    assert store.get("222222222222") is None


# ---- I-4 (audit-2 follow-up): atomic insert across works + works_fts ----


def test_insert_rolls_back_works_row_when_fts_insert_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the works_fts insert fails after the works insert succeeds, the
    works row must NOT be left dangling. Pre-fix: depending on autocommit
    semantics, the works row could persist with no FTS index — search_works
    silently never returns it.

    Repro: monkeypatch sqlite3.Connection.execute to raise OperationalError
    on the works_fts INSERT (second statement)."""
    import sqlite3
    store = WorksStore(tmp_path / "works.db")
    w = _w(work_id="aaaaaaaaaaaa", title="Atomic Test")

    real_connect = store._connect

    class _ConnWrapper:
        """Forwards to a real sqlite3.Connection except execute() raises
        on the works_fts INSERT — sqlite3.Connection itself is immutable
        so we wrap rather than monkey-patch."""
        def __init__(self, real): self._real = real
        def execute(self, sql, *a, **kw):
            if "INSERT INTO works_fts" in sql:
                raise sqlite3.OperationalError("simulated FTS5 failure")
            return self._real.execute(sql, *a, **kw)
        def close(self): return self._real.close()
        def __enter__(self): return self
        def __exit__(self, *a): return self._real.__exit__(*a)

    monkeypatch.setattr(store, "_connect", lambda: _ConnWrapper(real_connect()))

    with pytest.raises(sqlite3.OperationalError):
        store.insert(w, content="some content")

    monkeypatch.undo()
    # The works row must be absent — full rollback expected.
    assert store.get("aaaaaaaaaaaa") is None, (
        "works row leaked despite works_fts insert failing"
    )


def test_insert_concurrent_dedup_under_load(tmp_path: Path) -> None:
    """Two threads racing to insert the SAME work id must end with exactly
    one row (idempotent on id) and zero exceptions. BEGIN IMMEDIATE
    serializes the SELECT-then-INSERT compound op so the second writer
    sees the first's row in the same transaction."""
    import threading
    store = WorksStore(tmp_path / "works.db")
    w = _w(work_id="bbbbbbbbbbbb", title="Concurrent")

    errors: list[Exception] = []

    def writer():
        try:
            for _ in range(5):
                store.insert(w, content="payload")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == [], f"unexpected errors: {errors[:3]}"
    # Exactly one row in works AND exactly one in works_fts
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "works.db"))
    n_works = conn.execute(
        "SELECT COUNT(*) FROM works WHERE id = ?", (w.id,)
    ).fetchone()[0]
    n_fts = conn.execute(
        "SELECT COUNT(*) FROM works_fts WHERE id = ?", (w.id,)
    ).fetchone()[0]
    conn.close()
    assert n_works == 1, f"expected 1 works row, got {n_works}"
    assert n_fts == 1, f"expected 1 fts row, got {n_fts}"
