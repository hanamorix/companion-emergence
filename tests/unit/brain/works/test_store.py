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
