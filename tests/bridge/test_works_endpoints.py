"""Bridge endpoints for /self/works[*]."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain import works
from brain.bridge.server import build_app
from brain.works.storage import write_markdown
from brain.works.store import WorksStore


def _client(persona_dir: Path) -> TestClient:
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests"))


def _seed_two_works(persona_dir: Path) -> tuple[str, str]:
    older = works.Work(
        id="111111111111",
        title="A code snippet",
        type="code",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        session_id=None,
        word_count=5,
        summary="An older code work.",
    )
    newer = works.Work(
        id="222222222222",
        title="The Lighthouse",
        type="story",
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
        session_id=None,
        word_count=10,
        summary="A newer story.",
    )
    write_markdown(persona_dir, older, content="def foo(): pass")
    write_markdown(persona_dir, newer, content="A small story about a lighthouse keeper.")
    db = WorksStore(persona_dir / "data" / "works.db")
    db.insert(older, content="def foo(): pass")
    db.insert(newer, content="A small story about a lighthouse keeper.")
    return older.id, newer.id


def test_get_self_works_returns_list(persona_dir: Path) -> None:
    """GET /self/works returns the list of recent works (newest first)."""
    older_id, newer_id = _seed_two_works(persona_dir)
    with _client(persona_dir) as c:
        r = c.get("/self/works")
        assert r.status_code == 200
        body = r.json()
        assert "works" in body
        ids = [w["id"] for w in body["works"]]
        assert ids == [newer_id, older_id]


def test_get_self_works_filter_by_type(persona_dir: Path) -> None:
    older_id, newer_id = _seed_two_works(persona_dir)
    with _client(persona_dir) as c:
        r = c.get("/self/works?type=story")
        assert r.status_code == 200
        ids = [w["id"] for w in r.json()["works"]]
        assert ids == [newer_id]


def test_get_self_works_search_finds_match(persona_dir: Path) -> None:
    older_id, newer_id = _seed_two_works(persona_dir)
    with _client(persona_dir) as c:
        r = c.get("/self/works/search?q=lighthouse")
        assert r.status_code == 200
        ids = [w["id"] for w in r.json()["works"]]
        assert ids == [newer_id]


def test_get_self_works_search_with_type_filter(persona_dir: Path) -> None:
    _seed_two_works(persona_dir)
    with _client(persona_dir) as c:
        r = c.get("/self/works/search?q=lighthouse&type=code")
        assert r.status_code == 200
        # No 'lighthouse' in code-type → empty
        assert r.json()["works"] == []


def test_get_self_works_by_id_returns_full_content(persona_dir: Path) -> None:
    older_id, newer_id = _seed_two_works(persona_dir)
    with _client(persona_dir) as c:
        r = c.get(f"/self/works/{newer_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == newer_id
        assert body["content"] == "A small story about a lighthouse keeper."
        assert body["title"] == "The Lighthouse"


def test_get_self_works_by_id_unknown_returns_404(persona_dir: Path) -> None:
    _seed_two_works(persona_dir)
    with _client(persona_dir) as c:
        r = c.get("/self/works/zzzzzzzzzzzz")
        assert r.status_code == 404


def test_get_self_works_empty_when_no_db(persona_dir: Path) -> None:
    """No works.db yet — list returns empty array, not 500."""
    with _client(persona_dir) as c:
        r = c.get("/self/works")
        assert r.status_code == 200
        assert r.json()["works"] == []
