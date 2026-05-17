"""Tests for GET /persona/feed bridge endpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(persona_dir: Path, auth_token: str | None = None) -> TestClient:
    app = build_app(persona_dir=persona_dir, auth_token=auth_token)
    return TestClient(app)


def _seed_minimal_persona(tmp_path: Path) -> Path:
    """Return a minimal persona dir that build_app lifespan accepts."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')
    return persona_dir


def _seed_dream(persona_dir: Path) -> None:
    """Seed a persona dir with one dream so build_feed returns one entry."""
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        store.create(
            Memory(
                id="d_seed",
                memory_type="dream",
                content="seeded dream",
                domain="dream",
                emotions={},
                tags=[],
                importance=0.5,
                score=0.5,
                created_at=datetime(2026, 5, 17, 1, 0, tzinfo=UTC),
                active=True,
            )
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


def test_persona_feed_requires_bearer_auth(tmp_path: Path) -> None:
    """Hitting /persona/feed without a token returns 401."""
    persona_dir = _seed_minimal_persona(tmp_path)
    with _make_client(persona_dir, auth_token="test-token") as c:
        resp = c.get("/persona/feed")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_persona_feed_returns_empty_entries_with_valid_auth(tmp_path: Path) -> None:
    """Authenticated GET against an empty persona returns 200 with empty entries list."""
    persona_dir = _seed_minimal_persona(tmp_path)
    with _make_client(persona_dir, auth_token="test-token") as c:
        resp = c.get("/persona/feed", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    assert resp.json() == {"entries": []}


def test_persona_feed_returns_seeded_entries(tmp_path: Path) -> None:
    """Authenticated GET surfaces FeedEntry rows from build_feed."""
    persona_dir = _seed_minimal_persona(tmp_path)
    _seed_dream(persona_dir)
    with _make_client(persona_dir, auth_token="test-token") as c:
        resp = c.get("/persona/feed", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert "entries" in body
    assert len(body["entries"]) == 1
    entry = body["entries"][0]
    assert entry["type"] == "dream"
    assert entry["opener"] == "I dreamed"
    assert entry["body"] == "seeded dream"
    assert entry["ts"] == "2026-05-17T01:00:00+00:00"
    assert entry["audit_id"] is None
