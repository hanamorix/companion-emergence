"""POST /tools/search_memories — bridge-resident search endpoint (INC-4, S9/S10).

The full RSS/parity/failure-mode criteria (C1) need a real MCP child + live
bridge + real embedder/reranker models — that lives in
tests/bridge/test_search_via_bridge.py (requires_models, Linux local). This
file covers what's deterministic and model-free at the bridge-endpoint layer:
auth (C25), the transport wiring (same dispatch() call the in-process path
uses, so C1b's parity is a structural guarantee not a coincidence), argument
defaults, and a real (non-mocked) lexical-mode round trip — lexical mode
never touches the embedder/reranker, so it's a genuine end-to-end check with
no model stub needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


def _make_client(persona_dir: Path, *, auth_token: str | None = None) -> TestClient:
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=auth_token)
    return TestClient(app)


def test_search_memories_endpoint_requires_auth(persona_dir: Path) -> None:
    """C25: no bearer token -> 401, matches every other authed bridge route."""
    with _make_client(persona_dir, auth_token="secret") as c:
        r = c.post("/tools/search_memories", json={"query": "x"})
    assert r.status_code in (401, 403)


def test_search_memories_endpoint_rejects_wrong_token(persona_dir: Path) -> None:
    """C25: a wrong bearer token is rejected the same way as no token."""
    with _make_client(persona_dir, auth_token="secret") as c:
        r = c.post(
            "/tools/search_memories",
            json={"query": "x"},
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert r.status_code in (401, 403)


def test_search_memories_endpoint_accepts_correct_token(persona_dir: Path) -> None:
    with _make_client(persona_dir, auth_token="secret") as c:
        r = c.post(
            "/tools/search_memories",
            json={"query": "x", "mode": "lexical"},
            headers={"Authorization": "Bearer secret"},
        )
    assert r.status_code == 200


def test_search_memories_endpoint_calls_dispatch_not_a_reimplementation(
    persona_dir: Path,
) -> None:
    """The endpoint is a transport over the SAME dispatch() entry-point the
    in-process chat-engine tool loop uses (C1b's parity guarantee) — never a
    second implementation of search."""
    import brain.bridge.server as srv

    fake_result = {"query": "x", "mode": "lexical", "resolved_order": "relevance",
                   "emotion_filter": None, "count": 0, "memories": []}
    with patch.object(srv, "dispatch", return_value=fake_result) as mock_dispatch:
        with _make_client(persona_dir) as c:
            r = c.post("/tools/search_memories", json={"query": "x"})

    assert r.status_code == 200
    assert r.json() == fake_result
    mock_dispatch.assert_called_once()
    args, kwargs = mock_dispatch.call_args
    assert args[0] == "search_memories"
    assert args[1]["query"] == "x"
    assert kwargs["persona_dir"] == persona_dir
    assert isinstance(kwargs["store"], MemoryStore)
    assert isinstance(kwargs["hebbian"], HebbianMatrix)


def test_search_memories_endpoint_applies_schema_defaults(persona_dir: Path) -> None:
    """Body defaults (emotion=None, limit=5, exclude_ids=None, mode=semantic,
    order=relevance) match search_memories' own LLM-facing schema defaults —
    a bare {"query": ...} body must reach dispatch with all of them filled in."""
    import brain.bridge.server as srv

    fake_result = {"ok": True}
    with patch.object(srv, "dispatch", return_value=fake_result) as mock_dispatch:
        with _make_client(persona_dir) as c:
            r = c.post("/tools/search_memories", json={"query": "x"})

    assert r.status_code == 200
    _args, kwargs = mock_dispatch.call_args
    sent_arguments = mock_dispatch.call_args.args[1]
    assert sent_arguments == {
        "query": "x",
        "emotion": None,
        "limit": 5,
        "exclude_ids": None,
        "mode": "semantic",
        "order": "relevance",
    }


def test_search_memories_endpoint_closes_store_and_hebbian_per_call(
    persona_dir: Path,
) -> None:
    """H-A: per-call handles are opened and closed inside the request, not
    held across calls — never a shared/leaked connection."""
    import brain.bridge.server as srv

    fake_store = MagicMock(spec=MemoryStore)
    fake_hebbian = MagicMock(spec=HebbianMatrix)
    with (
        patch.object(srv, "MemoryStore", return_value=fake_store) as mock_store_cls,
        patch.object(srv, "HebbianMatrix", return_value=fake_hebbian),
        patch.object(srv, "dispatch", return_value={"ok": True}),
    ):
        with _make_client(persona_dir) as c:
            r = c.post("/tools/search_memories", json={"query": "x"})

    assert r.status_code == 200
    # build_app's own lifespan (walk_persona, persona/state, etc.) opens MemoryStore too — assert
    # our endpoint's own open happened with the right args, not that it was the ONLY open.
    mock_store_cls.assert_any_call(persona_dir / "memories.db", integrity_check=False)
    assert fake_store.close.call_count >= 1
    assert fake_hebbian.close.call_count >= 1


def test_search_memories_endpoint_real_lexical_round_trip(persona_dir: Path) -> None:
    """Genuine end-to-end: real store, real dispatch, real search_memories —
    lexical mode never touches the embedder/reranker, so this needs no model
    stub. Fail-first (pre-INC-4): this endpoint did not exist."""
    store = MemoryStore(persona_dir / "memories.db")
    try:
        m = Memory.create_new(
            content="Henryk likes long walks on the beach", memory_type="event", domain="d"
        )
        store.create(m)
    finally:
        store.close()

    with _make_client(persona_dir) as c:
        r = c.post(
            "/tools/search_memories",
            json={"query": "Henryk beach", "mode": "lexical"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "lexical"
    assert body["count"] == 1
    assert body["memories"][0]["id"] == m.id
