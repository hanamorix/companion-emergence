"""Bridge endpoints — sync TestClient against an in-memory FastAPI app."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(persona_dir: Path) -> TestClient:
    """Build the FastAPI app pinned to a tmp persona and return a TestClient."""
    app = build_app(persona_dir=persona_dir, client_origin="tests")
    return TestClient(app)


def _patch_fake_provider(monkeypatch, reply: str = "default reply"):
    """Replace get_provider so the lifespan returns a stub that yields `reply`.

    Patches brain.bridge.server.get_provider (not build_provider — Task 4 used
    get_provider). Must be called BEFORE opening a TestClient so the lifespan
    picks up the stub at app startup.
    """
    import brain.bridge.server as srv
    from brain.bridge.chat import ChatResponse

    class _Fake:
        def name(self):
            return "fake"

        def chat(self, messages, *, tools=None, options=None):
            return ChatResponse(content=reply, tool_calls=[])

        def generate(self, prompt, *, system=None):
            return reply

    monkeypatch.setattr(srv, "get_provider", lambda _name: _Fake())


def test_health_returns_ok(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["liveness"] == "ok"
        assert body["persona"] == persona_dir.name
        assert "uptime_s" in body
        assert body["sessions_active"] == 0
        assert "pending_alarms" in body  # from walk_persona/compute_pending_alarms
        assert body["supervisor_thread"] in ("not-started", "alive", "dead")


def test_session_new_returns_uuid_and_tracks_state(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.post("/session/new", json={"client": "tests"})
        assert r.status_code == 200
        body = r.json()
        sid = body["session_id"]
        assert len(sid) == 36  # uuid4 string
        assert body["persona"] == persona_dir.name
        assert "created_at" in body

        # /state/{sid} should now find it
        r2 = c.get(f"/state/{sid}")
        assert r2.status_code == 200
        s = r2.json()
        assert s["session_id"] == sid
        assert s["turns"] == 0
        assert s["history_len"] == 0
        assert s["in_flight"] is False


def test_state_returns_404_for_unknown_session(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.get("/state/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404


def test_health_sessions_active_increments_after_new(persona_dir: Path):
    with _make_client(persona_dir) as c:
        c.post("/session/new", json={"client": "tests"})
        c.post("/session/new", json={"client": "tests"})
        r = c.get("/health")
        assert r.json()["sessions_active"] == 2


# ---------------------------------------------------------------------------
# Task 5 — /chat, /sessions/close, /stream
# ---------------------------------------------------------------------------


def test_chat_404_unknown_session(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.post("/chat", json={"session_id": "no-such-sid", "message": "hi"})
        assert r.status_code == 404


def test_chat_round_trip_with_fake_provider(persona_dir: Path, monkeypatch):
    """Mock provider returns a fixed reply; verify /chat returns it and history persists."""
    _patch_fake_provider(monkeypatch, reply="hello, hana")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        r = c.post("/chat", json={"session_id": sid, "message": "hi"})
        assert r.status_code == 200
        body = r.json()
        assert body["reply"] == "hello, hana"
        assert body["turn"] == 1
        assert "duration_ms" in body

        s = c.get(f"/state/{sid}").json()
        assert s["turns"] == 1
        assert s["history_len"] == 2  # user + assistant


def test_sessions_close_returns_ingest_report(persona_dir: Path, monkeypatch):
    """Close after a chat: should return committed/deduped/etc counts."""
    _patch_fake_provider(monkeypatch, reply="goodbye")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        c.post("/chat", json={"session_id": sid, "message": "see you"})
        r = c.post("/sessions/close", json={"session_id": sid})
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == sid
        assert "committed" in body
        assert "deduped" in body
        assert "errors" in body


def test_stream_round_trip(persona_dir: Path, monkeypatch):
    """WS /stream/{sid} sends started, reply_chunks, done."""
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")  # no artificial delay in tests
    _patch_fake_provider(monkeypatch, reply="hello world from nell")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        with c.websocket_connect(f"/stream/{sid}") as ws:
            ws.send_json({"message": "hi"})
            frames = []
            while True:
                f = ws.receive_json()
                frames.append(f)
                if f.get("type") == "done":
                    break

    types = [f["type"] for f in frames]
    assert types[0] == "started"
    assert types[-1] == "done"
    assert "reply_chunk" in types
    chunked = "".join(f["text"] for f in frames if f.get("type") == "reply_chunk")
    assert chunked == "hello world from nell"
