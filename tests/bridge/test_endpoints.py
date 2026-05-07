"""Bridge endpoints — sync TestClient against an in-memory FastAPI app."""
from __future__ import annotations

from pathlib import Path

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
        assert body["health_scan"] == "ok"
        assert body["health_error"] is None


def test_health_reports_scan_failure(persona_dir: Path, monkeypatch):
    import brain.bridge.server as srv

    def boom(_persona_dir):
        raise OSError("scan broke")

    monkeypatch.setattr(srv, "walk_persona", boom)

    with _make_client(persona_dir) as c:
        r = c.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["liveness"] == "ok"
    assert body["health_scan"] == "failed"
    assert "scan broke" in body["health_error"]


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
        r = c.post(
            "/chat",
            json={"session_id": "00000000-0000-0000-0000-000000000000", "message": "hi"},
        )
        assert r.status_code == 404


def test_chat_rejects_invalid_session_id_shape(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.post("/chat", json={"session_id": "no-such-sid", "message": "hi"})
        assert r.status_code == 422


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
        assert body["metadata"]["persistence_ok"] is True

        s = c.get(f"/state/{sid}").json()
        assert s["turns"] == 1
        assert s["history_len"] == 2  # user + assistant


def test_chat_rejects_empty_and_oversized_messages(persona_dir: Path, monkeypatch):
    """HTTP chat should reject messages outside the supported size contract."""
    _patch_fake_provider(monkeypatch, reply="hello")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        assert c.post("/chat", json={"session_id": sid, "message": ""}).status_code == 422
        assert c.post("/chat", json={"session_id": sid, "message": "x" * 20001}).status_code == 422


def test_session_new_rejects_unknown_client_label(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.post("/session/new", json={"client": "surprise-browser"})
        assert r.status_code == 422


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
        assert "soul_queue_errors" in body
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
    assert frames[-1]["metadata"]["persistence_ok"] is True
    chunked = "".join(f["text"] for f in frames if f.get("type") == "reply_chunk")
    assert chunked == "hello world from nell"


def test_stream_accepts_bearer_websocket_subprotocol(persona_dir: Path, monkeypatch):
    """WS auth should not require putting the bearer token in the URL query string."""
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_fake_provider(monkeypatch, reply="hello")
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token="secret-token")
    with TestClient(app) as c:
        sid = c.post(
            "/session/new",
            json={"client": "tests"},
            headers={"Authorization": "Bearer secret-token"},
        ).json()["session_id"]
        with c.websocket_connect(
            f"/stream/{sid}", subprotocols=["bearer", "secret-token"]
        ) as ws:
            assert ws.accepted_subprotocol == "bearer"
            ws.send_json({"message": "hi"})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame.get("type") == "done":
                    break

    assert frames[-1]["type"] == "done"


def test_persona_state_endpoint_returns_aggregated_shape(persona_dir: Path):
    """GET /persona/state returns the aggregated panel data — emotions, body,
    interior, soul_highlight, mode. Auth-required like every other surface."""
    with _make_client(persona_dir) as c:
        r = c.get("/persona/state")
        assert r.status_code == 200
        body = r.json()
        # All five top-level fields present
        for k in ("persona", "emotions", "body", "interior", "soul_highlight", "mode"):
            assert k in body
        # Mode is "live" by default (provider-failover not yet wired)
        assert body["mode"] == "live"
        # Interior keys preserved even on a fresh persona
        for k in ("dream", "research", "heartbeat", "reflex"):
            assert k in body["interior"]


def test_persona_state_endpoint_requires_auth(persona_dir: Path):
    """No bearer token → 401, matches every other authed endpoint."""
    from brain.bridge.server import build_app
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token="secret")
    with TestClient(app) as c:
        r = c.get("/persona/state")  # no auth header
        assert r.status_code in (401, 403)


# ---------------------------------------------------------------------------
# image_shas — multimodal chat threading
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402 — late import groups with the image-tests block below

_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63606060600000000400015e36b8c80000000049454e44ae426082"
)


def test_chat_accepts_image_shas_and_records_in_buffer(
    persona_dir: Path, monkeypatch
):
    """POST /chat with image_shas: ingest buffer JSONL has the shas on the user turn."""
    _patch_fake_provider(monkeypatch, reply="I see you.")
    with _make_client(persona_dir) as c:
        # Upload a real image first so media_type lookup succeeds.
        up = c.post(
            "/upload",
            files={"file": ("p.png", _TINY_PNG, "image/png")},
        )
        assert up.status_code == 200
        sha = up.json()["sha"]
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        r = c.post(
            "/chat",
            json={"session_id": sid, "message": "look at this", "image_shas": [sha]},
        )
        assert r.status_code == 200, r.text
        # Buffer JSONL should have the user record with image_shas.
        from brain.ingest.buffer import read_session

        turns = read_session(persona_dir, sid)
        user_turn = next(t for t in turns if t["speaker"] == "user")
        assert user_turn["image_shas"] == [sha]


def test_chat_rejects_too_many_image_shas(persona_dir: Path, monkeypatch):
    _patch_fake_provider(monkeypatch, reply="ok")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        r = c.post(
            "/chat",
            json={
                "session_id": sid,
                "message": "lots",
                "image_shas": [hashlib.sha256(str(i).encode()).hexdigest() for i in range(9)],
            },
        )
        assert r.status_code == 422


def test_chat_image_shas_default_empty_works(persona_dir: Path, monkeypatch):
    """No image_shas key — backward compat, behaves like text-only chat."""
    _patch_fake_provider(monkeypatch, reply="ok")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        r = c.post("/chat", json={"session_id": sid, "message": "no images"})
        assert r.status_code == 200
        from brain.ingest.buffer import read_session

        turns = read_session(persona_dir, sid)
        user_turn = next(t for t in turns if t["speaker"] == "user")
        assert "image_shas" not in user_turn


def test_chat_with_missing_image_sha_still_completes(persona_dir: Path, monkeypatch):
    """Sha references a file that doesn't exist — chat still completes; image is dropped."""
    _patch_fake_provider(monkeypatch, reply="ok")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        ghost = "f" * 64
        r = c.post(
            "/chat",
            json={"session_id": sid, "message": "ghost image", "image_shas": [ghost]},
        )
        # Chat must succeed — the engine logs and skips the missing image,
        # passing the user's text portion through unchanged.
        assert r.status_code == 200


def test_stream_accepts_image_shas_in_request_frame(persona_dir: Path, monkeypatch):
    """WS /stream/{sid} accepts image_shas alongside message; buffer carries them."""
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_fake_provider(monkeypatch, reply="seen")
    with _make_client(persona_dir) as c:
        up = c.post(
            "/upload",
            files={"file": ("p.png", _TINY_PNG, "image/png")},
        )
        sha = up.json()["sha"]
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        with c.websocket_connect(f"/stream/{sid}") as ws:
            ws.send_json({"message": "look", "image_shas": [sha]})
            while True:
                f = ws.receive_json()
                if f.get("type") == "done":
                    break
        from brain.ingest.buffer import read_session

        turns = read_session(persona_dir, sid)
        user_turn = next(t for t in turns if t["speaker"] == "user")
        assert user_turn["image_shas"] == [sha]


def test_stream_rejects_invalid_image_shas_field(persona_dir: Path, monkeypatch):
    """WS frame with non-list image_shas closes with an error frame."""
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_fake_provider(monkeypatch, reply="ok")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        with c.websocket_connect(f"/stream/{sid}") as ws:
            # Too many shas
            ws.send_json({"message": "x", "image_shas": ["a" * 64] * 9})
            f = ws.receive_json()
            assert f.get("type") == "error"
            assert f.get("code") == "invalid_image_shas"
