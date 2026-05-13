"""Bridge endpoints — sync TestClient against an in-memory FastAPI app."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.ingest.buffer import ingest_turn


def _make_client(persona_dir: Path) -> TestClient:
    """Build the FastAPI app pinned to a tmp persona and return a TestClient."""
    app = build_app(persona_dir=persona_dir, client_origin="tests")
    return TestClient(app)


def _patch_fake_provider(monkeypatch, reply: str = "default reply", extraction: str = "[]"):
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
            return extraction

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


def test_state_rejects_malformed_path_session_id(persona_dir: Path):
    with _make_client(persona_dir) as c:
        assert c.get("/state/not-a-uuid").status_code == 422
        assert c.get("/state/../../etc/passwd").status_code == 404


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
        assert body["persistence_ok"] is True
        assert body["persistence_error"] is None
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


def test_sessions_close_with_report_errors_stays_retryable(persona_dir: Path, monkeypatch):
    """A retryable ingest report must not remove the live session registry entry."""
    from brain.ingest.types import IngestReport

    calls = 0

    def fake_close(*args, **kwargs):
        nonlocal calls
        calls += 1
        report = IngestReport(session_id=args[1])
        if calls == 1:
            report.errors = 1
        return report

    monkeypatch.setattr("brain.bridge.server._close_session_blocking", fake_close)

    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        first = c.post("/sessions/close", json={"session_id": sid})
        assert first.status_code == 502
        detail = first.json()["detail"]
        assert detail["closed"] is False
        assert detail["errors"] == 1

        # The same session id is still known and can be retried.
        assert c.get(f"/state/{sid}").status_code == 200
        second = c.post("/sessions/close", json={"session_id": sid})
        assert second.status_code == 200
        assert second.json()["closed"] is True


def test_sessions_close_exception_stays_retryable(persona_dir: Path, monkeypatch):
    calls = 0

    def fake_close(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("extractor exploded")
        from brain.ingest.types import IngestReport

        return IngestReport(session_id=args[1])

    monkeypatch.setattr("brain.bridge.server._close_session_blocking", fake_close)

    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        first = c.post("/sessions/close", json={"session_id": sid})
        assert first.status_code == 502
        assert first.json()["detail"]["closed"] is False

        second = c.post("/sessions/close", json={"session_id": sid})
        assert second.status_code == 200
        assert second.json()["closed"] is True


def test_stream_rejects_malformed_path_session_id(persona_dir: Path, monkeypatch):
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_fake_provider(monkeypatch, reply="hello")
    with _make_client(persona_dir) as c:
        with c.websocket_connect("/stream/not-a-uuid") as ws:
            frame = ws.receive_json()
            assert frame == {"type": "error", "code": "invalid_session_id", "done": True}


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
    assert frames[-1]["persistence_ok"] is True
    assert frames[-1]["persistence_error"] is None
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


def test_stream_with_reply_to_audit_id_transitions_audit_to_replied_explicit(
    persona_dir: Path, monkeypatch,
):
    """Bundle A #4: WS /stream payload with ``reply_to_audit_id`` triggers a
    server-side audit transition + memory re-render atomically with the chat
    turn — no renderer-side POST /initiate/state needed.
    """
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_fake_provider(monkeypatch, reply="ack")
    _seed_audit_row(persona_dir, audit_id="ia_streamreply", state="delivered")

    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        with c.websocket_connect(f"/stream/{sid}") as ws:
            ws.send_json(
                {"message": "yes, I felt it too", "reply_to_audit_id": "ia_streamreply"},
            )
            while True:
                f = ws.receive_json()
                if f.get("type") == "done":
                    break

    rows = _read_audit_rows(persona_dir)
    target = next(r for r in rows if r["audit_id"] == "ia_streamreply")
    assert target["delivery"]["current_state"] == "replied_explicit"


def test_stream_without_reply_to_audit_id_leaves_audit_untouched(
    persona_dir: Path, monkeypatch,
):
    """Sanity: a chat turn without ``reply_to_audit_id`` doesn't mutate any
    audit row. Prevents the server-side transition from over-firing.
    """
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_fake_provider(monkeypatch, reply="ok")
    _seed_audit_row(persona_dir, audit_id="ia_untouched", state="delivered")

    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        with c.websocket_connect(f"/stream/{sid}") as ws:
            ws.send_json({"message": "regular message"})
            while True:
                f = ws.receive_json()
                if f.get("type") == "done":
                    break

    rows = _read_audit_rows(persona_dir)
    target = next(r for r in rows if r["audit_id"] == "ia_untouched")
    assert target["delivery"]["current_state"] == "delivered"


def test_stream_rejects_non_string_reply_to_audit_id(persona_dir: Path, monkeypatch):
    """Invalid ``reply_to_audit_id`` (non-string) closes with a typed error."""
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_fake_provider(monkeypatch, reply="ok")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        with c.websocket_connect(f"/stream/{sid}") as ws:
            ws.send_json({"message": "x", "reply_to_audit_id": 42})
            f = ws.receive_json()
            assert f.get("type") == "error"
            assert f.get("code") == "invalid_reply_to_audit_id"


def test_stream_closes_cleanly_after_done(persona_dir: Path, monkeypatch):
    """The WS Close frame after `done` carries code 1000 (not 1006).

    Regression guard for the 2026-05-07 wizard-validation finding:
    Hana saw 'ws closed (1006): unknown' in the chat panel after every
    successful chat round-trip because the bridge handler returned
    without explicitly closing the WS, so FastAPI tore down the
    underlying TCP without sending a Close frame and the browser
    reported abnormal closure.
    """
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_fake_provider(monkeypatch, reply="all good")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        with c.websocket_connect(f"/stream/{sid}") as ws:
            ws.send_json({"message": "hi"})
            saw_done = False
            while True:
                f = ws.receive_json()
                if f.get("type") == "done":
                    saw_done = True
                    break
            assert saw_done, "stream never sent the done frame"
            # After done the server should close cleanly. Starlette's
            # TestClient surfaces this via WebSocketDisconnect with the
            # close code on .code.
            from starlette.websockets import WebSocketDisconnect

            try:
                # Any further receive should raise with the close code.
                ws.receive_json()
                raise AssertionError("server did not close after done")
            except WebSocketDisconnect as exc:
                assert exc.code == 1000, f"expected clean close 1000, got {exc.code}"


# ---------------------------------------------------------------------------
# F-201 Phase B — /sessions/active + hydrate-on-miss
# ---------------------------------------------------------------------------


def _seed_buffer(persona_dir: Path, sid: str, *, last_age_hours: float, pairs: int = 1) -> None:
    """Seed an active_conversations/<sid>.jsonl with ``pairs`` user+assistant
    turns whose last ts is ``last_age_hours`` in the past."""
    now = datetime.now(UTC)
    last_ts = now - timedelta(hours=last_age_hours)
    for i in range(pairs):
        # Stamp earlier turns slightly before the last one — order matters
        # because /sessions/active reads the *last* line.
        u_ts = last_ts - timedelta(seconds=(pairs - i) * 2)
        a_ts = last_ts - timedelta(seconds=(pairs - i) * 2 - 1)
        if i == pairs - 1:
            a_ts = last_ts  # final assistant turn anchors the age window
        ingest_turn(persona_dir, {
            "session_id": sid, "speaker": "user", "text": f"u{i}",
            "ts": u_ts.isoformat(),
        })
        ingest_turn(persona_dir, {
            "session_id": sid, "speaker": "assistant", "text": f"a{i}",
            "ts": a_ts.isoformat(),
        })


def test_sessions_active_returns_null_when_no_buffers(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.get("/sessions/active")
        assert r.status_code == 200
        assert r.json() == {"session_id": None}


def test_sessions_active_returns_youngest_recent_session(persona_dir: Path):
    """Two recent buffers — return the one with the most recent last turn."""
    old_recent = "11111111-1111-1111-1111-111111111111"
    young_recent = "22222222-2222-2222-2222-222222222222"
    _seed_buffer(persona_dir, old_recent, last_age_hours=2.0)
    _seed_buffer(persona_dir, young_recent, last_age_hours=0.5)
    with _make_client(persona_dir) as c:
        r = c.get("/sessions/active")
        assert r.status_code == 200
        assert r.json() == {"session_id": young_recent}


def test_sessions_active_skips_stale_over_24h(persona_dir: Path):
    """Buffer older than 24h is not attach-eligible."""
    stale = "33333333-3333-3333-3333-333333333333"
    _seed_buffer(persona_dir, stale, last_age_hours=25.0)
    with _make_client(persona_dir) as c:
        r = c.get("/sessions/active")
        assert r.json() == {"session_id": None}


def test_sessions_active_picks_recent_over_stale(persona_dir: Path):
    stale = "44444444-4444-4444-4444-444444444444"
    recent = "55555555-5555-5555-5555-555555555555"
    _seed_buffer(persona_dir, stale, last_age_hours=48.0)
    _seed_buffer(persona_dir, recent, last_age_hours=1.0)
    with _make_client(persona_dir) as c:
        r = c.get("/sessions/active")
        assert r.json() == {"session_id": recent}


def test_state_hydrates_from_buffer_on_unknown_session(persona_dir: Path):
    """F-201 Phase B: /state/<sid> hydrates a session whose buffer exists
    on disk but whose in-memory entry was lost (bridge restart simulation)."""
    sid = "66666666-6666-6666-6666-666666666666"
    _seed_buffer(persona_dir, sid, last_age_hours=0.1, pairs=2)
    with _make_client(persona_dir) as c:
        r = c.get(f"/state/{sid}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == sid
        assert body["persona"] == persona_dir.name
        # 2 user + 2 assistant turns -> 2 pairs.
        assert body["turns"] == 2
        # history_len is the in-memory history list, which is empty after
        # hydration (engine reads the buffer for prompt context).
        assert body["history_len"] == 0


def test_chat_hydrates_from_buffer_on_unknown_session(persona_dir: Path, monkeypatch):
    """F-201 Phase B: /chat hydrates instead of 404 when a buffer survives a restart."""
    _patch_fake_provider(monkeypatch, reply="welcome back")
    sid = "77777777-7777-7777-7777-777777777777"
    _seed_buffer(persona_dir, sid, last_age_hours=0.1, pairs=1)
    with _make_client(persona_dir) as c:
        r = c.post("/chat", json={"session_id": sid, "message": "still here"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == sid
        assert body["reply"] == "welcome back"


def test_sessions_close_hydrates_from_buffer_on_unknown_session(
    persona_dir: Path, monkeypatch
):
    """F-201 Phase B: /sessions/close drains a buffer left over from a restart."""
    _patch_fake_provider(monkeypatch)
    sid = "88888888-8888-8888-8888-888888888888"
    _seed_buffer(persona_dir, sid, last_age_hours=0.1, pairs=1)
    with _make_client(persona_dir) as c:
        r = c.post("/sessions/close", json={"session_id": sid})
        assert r.status_code == 200, r.text
        assert r.json()["closed"] is True


# ── /initiate/state — renderer-driven state transitions ─────────────────────


def _seed_audit_row(
    persona_dir: Path,
    *,
    audit_id: str,
    state: str,
    subject: str = "the dream",
    tone_rendered: str = "the dream landed somewhere this morning",
) -> None:
    """Append one AuditRow with delivery.current_state preset."""
    try:
        from brain.initiate.audit import append_audit_row
        from brain.initiate.schemas import AuditRow
    except ModuleNotFoundError:
        import pytest
        pytest.skip("brain.initiate not available in public build")

    row = AuditRow(
        audit_id=audit_id,
        candidate_id=f"ic_{audit_id}",
        ts="2026-05-11T14:47:09+00:00",
        kind="message",
        subject=subject,
        tone_rendered=tone_rendered,
        decision="send_quiet",
        decision_reasoning="resonance is real",
        gate_check={"allowed": True, "reason": None},
        delivery={
            "delivered_at": "2026-05-11T14:47:09+00:00",
            "state_transitions": [
                {"to": "delivered", "at": "2026-05-11T14:47:09+00:00"},
            ],
            "current_state": state,
        },
    )
    append_audit_row(persona_dir, row)


def _read_audit_rows(persona_dir: Path) -> list[dict]:
    """Read initiate_audit.jsonl as raw dicts (for asserting current_state)."""
    import json

    path = persona_dir / "initiate_audit.jsonl"
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            out.append(json.loads(stripped))
    return out


def test_post_initiate_state_transition_records_audit_and_memory(
    persona_dir: Path,
) -> None:
    """POST /initiate/state — renderer reports a state event (read/dismissed)."""
    _seed_audit_row(persona_dir, audit_id="ia_001", state="delivered")

    with _make_client(persona_dir) as c:
        r = c.post(
            "/initiate/state",
            json={"audit_id": "ia_001", "new_state": "read"},
        )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "new_state": "read"}

    rows = _read_audit_rows(persona_dir)
    target = next(r for r in rows if r["audit_id"] == "ia_001")
    assert target["delivery"]["current_state"] == "read"


def test_post_initiate_state_rejects_unknown_state(persona_dir: Path) -> None:
    _seed_audit_row(persona_dir, audit_id="ia_001", state="delivered")
    with _make_client(persona_dir) as c:
        r = c.post(
            "/initiate/state",
            json={"audit_id": "ia_001", "new_state": "garbage"},
        )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /initiate/voice-edit/accept and /reject
# ---------------------------------------------------------------------------


def _seed_voice_edit_audit(
    persona_dir: Path,
    *,
    audit_id: str,
    old_text: str,
    new_text: str,
) -> None:
    """Append a voice_edit_proposal audit row with a one-line unified diff."""
    from brain.initiate.audit import append_audit_row
    from brain.initiate.schemas import AuditRow

    diff = f"- {old_text}\n+ {new_text}\n"
    row = AuditRow(
        audit_id=audit_id,
        candidate_id=f"ic_{audit_id}",
        ts="2026-05-11T14:47:09+00:00",
        kind="voice_edit_proposal",
        subject="a small edit to my voice",
        tone_rendered=(
            f"Proposing to change my voice: {old_text!r} -> {new_text!r}."
        ),
        decision="send_quiet",
        decision_reasoning="the pattern showed up three times",
        gate_check={"allowed": True, "reason": None},
        delivery={
            "delivered_at": "2026-05-11T14:47:09+00:00",
            "state_transitions": [
                {"to": "delivered", "at": "2026-05-11T14:47:09+00:00"},
            ],
            "current_state": "delivered",
        },
        diff=diff,
    )
    append_audit_row(persona_dir, row)


def test_post_voice_edit_accept_applies_diff_and_writes_three_places(
    persona_dir: Path,
) -> None:
    """Accept writes audit + memory + voice_evolution AND modifies nell-voice.md."""
    voice_path = persona_dir / "nell-voice.md"
    voice_path.write_text("line A\nold line\nline C\n")
    _seed_voice_edit_audit(
        persona_dir, audit_id="ia_ve_001",
        old_text="old line", new_text="new line",
    )
    with _make_client(persona_dir) as c:
        r = c.post(
            "/initiate/voice-edit/accept",
            json={"audit_id": "ia_ve_001", "with_edits": None},
        )
    assert r.status_code == 200, r.text
    body = voice_path.read_text()
    assert "new line" in body
    assert "old line" not in body

    from brain.soul.store import SoulStore
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        evolutions = store.list_voice_evolution()
    finally:
        store.close()
    assert len(evolutions) == 1
    assert evolutions[0].audit_id == "ia_ve_001"
    assert evolutions[0].new_text == "new line"
    assert evolutions[0].user_modified is False


def test_post_voice_edit_accept_with_edits_records_user_modified(
    persona_dir: Path,
) -> None:
    voice_path = persona_dir / "nell-voice.md"
    voice_path.write_text("line A\nold line\nline C\n")
    _seed_voice_edit_audit(
        persona_dir, audit_id="ia_ve_001",
        old_text="old line", new_text="new line proposed",
    )
    with _make_client(persona_dir) as c:
        r = c.post(
            "/initiate/voice-edit/accept",
            json={
                "audit_id": "ia_ve_001",
                "with_edits": "hana's tweaked line",
            },
        )
    assert r.status_code == 200, r.text
    assert "hana's tweaked line" in voice_path.read_text()

    from brain.soul.store import SoulStore
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        ev = store.list_voice_evolution()[0]
    finally:
        store.close()
    assert ev.user_modified is True
    assert ev.new_text == "hana's tweaked line"


def test_post_voice_edit_reject_records_dismissed_no_voice_write(
    persona_dir: Path,
) -> None:
    voice_path = persona_dir / "nell-voice.md"
    voice_path.write_text("line A\nold line\nline C\n")
    _seed_voice_edit_audit(
        persona_dir, audit_id="ia_ve_001",
        old_text="old line", new_text="new line",
    )
    with _make_client(persona_dir) as c:
        r = c.post(
            "/initiate/voice-edit/reject",
            json={"audit_id": "ia_ve_001"},
        )
    assert r.status_code == 200, r.text
    assert "old line" in voice_path.read_text()  # unchanged

    from brain.soul.store import SoulStore
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        evolutions = store.list_voice_evolution()
    finally:
        store.close()
    assert evolutions == []
