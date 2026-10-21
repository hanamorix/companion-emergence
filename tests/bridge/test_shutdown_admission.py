"""After a stop begins the bridge admits no new work; work already admitted finishes and is waited for (C8, C40, C53;
D1, D21, Q-B)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.bridge import cli_throttle, shutdown_progress
from brain.bridge.server import build_app
from brain.bridge.shutdown import BridgeShutdownController


@pytest.fixture(autouse=True)
def _fresh():
    cli_throttle.reset()
    shutdown_progress.reset_for_tests()
    yield
    cli_throttle.reset()
    shutdown_progress.reset_for_tests()


class _FakeServer:
    should_exit = False


def _controller(persona_dir: Path) -> BridgeShutdownController:
    c = BridgeShutdownController()
    c.configure(persona_dir)
    c.bind_server(_FakeServer())
    return c


def _client(persona_dir: Path, controller: BridgeShutdownController) -> TestClient:
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests", shutdown_controller=controller))


def _new_session(c: TestClient) -> str:
    return c.post("/session/new", json={"client": "tests"}).json()["session_id"]


def test_after_the_stop_new_work_gets_bridge_stopping_and_health_still_answers(persona_dir: Path, monkeypatch):
    """C40: chat (HTTP and WebSocket), close, snapshot, the model check and other writers get "bridge is stopping"
    and run no engine code; /health still answers."""
    import brain.bridge.server as srv

    ran: list[str] = []
    monkeypatch.setattr(srv, "_respond_blocking", lambda *a, **k: ran.append("respond"))
    monkeypatch.setattr(srv, "_close_session_blocking", lambda *a, **k: ran.append("close"))
    monkeypatch.setattr(srv, "_snapshot_session_blocking", lambda *a, **k: ran.append("snapshot"))
    monkeypatch.setattr(srv, "_check_model_blocking", lambda *a, **k: ran.append("check") or ("ok", ""))
    ctrl = _controller(persona_dir)
    with _client(persona_dir, ctrl) as c:
        sid = _new_session(c)
        assert c.post("/supervisor/shutdown", json={"reason": "user_stop"}).status_code == 202
        assert ctrl.reason == "user_stop"
        for method, url, body in [
            ("post", "/chat", {"session_id": sid, "message": "hi"}),
            ("post", "/sessions/close", {"session_id": sid}),
            ("post", "/sessions/snapshot", {"session_id": sid}),
            ("post", "/persona/config/model", {"model": "claude-sonnet-5"}),
            ("post", "/session/new", {"client": "tests"}),
            ("post", "/supervisor/shutdown", None),
        ]:
            r = getattr(c, method)(url, json=body) if body is not None else getattr(c, method)(url)
            assert r.status_code == 503, (url, r.status_code)
            assert r.json()["code"] == "bridge_stopping", url
        assert c.get("/health").status_code == 200
        with c.websocket_connect(f"/stream/{sid}") as ws:
            ws.send_json({"message": "hi"})
            frame = ws.receive_json()
            assert frame.get("type") == "error" and frame.get("code") == "bridge_stopping"
    assert ran == []


def test_shutdown_reason_body_defaults_to_manual_restart(persona_dir: Path):
    ctrl = _controller(persona_dir)
    with _client(persona_dir, ctrl) as c:
        assert c.post("/supervisor/shutdown").status_code == 202
    assert ctrl.reason == "manual_restart"


def test_a_turn_admitted_before_the_stop_finishes_its_tool_loop(persona_dir: Path, monkeypatch):
    """C53 (D21): a chat turn admitted before the stop completes its tool loop after it (both chat() calls happen),
    persists its reply, and is a counted task until it ends."""
    import brain.bridge.provider as provider_module
    from brain.bridge.chat import ChatResponse, ToolCall

    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    in_first, release = threading.Event(), threading.Event()
    calls: list[int] = []

    class _Fake:
        def name(self):
            return "fake"

        def chat(self, messages, *, tools=None, options=None):
            calls.append(1)
            if len(calls) == 1:
                in_first.set()
                assert release.wait(30)
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(id="t1", name="search_memories", arguments={"query": "rain"})],
                )
            return ChatResponse(content="the rain again", tool_calls=[])

        def generate(self, prompt, *, system=None):
            return "[]"

    monkeypatch.setattr(provider_module, "get_provider", lambda _name, **_kw: _Fake())
    ctrl = _controller(persona_dir)
    result: dict = {}
    with _client(persona_dir, ctrl) as c:
        sid = _new_session(c)
        t = threading.Thread(
            target=lambda: result.setdefault("r", c.post("/chat", json={"session_id": sid, "message": "rain?"}))
        )
        t.start()
        assert in_first.wait(30)
        assert c.post("/supervisor/shutdown", json={"reason": "app_restart"}).status_code == 202
        assert "chat reply" in shutdown_progress.registry.snapshot()
        release.set()
        t.join(30)
        assert result["r"].status_code == 200 and result["r"].json()["reply"] == "the rain again"
    assert len(calls) == 2
    buffer = (persona_dir / "active_conversations" / f"{sid}.jsonl").read_text(encoding="utf-8")
    assert "the rain again" in buffer
    assert shutdown_progress.registry.count() == 0


def test_the_stop_neither_drains_sessions_nor_runs_a_close_heartbeat(persona_dir: Path, monkeypatch):
    """C8 [Q-B]: neither shutdown step runs; open buffers and cursors are byte-identical."""
    import brain.bridge.server as srv
    from brain.ingest.buffer import ingest_turn

    called: list[str] = []
    for name in ("_drain_sessions_blocking", "_run_heartbeat_close"):
        if hasattr(srv, name):
            monkeypatch.setattr(srv, name, lambda *a, _n=name, **k: called.append(_n))
    from brain.ingest import pipeline

    monkeypatch.setattr(pipeline, "snapshot_stale_sessions", lambda *a, **k: called.append("snapshot") or [])
    sid = "sess_open"
    ingest_turn(persona_dir, {"session_id": sid, "speaker": "user", "text": "hello", "ts": "2026-01-01T10:00:00+00:00"})
    buf = persona_dir / "active_conversations" / f"{sid}.jsonl"
    before = sorted((p.name, p.read_bytes()) for p in (persona_dir / "active_conversations").iterdir() if p.is_file())
    ctrl = _controller(persona_dir)
    with _client(persona_dir, ctrl):
        ctrl.begin_stop("user_stop")
    after = sorted((p.name, p.read_bytes()) for p in (persona_dir / "active_conversations").iterdir() if p.is_file())
    assert called == []
    assert buf.exists() and after == before


def test_the_stop_waits_for_a_running_worker_without_a_time_limit(persona_dir: Path, monkeypatch):
    """C10 (in-process half): the lifespan ends only once every counted task has ended (no 30 s / 180 s cap)."""
    ctrl = _controller(persona_dir)
    release = threading.Event()
    handle_box: dict = {}
    exited = threading.Event()

    def run_client():
        with _client(persona_dir, ctrl):
            handle_box["h"] = shutdown_progress.registry.register("stub task")
            ctrl.begin_stop("user_stop")
        exited.set()

    t = threading.Thread(target=run_client, daemon=True)
    t.start()
    time.sleep(1.0)
    assert not exited.is_set(), "the lifespan ended while a task was still running"
    shutdown_progress.registry.unregister(handle_box["h"])
    release.set()
    assert exited.wait(30)
    assert shutdown_progress.registry.sealed


def test_a_stop_that_began_before_the_lifespan_keeps_the_flag_set(persona_dir: Path):
    """Review S1 (C1, C4): a signal applied between the runner's start and the lifespan's first lines began the stop;
    the lifespan must not clear the process-wide flag, or no gate refuses during that stop."""
    ctrl = _controller(persona_dir)
    assert ctrl.begin_stop("os_signal") is True
    assert cli_throttle.shutdown_requested()
    with _client(persona_dir, ctrl) as c:
        assert cli_throttle.shutdown_requested()
        assert c.get("/health").status_code == 200


def test_the_lifespan_backstop_keeps_a_pending_signal_as_the_reason(persona_dir: Path):
    """Review R2-S3 (C13): an OS signal noted but not yet applied by the monitor when the lifespan's backstop stop runs
    is recorded as "os_signal", not "unknown"."""
    ctrl = _controller(persona_dir)
    with _client(persona_dir, ctrl) as c:
        assert c.get("/health").status_code == 200
        ctrl.note_signal()  # the signal handler ran; the monitor has not polled yet
    assert ctrl.reason == "os_signal"


def _on_event_loop() -> bool:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_the_stop_request_writes_its_mark_off_the_event_loop(persona_dir: Path, monkeypatch):
    """Pre-merge review O-6: the "stopping" mark (a bridge.json write that can fsync) does not run on the event loop;
    it is still on disk before the 202 (C6)."""
    import os

    from brain import process_identity
    from brain.bridge import state_file

    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona=persona_dir.name, pid=os.getpid(), port=5000, started_at="t", stopped_at=None,
            shutdown_clean=False, client_origin="tests", pid_token=process_identity.own_token(), identity="token",
            phase="running",
        ),
    )
    ctrl = _controller(persona_dir)
    seen: list[bool] = []
    real_update = state_file.update

    def spy(pd, fn, *a, **k):
        seen.append(_on_event_loop())
        return real_update(pd, fn, *a, **k)

    monkeypatch.setattr(state_file, "update", spy)
    with _client(persona_dir, ctrl) as c:
        seen.clear()
        assert c.post("/supervisor/shutdown", json={"reason": "user_stop"}).status_code == 202
        marks = list(seen)
        assert state_file.read(persona_dir).phase == "stopping"
    assert marks and not any(marks), marks


def test_the_idle_stop_writes_its_mark_off_the_event_loop(persona_dir: Path):
    """Pre-merge review O-6 (idle half): the idle watcher asks for the stop from a worker thread, not the loop."""
    import asyncio
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    import brain.bridge.server as srv

    calls: list[bool] = []

    class _Ctrl:
        def request(self, reason: str) -> bool:
            calls.append(_on_event_loop())
            return True

    state = SimpleNamespace(
        last_chat_at=None, started_at=datetime.now(UTC) - timedelta(hours=1), in_flight_locks={},
        shutdown_controller=_Ctrl(),
    )
    asyncio.run(asyncio.wait_for(srv._idle_watcher(state, 0.01), timeout=10))
    assert calls == [False]
