"""Bridge lifecycle — supervisor tick, idle-shutdown, graceful close."""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _client(persona_dir: Path, **kw) -> TestClient:
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests", **kw))


def test_supervisor_tick_emits_event(persona_dir: Path):
    """With tick_interval_s=0.2, /events sees supervisor_tick within ~3s."""
    with _client(persona_dir, tick_interval_s=0.2) as c:
        with c.websocket_connect("/events") as ws:
            ws.receive_json()  # connected greeting
            seen = False
            t0 = time.time()
            while time.time() - t0 < 3:
                f = ws.receive_json()
                if f["type"] == "supervisor_tick":
                    seen = True
                    break
            assert seen, "supervisor_tick not received within 3s"


def test_supervisor_prunes_old_empty_sessions(persona_dir: Path):
    """Empty app-created sessions have no buffer file, so supervisor must prune registry."""
    from brain.chat.session import all_sessions, reset_registry

    reset_registry()
    try:
        with _client(persona_dir, tick_interval_s=0.1, silence_minutes=0.001) as c:
            c.post("/session/new", json={"client": "tests"})
            assert len(all_sessions()) == 1
            deadline = time.time() + 3
            while time.time() < deadline and all_sessions():
                time.sleep(0.05)
            assert all_sessions() == []
    finally:
        reset_registry()


def test_graceful_shutdown_closes_active_sessions(persona_dir: Path, monkeypatch):
    """With one chat session, lifespan teardown drains via close_stale_sessions(silence_minutes=0)."""
    from tests.bridge.test_endpoints import _patch_fake_provider

    _patch_fake_provider(monkeypatch, reply="bye")

    closed: list = []
    from brain.ingest import pipeline as ingest_pipeline

    real_close = ingest_pipeline.close_stale_sessions

    def spy(persona_dir, *, silence_minutes, **kw):
        closed.append(silence_minutes)
        return real_close(persona_dir, silence_minutes=silence_minutes, **kw)

    monkeypatch.setattr(ingest_pipeline, "close_stale_sessions", spy)

    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    with _client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        c.post("/chat", json={"session_id": sid, "message": "hi"})
    # __exit__ has run — assert close_stale_sessions(silence_minutes=0) was called
    assert 0 in closed or 0.0 in closed, f"silence_minutes=0 not seen; saw: {closed}"


def test_check_idle_predicate(persona_dir: Path):
    """_check_idle is a pure predicate; no side effects."""
    from brain.bridge.server import _check_idle

    class FakeState:
        last_chat_at = None
        started_at = datetime.now(UTC) - timedelta(seconds=5)
        in_flight_locks: dict = {}

    s = FakeState()
    assert _check_idle(s, idle_shutdown_seconds=1) is True  # startup older than threshold

    s.started_at = datetime.now(UTC)
    assert _check_idle(s, idle_shutdown_seconds=1) is False  # fresh startup grace

    s.started_at = datetime.now(UTC) - timedelta(seconds=5)

    s.last_chat_at = datetime.now(UTC) - timedelta(seconds=5)
    assert _check_idle(s, idle_shutdown_seconds=1) is True

    s.last_chat_at = datetime.now(UTC)
    assert _check_idle(s, idle_shutdown_seconds=1) is False  # too fresh


def test_dirty_shutdown_drains_orphan_buffers(persona_dir: Path, monkeypatch):
    """Pre-write a stale bridge.json + a session buffer; recovery should drain it."""
    from brain.bridge import daemon, state_file
    from brain.ingest.buffer import ingest_turn

    s = state_file.BridgeState(
        persona=persona_dir.name,
        pid=999_999,
        port=51234,
        started_at="2026-04-28T10:00:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)

    ingest_turn(persona_dir, {"session_id": "orphan", "speaker": "user", "text": "hi"})
    ingest_turn(persona_dir, {"session_id": "orphan", "speaker": "assistant", "text": "hello"})

    drained_sessions: list = []

    def fake_drain(persona_dir, **kw):
        drained_sessions.append(kw.get("silence_minutes"))
        return []

    monkeypatch.setattr("brain.bridge.daemon.close_stale_sessions", fake_drain)

    daemon.run_recovery_if_needed(persona_dir)
    assert 0 in drained_sessions or 0.0 in drained_sessions


def test_clean_shutdown_skips_recovery(persona_dir: Path, monkeypatch):
    from brain.bridge import daemon, state_file

    s = state_file.BridgeState(
        persona=persona_dir.name,
        pid=None,
        port=None,
        started_at="2026-04-28T10:00:00Z",
        stopped_at="2026-04-28T10:30:00Z",
        shutdown_clean=True,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)

    called: list = []
    monkeypatch.setattr(
        "brain.bridge.daemon.close_stale_sessions",
        lambda persona_dir, **kw: called.append(kw.get("silence_minutes")) or [],
    )

    daemon.run_recovery_if_needed(persona_dir)
    assert called == []
