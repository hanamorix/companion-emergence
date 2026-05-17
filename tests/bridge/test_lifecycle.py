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


def test_graceful_shutdown_drains_via_snapshot_and_preserves_buffer(
    persona_dir: Path,
    monkeypatch,
):
    """Lifespan teardown drains via snapshot_stale_sessions(silence_minutes=0)
    — non-destructive: memories extract durably, but the buffer + cursor
    survive on disk so the next bridge start can resume the sticky session.
    """
    from tests.bridge.test_endpoints import _patch_fake_provider

    _patch_fake_provider(monkeypatch, reply="bye")

    drained: list = []
    from brain.ingest import pipeline as ingest_pipeline

    real_snapshot = ingest_pipeline.snapshot_stale_sessions

    def spy(persona_dir, *, silence_minutes, **kw):
        drained.append(silence_minutes)
        return real_snapshot(persona_dir, silence_minutes=silence_minutes, **kw)

    monkeypatch.setattr(ingest_pipeline, "snapshot_stale_sessions", spy)

    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    with _client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        c.post("/chat", json={"session_id": sid, "message": "hi"})

        # Sanity: the buffer file exists while the bridge is live.
        buffer_path = persona_dir / "active_conversations" / f"{sid}.jsonl"
        assert buffer_path.exists(), "buffer should exist after a /chat"

    # __exit__ has run.
    # 1. The drain ran with silence_minutes=0 (the shutdown signal).
    assert 0 in drained or 0.0 in drained, f"silence_minutes=0 not seen; saw: {drained}"
    # 2. The buffer SURVIVED — sticky-session contract. The 24h
    #    finalize cadence handles genuinely-stale buffers on a later run.
    assert buffer_path.exists(), (
        "buffer must survive graceful shutdown (sticky-session contract); "
        "snapshot_stale_sessions is non-destructive"
    )


def test_graceful_shutdown_records_dirty_when_snapshot_drain_raises(
    persona_dir: Path,
    monkeypatch,
):
    """A catastrophic shutdown drain exception must still arm dirty recovery."""
    from brain.bridge import state_file
    from tests.bridge.test_endpoints import _patch_fake_provider

    _patch_fake_provider(monkeypatch, reply="bye")
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona=persona_dir.name,
            pid=12345,
            port=51234,
            started_at="2026-05-13T00:00:00Z",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="tests",
        ),
    )

    def boom(*args, **kwargs):
        raise RuntimeError("snapshot exploded")

    monkeypatch.setattr("brain.bridge.server._drain_sessions_blocking", boom)

    with _client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        c.post("/chat", json={"session_id": sid, "message": "hi"})

    after = state_file.read(persona_dir)
    assert after is not None
    assert after.drain_errors >= 1
    assert after.shutdown_clean is False


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


def test_clean_shutdown_with_drain_errors_triggers_recovery(persona_dir: Path, monkeypatch):
    """Phase 1.F: a clean exit that nonetheless reported drain ingest errors
    must NOT mask itself as fully clean. The next start should rerun
    recovery on the orphan buffers."""
    from brain.bridge import daemon, state_file
    from brain.ingest.buffer import ingest_turn

    s = state_file.BridgeState(
        persona=persona_dir.name,
        pid=999_999,
        port=51234,
        started_at="2026-04-28T10:00:00Z",
        stopped_at="2026-04-28T10:30:00Z",
        # The masked-as-clean shape from the bug we're fixing.
        shutdown_clean=False,
        client_origin="cli",
        drain_errors=2,
    )
    state_file.write(persona_dir, s)

    ingest_turn(persona_dir, {"session_id": "orphan", "speaker": "user", "text": "hi"})

    drained_sessions: list = []
    monkeypatch.setattr(
        "brain.bridge.daemon.close_stale_sessions",
        lambda persona_dir, **kw: drained_sessions.append(kw.get("silence_minutes")) or [],
    )

    daemon.run_recovery_if_needed(persona_dir)
    assert drained_sessions, "recovery must fire for drain_errors > 0"


def test_recovery_needed_predicate_drain_errors_arm(tmp_path: Path):
    """Phase 1.F: recovery_needed() returns True when drain_errors > 0
    even if shutdown_clean was somehow True (defensive — if the runner
    flipped clean=True before the lifespan had a chance to write
    drain_errors, the saved drain_errors still wins)."""
    from brain.bridge import state_file

    persona_dir = tmp_path
    persona_dir.mkdir(exist_ok=True)
    s = state_file.BridgeState(
        persona=persona_dir.name,
        pid=999_999,
        port=51234,
        started_at="2026-04-28T10:00:00Z",
        stopped_at="2026-04-28T10:30:00Z",
        shutdown_clean=True,  # masked clean
        client_origin="cli",
        drain_errors=1,
    )
    state_file.write(persona_dir, s)
    # PID 999_999 is dead → predicate should return True via the
    # drain_errors arm.
    assert state_file.recovery_needed(persona_dir) is True


def test_write_clean_shutdown_keeps_dirty_when_drain_errors(tmp_path: Path):
    """Phase 1.F: _write_clean_shutdown must NOT flip clean=True when
    drain_errors > 0. The next bridge start re-runs recovery on the
    orphan buffers."""
    from brain.bridge import state_file
    from brain.bridge.runner import _write_clean_shutdown

    persona_dir = tmp_path
    persona_dir.mkdir(exist_ok=True)
    s = state_file.BridgeState(
        persona=persona_dir.name,
        pid=12345,
        port=51234,
        started_at="2026-04-28T10:00:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
        drain_errors=3,
    )
    state_file.write(persona_dir, s)

    _write_clean_shutdown(persona_dir)

    after = state_file.read(persona_dir)
    assert after is not None
    assert after.shutdown_clean is False, "must remain dirty when drain_errors > 0"
    assert after.drain_errors == 3
    # pid + port still cleared so liveness checks see "not running".
    assert after.pid is None
    assert after.port is None
    assert after.stopped_at is not None


def test_write_clean_shutdown_flips_clean_when_drain_errors_zero(tmp_path: Path):
    """Sanity-check: the no-error path still flips to clean=True."""
    from brain.bridge import state_file
    from brain.bridge.runner import _write_clean_shutdown

    persona_dir = tmp_path
    persona_dir.mkdir(exist_ok=True)
    s = state_file.BridgeState(
        persona=persona_dir.name,
        pid=12345,
        port=51234,
        started_at="2026-04-28T10:00:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
        drain_errors=0,
    )
    state_file.write(persona_dir, s)

    _write_clean_shutdown(persona_dir)

    after = state_file.read(persona_dir)
    assert after is not None
    assert after.shutdown_clean is True
    assert after.pid is None


def test_legacy_state_file_without_drain_errors_field_loads(tmp_path: Path):
    """Backward-compat: a bridge.json written before drain_errors existed
    must still deserialise cleanly. The field defaults to 0."""
    import json

    from brain.bridge import state_file

    persona_dir = tmp_path
    persona_dir.mkdir(exist_ok=True)
    legacy = {
        "persona": "test",
        "pid": 12345,
        "port": 51234,
        "started_at": "2026-04-28T10:00:00Z",
        "stopped_at": None,
        "shutdown_clean": True,
        "client_origin": "cli",
        "auth_token": None,
        # drain_errors deliberately missing.
    }
    (persona_dir / "bridge.json").write_text(json.dumps(legacy))

    state = state_file.read(persona_dir)
    assert state is not None
    assert state.drain_errors == 0
