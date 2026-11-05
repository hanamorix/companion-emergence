"""Bridge lifecycle — supervisor tick, idle-shutdown, graceful close."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _client(persona_dir: Path, **kw) -> TestClient:
    # This module tests the lifecycle itself (supervisor tick, pruning, shutdown drain),
    # so it opts back INTO the background threads the root conftest inhibits for
    # endpoint tests (hunts/bridge-order-pollution-flakes; C7 opt-in list).
    kw.setdefault("background_threads", True)
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


def test_supervisor_prunes_old_empty_sessions(
    persona_dir: Path, monkeypatch
) -> None:
    """Empty app-created sessions have no buffer file, so supervisor must prune registry.

    ram-spike-fix INC-9 (C39, S72, 2-plan §3.3a): the prune is a gated job of
    the central cadence function and prunes only empty sessions that PREDATE
    the current idle window (``created_at`` at or before the last message) —
    never one opened during it (fbdd3acc: the app opens a session on mount).
    So: a session opened before any message is kept; once a message lands
    AFTER it and the lull has passed, the next idle pass prunes it.
    """
    from brain.bridge import cli_throttle
    from brain.chat.session import all_sessions, reset_registry

    reset_registry()
    try:
        with _client(persona_dir, tick_interval_s=0.1) as c:
            c.post("/session/new", json={"client": "tests"})
            assert len(all_sessions()) == 1
            # No message yet: the session was opened during the current idle
            # window, so idle passes must NOT prune it (C39(ii)).
            time.sleep(0.5)
            assert len(all_sessions()) == 1
            # Backdate the session and put the last message after its creation
            # but more than a lull ago: it now predates the idle window.
            all_sessions()[0].created_at = datetime.now(UTC) - timedelta(hours=2)
            cli_throttle.mark_interactive_active(at=time.monotonic() - 3600.0)
            deadline = time.time() + 3
            while time.time() < deadline and all_sessions():
                time.sleep(0.05)
            assert all_sessions() == []
    finally:
        reset_registry()


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


def test_idle_watcher_uses_shutdown_controller_not_os_kill(monkeypatch):
    import asyncio
    from datetime import UTC, datetime, timedelta

    from brain.bridge.events import EventBus
    from brain.bridge.server import BridgeAppState, _idle_watcher

    class FakeController:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def request(self, reason: str) -> bool:
            self.reasons.append(reason)
            return True

    controller = FakeController()
    state = BridgeAppState(
        persona_dir=Path("."),
        persona="nell",
        client_origin="tests",
        started_at=datetime.now(UTC) - timedelta(seconds=10),
        provider=object(),
        event_bus=EventBus(),
        in_flight_locks={},
        shutdown_controller=controller,
    )

    monkeypatch.setattr(
        "brain.bridge.server.os.kill",
        lambda *_a: (_ for _ in ()).throw(AssertionError("must not signal")),
    )

    asyncio.run(_idle_watcher(state, idle_shutdown_seconds=0.01))

    assert controller.reasons == ["idle_timeout"]


def _buffered_cost_samples(persona_dir: Path, n: int) -> None:
    """Buffer `n` cost-series samples against the persona's memories.db, as a
    passive recall would, without flushing them."""
    from brain.memory import cost_series
    from brain.memory.store import MemoryStore

    store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
    try:
        series = cost_series.CostSeries("rerank")
        for i in range(1, n + 1):
            series.record(store, "m", 1000 * i, 0.1 * i)
    finally:
        store.close()


def _saved_cost_samples(persona_dir: Path) -> int:
    import sqlite3

    con = sqlite3.connect(persona_dir / "memories.db")
    try:
        return con.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0]
    finally:
        con.close()


@pytest.mark.slow(reason='1.1 s of CPU in its call phase on ubuntu in CI (wall 1.1 s); #540, J12')
def test_clean_stop_saves_the_buffered_cost_series_samples(persona_dir: Path) -> None:
    """S146/S164 (CS17e): the lifespan teardown flushes, so a normal stop does
    not lose the samples taken since the last lull. No supervisor runs here, so
    the teardown is the only flusher."""
    _buffered_cost_samples(persona_dir, 3)
    assert _saved_cost_samples(persona_dir) == 0
    with _client(persona_dir, background_threads=False):
        assert _saved_cost_samples(persona_dir) == 0, "nothing is written while the bridge runs"
    assert _saved_cost_samples(persona_dir) == 3


def test_a_failing_cost_series_flush_does_not_stop_the_teardown(persona_dir: Path, monkeypatch) -> None:
    from brain.memory import cost_series

    _buffered_cost_samples(persona_dir, 2)

    def boom(_store):
        raise RuntimeError("flush failed")

    monkeypatch.setattr(cost_series, "flush", boom)
    with _client(persona_dir, background_threads=False):
        pass
    assert _saved_cost_samples(persona_dir) == 0


def test_the_teardown_flush_runs_after_the_supervisor_has_stopped(persona_dir: Path, monkeypatch) -> None:
    """S164: the supervisor thread is the only other flusher, so the flush waits
    for it to stop. (#456 Q-B: no close heartbeat follows any more.)"""
    from brain.bridge import supervisor
    from brain.memory import cost_series

    order: list[str] = []

    def fake_supervisor(*, stop_event, **_kw):
        stop_event.wait(30)
        time.sleep(0.3)  # still "running" for a moment after the stop is requested
        order.append("supervisor-stopped")

    monkeypatch.setattr(supervisor, "run_folded", fake_supervisor)
    monkeypatch.setattr(cost_series, "flush", lambda _store: order.append("flush"))
    with _client(persona_dir, tick_interval_s=3600):
        pass
    assert order == ["supervisor-stopped", "flush"]
