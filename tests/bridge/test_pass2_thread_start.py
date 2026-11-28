"""Where pass 2's own short-lived thread is started (#650 finding 5, L2 item 32): after a reply, and once at bridge start.

The thread body is `pass2_queue.drain_in_own_thread` (tests/unit/brain/chat/test_pass2_own_thread.py); here it is a recorder, so no
model call and no real drain. Synthetic tmp persona dirs only.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from brain.bridge import cli_throttle, server, shutdown_progress
from brain.chat import pass2_queue


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')
    return p


@pytest.fixture
def drain_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []
    monkeypatch.setattr(pass2_queue, "drain_in_own_thread", calls.append)
    return calls


def _queue_one(persona_dir: Path) -> None:
    pass2_queue.enqueue({"id": pass2_queue.new_record_id(), "kind": "test_probe"}, persona_dir=persona_dir)


def _state(persona_dir: Path, *, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(persona_dir=persona_dir, pass2_drain_enabled=enabled)


def _wait_for_thread(name: str = "pass2-drain") -> None:
    for t in [t for t in threading.enumerate() if t.name == name]:
        t.join()


def test_a_non_empty_queue_starts_one_counted_thread(persona_dir: Path, drain_calls: list[Path]) -> None:
    _queue_one(persona_dir)
    server._start_pass2_drain(_state(persona_dir))
    _wait_for_thread()
    assert drain_calls == [persona_dir]


def test_an_empty_queue_starts_nothing(persona_dir: Path, drain_calls: list[Path]) -> None:
    server._start_pass2_drain(_state(persona_dir))
    _wait_for_thread()
    assert drain_calls == []


def test_a_bridge_without_background_threads_starts_nothing(persona_dir: Path, drain_calls: list[Path]) -> None:
    _queue_one(persona_dir)
    server._start_pass2_drain(_state(persona_dir, enabled=False))
    _wait_for_thread()
    assert drain_calls == []


def test_the_thread_is_a_counted_task_while_it_runs_and_none_starts_after_the_stop(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    running = threading.Event()
    seen: list[list[str]] = []
    reg = shutdown_progress.registry

    def body(_pd: Path) -> None:
        seen.append(reg.snapshot())
        running.set()
        release.wait()

    monkeypatch.setattr(pass2_queue, "drain_in_own_thread", body)
    _queue_one(persona_dir)
    server._start_pass2_drain(_state(persona_dir))
    running.wait()
    release.set()
    _wait_for_thread()
    assert seen == [["pass-2 drain"]]
    assert reg.count() == 0
    # after the stop began nothing new starts (start_tracked_thread refuses)
    reg.seal()
    started: list[int] = []
    monkeypatch.setattr(pass2_queue, "drain_in_own_thread", lambda _pd: started.append(1))
    server._start_pass2_drain(_state(persona_dir))
    _wait_for_thread()
    assert started == []


def test_the_after_reply_helper_runs_the_start_off_the_loop_and_a_stop_is_not_an_error(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[str] = []

    async def fake_tracked(label: str, fn, /, *args, **kwargs):  # noqa: ANN001
        seen.append(label)
        return fn(*args, **kwargs)

    started: list[object] = []
    monkeypatch.setattr(server, "tracked_to_thread", fake_tracked)
    monkeypatch.setattr(server, "_start_pass2_drain", lambda s: started.append(s))
    state = _state(persona_dir)
    asyncio.run(server._start_pass2_drain_after_reply(state))
    assert seen == ["pass-2 start"] and started == [state]

    async def refused(label: str, fn, /, *args, **kwargs):  # noqa: ANN001
        raise cli_throttle.ShutdownDeferred("bridge is stopping")

    monkeypatch.setattr(server, "tracked_to_thread", refused)
    with caplog.at_level(logging.INFO, logger="brain.bridge.server"):
        asyncio.run(server._start_pass2_drain_after_reply(state))  # no exception
    assert [r.levelno for r in caplog.records if "pass-2 thread not started" in r.getMessage()] == [logging.INFO]


# --- the call sites ---------------------------------------------------------------------------------------------------


def _patch_provider(monkeypatch: pytest.MonkeyPatch, behaviour: dict[str, bool]) -> None:
    import brain.bridge.provider as provider_module
    from brain.bridge.chat import ChatResponse

    class _Fake:
        def name(self) -> str:
            return "fake"

        def chat(self, messages, *, tools=None, options=None):  # noqa: ANN001
            if behaviour["fail"]:
                raise RuntimeError("the provider failed")
            return ChatResponse(content="a reply", tool_calls=[])

        def generate(self, prompt, *, system=None) -> str:  # noqa: ANN001
            return "[]"

    monkeypatch.setattr(provider_module, "get_provider", lambda _name, **_kw: _Fake())


def test_an_http_reply_starts_the_thread_after_a_success_and_not_after_a_failure(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behaviour = {"fail": False}
    _patch_provider(monkeypatch, behaviour)
    after_reply: list[object] = []

    async def record(s) -> None:  # noqa: ANN001
        after_reply.append(s)

    monkeypatch.setattr(server, "_start_pass2_drain_after_reply", record)
    with TestClient(server.build_app(persona_dir=persona_dir, client_origin="tests")) as client:
        sid = client.post("/session/new", json={"client": "tests"}).json()["session_id"]
        behaviour["fail"] = True
        assert client.post("/chat", json={"session_id": sid, "message": "hi"}).status_code == 502
        assert after_reply == []
        behaviour["fail"] = False
        assert client.post("/chat", json={"session_id": sid, "message": "again"}).status_code == 200
        state = client.app.state.bridge
    assert after_reply == [state]


def test_a_websocket_reply_starts_the_thread_after_the_clean_close(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_provider(monkeypatch, {"fail": False})
    after_reply: list[object] = []
    started = threading.Event()

    async def record(s) -> None:  # noqa: ANN001
        after_reply.append(s)
        started.set()

    monkeypatch.setattr(server, "_start_pass2_drain_after_reply", record)
    with TestClient(server.build_app(persona_dir=persona_dir, client_origin="tests")) as client:
        sid = client.post("/session/new", json={"client": "tests"}).json()["session_id"]
        with client.websocket_connect(f"/stream/{sid}") as ws:
            ws.send_json({"message": "hi"})
            while ws.receive_json().get("type") not in ("done", "error"):
                pass
            # The hook runs on the server after the clean close and after the emotion drain (off the loop), so the client's
            # last frame does not mean it ran yet. Leaving the websocket block ends the handler's task (the test client cancels
            # it), so wait for the hook itself while the client is still attached.
            assert started.wait(timeout=30), "the after-reply hook never ran"
        state = client.app.state.bridge
    assert after_reply == [state]


# --- the lifespan -----------------------------------------------------------------------------------------------------


def test_a_bridge_with_background_threads_enables_it_and_starts_once_for_a_leftover(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brain.bridge import supervisor as supervisor_mod

    monkeypatch.setattr(supervisor_mod, "run_supervised", lambda stop_event, **_kw: stop_event.wait())
    started: list[object] = []
    monkeypatch.setattr(server, "_start_pass2_drain", lambda s: started.append(s.pass2_drain_enabled))
    app = server.build_app(persona_dir=persona_dir, client_origin="tests", background_threads=True)
    with TestClient(app):
        assert app.state.bridge.pass2_drain_enabled is True
    assert started == [True]


def test_a_bridge_without_background_threads_or_with_test_hooks_does_not_enable_it(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[object] = []
    monkeypatch.setattr(server, "_start_pass2_drain", lambda s: started.append(1))
    app = server.build_app(persona_dir=persona_dir, client_origin="tests")
    with TestClient(app):
        assert app.state.bridge.pass2_drain_enabled is False

    shutdown_progress.reset_for_tests()  # the first app's stop sealed the registry
    monkeypatch.setattr(
        server, "_lifespan_test_hooks", {"supervisor": lambda stop: stop.wait(), "start_tasks": lambda pd: None}
    )
    hooked = server.build_app(persona_dir=persona_dir, client_origin="tests", background_threads=True)
    with TestClient(hooked):
        assert hooked.state.bridge.pass2_drain_enabled is False
    assert started == []
