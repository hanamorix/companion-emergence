"""The emotion engine's drain at the bridge call sites (Engine 4, #579): the supervisor pass, lifespan, HTTP chat, WebSocket stream.

The engine is the shared recording stub (`emotion_engine_stub`); `("note_active", now)` entries and a shared order list show
what ran, when, and that an error path ran none.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from brain.bridge import cli_throttle, server
from brain.bridge.server import build_app
from brain.bridge.supervisor import run_folded


def _note_active_calls(stub) -> list[tuple]:  # noqa: ANN001
    return [call for call in stub.recorded_calls if call[0] == "note_active"]


def _record_into(stub, order: list[str]) -> None:  # noqa: ANN001
    real = stub.note_active
    stub.note_active = lambda now: (order.append("drain"), real(now))[1]


def _patch_provider(monkeypatch: pytest.MonkeyPatch, *, reply: str = "a reply", fails: bool = False) -> None:
    import brain.bridge.provider as provider_module
    from brain.bridge.chat import ChatResponse

    class _Fake:
        def name(self) -> str:
            return "fake"

        def chat(self, messages, *, tools=None, options=None):  # noqa: ANN001
            if fails:
                raise RuntimeError("the provider failed")
            return ChatResponse(content=reply, tool_calls=[])

        def generate(self, prompt, *, system=None) -> str:  # noqa: ANN001
            return "[]"

    monkeypatch.setattr(provider_module, "get_provider", lambda _name, **_kw: _Fake())


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')
    return p


# --- the supervisor pass (W1) ---------------------------------------------------------------------------------


def _run_supervisor(persona_dir: Path, monkeypatch: pytest.MonkeyPatch, order: list[str], *, passes: int) -> None:
    stop_event = threading.Event()
    done = {"n": 0}

    def kindled(*args: object, **kwargs: object) -> None:
        order.append("kindled")

    def count_pass_and_wait(timeout: float | None = None) -> bool:
        order.append("wait")
        done["n"] += 1
        if done["n"] >= passes:
            stop_event.set()
        return stop_event.is_set()

    monkeypatch.setattr("brain.bridge.supervisor._maybe_run_kindled_link_tick", kindled)
    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", lambda *a, **k: (0, 0))
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())
    monkeypatch.setattr("brain.engines.interest_sweep.run_sweep_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_voice_reflection_tick", lambda *a, **k: None)
    monkeypatch.setattr(stop_event, "wait", count_pass_and_wait)
    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
        tick_interval_s=0.01,
        heartbeat_interval_s=None,
        soul_review_interval_s=None,
        finalize_interval_s=None,
    )


def test_the_supervisor_drains_at_the_end_of_each_pass_after_the_kindled_block_and_before_the_wait(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _record_into(emotion_engine_stub, order)
    monkeypatch.setattr(cli_throttle, "reply_in_flight", lambda: False)
    _run_supervisor(persona_dir, monkeypatch, order, passes=2)
    assert order[:3] == ["kindled", "drain", "wait"]
    assert order.count("drain") == 2
    assert emotion_engine_stub.recorded_calls[0][1].tzinfo is not None


def test_the_supervisor_does_not_drain_while_a_reply_is_in_flight(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _record_into(emotion_engine_stub, order)
    monkeypatch.setattr(cli_throttle, "reply_in_flight", lambda: True)
    _run_supervisor(persona_dir, monkeypatch, order, passes=2)
    assert "drain" not in order and order.count("kindled") == 2


# --- lifespan (W3) ------------------------------------------------------------------------------------------------


def test_lifespan_drains_once_and_alone_after_the_vocabulary_load(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _record_into(emotion_engine_stub, order)
    real_load = server.ensure_persona_vocabulary_loaded
    monkeypatch.setattr(server, "ensure_persona_vocabulary_loaded", lambda *a, **k: (order.append("vocabulary"), real_load(*a, **k))[1])
    _patch_provider(monkeypatch)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    assert order == ["vocabulary", "drain"]
    assert len(_note_active_calls(emotion_engine_stub)) == 1


# --- HTTP chat and WebSocket stream (W2) ------------------------------------------------------------------------------


def test_http_chat_queues_the_drain_as_a_background_task_and_runs_it_after_the_reply(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _patch_provider(monkeypatch)
    real_add = BackgroundTasks.add_task
    queued: list[object] = []

    def add_task(self: BackgroundTasks, func, *args, **kwargs) -> None:  # noqa: ANN001
        queued.append(func)
        order.append("queued")
        real_add(self, func, *args, **kwargs)

    monkeypatch.setattr(BackgroundTasks, "add_task", add_task)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")) as client:
        sid = client.post("/session/new", json={"client": "tests"}).json()["session_id"]
        emotion_engine_stub.recorded_calls.clear()  # the start-up drain is not under test here
        _record_into(emotion_engine_stub, order)
        response = client.post("/chat", json={"session_id": sid, "message": "hi"})
    assert response.status_code == 200
    assert queued == [server._drain_emotion_engine]
    assert order == ["queued", "drain"]  # the drain is only queued by the handler; it runs once the response has gone out
    assert len(_note_active_calls(emotion_engine_stub)) == 1


def test_http_chat_queues_no_drain_on_an_error_path(persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_provider(monkeypatch, fails=True)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")) as client:
        sid = client.post("/session/new", json={"client": "tests"}).json()["session_id"]
        emotion_engine_stub.recorded_calls.clear()
        response = client.post("/chat", json={"session_id": sid, "message": "hi"})
    assert response.status_code == 502
    assert _note_active_calls(emotion_engine_stub) == []


def _stream_one_turn(client: TestClient) -> list[dict]:
    sid = client.post("/session/new", json={"client": "tests"}).json()["session_id"]
    frames: list[dict] = []
    with client.websocket_connect(f"/stream/{sid}") as ws:
        ws.send_json({"message": "hi"})
        while True:
            frame = ws.receive_json()
            frames.append(frame)
            if frame.get("type") in ("done", "error"):
                break
    return frames


def test_the_websocket_stream_drains_after_the_clean_close_on_a_success_path_only(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_provider(monkeypatch)
    order: list[str] = []
    from starlette.websockets import WebSocket

    real_close = WebSocket.close

    async def close(self: WebSocket, code: int = 1000, reason: str | None = None) -> None:
        order.append(f"close:{code}")
        await real_close(self, code, reason)

    monkeypatch.setattr(WebSocket, "close", close)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")) as client:
        emotion_engine_stub.recorded_calls.clear()
        _record_into(emotion_engine_stub, order)
        frames = _stream_one_turn(client)
    assert frames[-1]["type"] == "done"
    assert order[-2:] == ["close:1000", "drain"]
    assert len(_note_active_calls(emotion_engine_stub)) == 1


def test_the_websocket_stream_runs_no_drain_on_an_error_path(persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    _patch_provider(monkeypatch, fails=True)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")) as client:
        emotion_engine_stub.recorded_calls.clear()
        frames = _stream_one_turn(client)
    assert frames[-1]["type"] == "error"
    assert _note_active_calls(emotion_engine_stub) == []


# --- the helper (V2) ---------------------------------------------------------------------------------------------


def test_the_drain_helper_runs_note_active_once_off_the_loop_with_an_aware_time(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    async def fake_tracked(label: str, fn, /, *args, **kwargs):  # noqa: ANN001
        seen.append(label)
        return fn(*args, **kwargs)

    monkeypatch.setattr(server, "tracked_to_thread", fake_tracked)
    asyncio.run(server._drain_emotion_engine(persona_dir))
    assert seen == ["emotion drain"]
    (call,) = _note_active_calls(emotion_engine_stub)
    assert call[1].tzinfo is not None


def test_a_refused_registration_during_the_stop_is_logged_at_info_and_nothing_else_is_swallowed(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def refused(label: str, fn, /, *args, **kwargs):  # noqa: ANN001
        raise cli_throttle.ShutdownDeferred("bridge is stopping")

    monkeypatch.setattr(server, "tracked_to_thread", refused)
    with caplog.at_level(logging.INFO, logger="brain.bridge.server"):
        asyncio.run(server._drain_emotion_engine(persona_dir))  # no exception
    assert [r.levelno for r in caplog.records if "emotion drain not run" in r.getMessage()] == [logging.INFO]

    async def broken(label: str, fn, /, *args, **kwargs):  # noqa: ANN001
        raise ValueError("a different failure")

    monkeypatch.setattr(server, "tracked_to_thread", broken)
    with pytest.raises(ValueError, match="a different failure"):
        asyncio.run(server._drain_emotion_engine(persona_dir))
