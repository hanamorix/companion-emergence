"""Bridge /events WS — server-pushed broadcast tests."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _client(persona_dir: Path) -> TestClient:
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests"))


def test_events_greets_with_connected(persona_dir: Path):
    with _client(persona_dir) as c:
        with c.websocket_connect("/events") as ws:
            f = ws.receive_json()
            assert f["type"] == "connected"
            assert f["subscribers"] >= 1


def test_events_receives_chat_done_after_chat(persona_dir: Path, monkeypatch):
    """A chat turn must emit chat_done on /events."""
    monkeypatch.setenv("NELL_STREAM_CHUNK_DELAY_MS", "0")
    from tests.bridge.test_endpoints import _patch_fake_provider

    _patch_fake_provider(monkeypatch, reply="hi back")
    with _client(persona_dir) as c:
        with c.websocket_connect("/events") as evt_ws:
            evt_ws.receive_json()  # drain connected greeting
            sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
            c.post("/chat", json={"session_id": sid, "message": "hello"})
            seen_types = set()
            for _ in range(20):
                f = evt_ws.receive_json()
                seen_types.add(f["type"])
                if f["type"] == "chat_done":
                    return
            raise AssertionError(f"chat_done never seen; saw: {seen_types}")
