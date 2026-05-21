"""Integration: ingest real turns through ``ingest_turn`` → GET /chat/history
returns them through the bridge HTTP surface.

Unit tests for the endpoint hand-seed the JSONL file. This pass goes through
the canonical writer instead, so the on-disk schema stays in sync between
the writer (``brain.ingest.buffer.ingest_turn``) and the reader
(``GET /chat/history``). If someone renames ``speaker`` → ``role`` on
the writer side without updating the reader, this test catches it before
the renderer hydrates with empty bubbles.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.bridge.chat import ChatResponse
from brain.bridge.provider import LLMProvider
from brain.bridge.server import build_app
from brain.ingest.buffer import ingest_turn


class _FakeProvider(LLMProvider):
    def name(self) -> str:
        return "fake"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "[]"

    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content="ok", tool_calls=())


@pytest.fixture(autouse=True)
def _reset_sessions():
    from brain.chat.session import reset_registry

    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "nell"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "noop"}', encoding="utf-8"
    )
    return p


def test_chat_history_round_trip_via_buffer_writer(persona_dir: Path, monkeypatch):
    """Three turns written through ingest_turn surface intact over HTTP."""
    import brain.bridge.server as srv

    monkeypatch.setattr(srv, "get_provider", lambda _name, **_kw: _FakeProvider())

    sid = "sess_abc12345"
    ingest_turn(persona_dir, {"session_id": sid, "speaker": "user", "text": "hi"})
    ingest_turn(persona_dir, {"session_id": sid, "speaker": "assistant", "text": "hello"})
    ingest_turn(persona_dir, {"session_id": sid, "speaker": "user", "text": "how are you"})

    app = build_app(persona_dir=persona_dir, client_origin="tests")
    with TestClient(app) as c:
        r = c.get("/chat/history", params={"session_id": sid})

    assert r.status_code == 200
    body = r.json()
    assert body["next_before_turn"] is None
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert [m["content"] for m in body["messages"]] == ["hi", "hello", "how are you"]
    assert [m["turn"] for m in body["messages"]] == [1, 2, 3]
    # ts is whatever ingest_turn stamped — just confirm it's non-empty so
    # the renderer's "time" column has something to render.
    assert all(m["ts"] for m in body["messages"])
