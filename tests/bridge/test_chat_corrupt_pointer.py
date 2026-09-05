"""POST /chat on a session with a corrupt rolled_to pointer is a 404, not a 500 (C3b iii)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _seed(persona_dir: Path, sid: str, *, with_buffer: bool) -> None:
    d = persona_dir / "active_conversations"
    if with_buffer:
        (d / f"{sid}.jsonl").write_text(
            json.dumps({"session_id": sid, "speaker": "user", "text": "hi", "ts": "2026-05-20T10:00:00Z"})
            + "\n",
            encoding="utf-8",
        )
    (d / f"{sid}.rolled_to").write_text('{"successor": "../x"}', encoding="utf-8")


def test_chat_invalid_successor_pointer_with_buffer_is_200_not_500(persona_dir: Path) -> None:
    """Pointer corrupt but the sid's own buffer still exists → /chat serves that session."""
    sid = str(uuid.uuid4())  # ChatReq requires a UUID-shaped session_id
    _seed(persona_dir, sid, with_buffer=True)
    app = build_app(persona_dir=persona_dir, client_origin="tests")
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/chat", json={"session_id": sid, "message": "hello"})
    assert r.status_code == 200, r.text


def test_chat_invalid_successor_pointer_without_buffer_is_404_not_500(persona_dir: Path) -> None:
    """Pointer corrupt and the old buffer already gone (post-rollover) → 404, never 500."""
    sid = str(uuid.uuid4())
    _seed(persona_dir, sid, with_buffer=False)
    app = build_app(persona_dir=persona_dir, client_origin="tests")
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/chat", json={"session_id": sid, "message": "hello"})
    assert r.status_code == 404, r.text
