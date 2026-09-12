"""#246 — auth expiry surfaces through /chat, /health and /persona/state (real provider, fake CLI)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from brain.bridge import provider_auth
from brain.bridge.provider import ClaudeCliProvider
from brain.bridge.server import build_app

LIVE = "Failed to authenticate: OAuth session expired and could not be refreshed"


class _FakePopen:
    def __init__(self, lines, rc=0, stderr=""):
        self.stdin = io.StringIO()
        self.stdout = iter(lines)
        self.stderr = io.StringIO(stderr)
        self._rc = rc
        self.pid = 4242

    def wait(self, timeout=None):
        return self._rc

    def poll(self):
        return self._rc

    def terminate(self):
        pass

    def kill(self):
        pass


def _result_line(is_error: bool, text: str) -> str:
    return json.dumps({"type": "result", "is_error": is_error, "result": text}) + "\n"


def _run(rc: int, is_error: bool, text: str) -> MagicMock:
    m = MagicMock()
    m.returncode = rc
    m.stdout = json.dumps({"is_error": is_error, "result": text})
    m.stderr = ""
    return m


def _client(persona_dir: Path, monkeypatch) -> TestClient:
    # Route every provider construction to the REAL ClaudeCliProvider; the CLI itself is faked.
    monkeypatch.setattr("brain.bridge.provider.get_provider", lambda *a, **k: ClaudeCliProvider(model="sonnet"))
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests"))


def test_auth_expiry_surfaces_and_recovers_through_the_bridge(persona_dir: Path, monkeypatch):
    with _client(persona_dir, monkeypatch) as c:
        sid = c.post("/session/new", json={}).json()["session_id"]

        # 1. the CLI reports an expired login on this chat turn
        with patch("subprocess.Popen", return_value=_FakePopen([_result_line(True, LIVE)])), \
             patch("subprocess.run", return_value=_run(1, True, LIVE)):
            r = c.post("/chat", json={"session_id": sid, "message": "hello"})
        assert r.status_code == 502
        assert r.json()["detail"] == "auth_expired"

        h = c.get("/health").json()
        assert h["provider_auth"]["status"] == "expired"
        assert h["provider_auth"]["since"] is not None
        assert "OAuth session expired" in h["provider_auth"]["detail"]
        assert c.get("/persona/state").json()["provider_auth_expired"] is True

        # 2. a clean frame on the next turn clears it (the production chat_stream path)
        with patch("subprocess.Popen", return_value=_FakePopen([_result_line(False, "hi there")])), \
             patch("subprocess.run", return_value=_run(0, False, "hi there")):
            r2 = c.post("/chat", json={"session_id": sid, "message": "again"})
        assert r2.status_code == 200
        assert c.get("/health").json()["provider_auth"]["status"] == "ok"
        assert c.get("/persona/state").json()["provider_auth_expired"] is False


def test_chat_error_code_is_classified_from_this_call_not_global_state(persona_dir: Path, monkeypatch):
    provider_auth.note_cli_failure(LIVE)  # global state already expired
    with _client(persona_dir, monkeypatch) as c:
        sid = c.post("/session/new", json={}).json()["session_id"]
        limit = "api_error_status=429; You've hit your session limit"
        with patch("subprocess.Popen", return_value=_FakePopen([_result_line(True, limit)])), \
             patch("subprocess.run", return_value=_run(1, True, limit)):
            r = c.post("/chat", json={"session_id": sid, "message": "hello"})
        assert r.status_code == 502
        assert r.json()["detail"] == "provider_failed"


def test_ws_error_frame_carries_auth_expired(persona_dir: Path, monkeypatch):
    from brain.bridge import server as server_mod

    assert server_mod._chat_error_code(RuntimeError(f"[claude_cli_error] {LIVE}")) == "auth_expired"
    assert server_mod._chat_error_code(RuntimeError("[claude_cli_exit] exit 1: api_error_status=429")) == "provider_failed"
