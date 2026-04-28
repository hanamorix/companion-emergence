"""Bridge auth tests — H-C ephemeral bearer token + WS Origin allowlist."""
from __future__ import annotations

import json as _json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.bridge.chat import ChatResponse
from brain.bridge.provider import LLMProvider
from brain.bridge.server import build_app


class _FakeProvider(LLMProvider):
    def __init__(self, reply: str = "ok"):
        self._reply = reply

    def name(self):
        return "fake-auth"

    def generate(self, prompt, *, system=None):
        return self._reply

    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content=self._reply, tool_calls=[])


@pytest.fixture(autouse=True)
def _reset_session_registry():
    from brain.chat.session import reset_registry
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def persona_dir_for_auth(tmp_path: Path) -> Path:
    p = tmp_path / "auth-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text(
        '{"provider": "fake-auth", "searcher": "noop"}'
    )
    (p / "emotion_vocabulary.json").write_text(
        _json.dumps({"version": 1, "emotions": []})
    )
    return p


def _patch_provider(monkeypatch, provider: LLMProvider) -> None:
    import brain.bridge.server as srv
    monkeypatch.setattr(srv, "get_provider", lambda _name: provider)


# ---------- HTTP auth ----------


def test_http_no_auth_when_token_disabled(persona_dir_for_auth: Path, monkeypatch):
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token=None)
    with TestClient(app) as c:
        r = c.post("/session/new", json={"client": "tests"})
        assert r.status_code == 200


def test_http_rejects_request_without_token(persona_dir_for_auth: Path, monkeypatch):
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        r = c.post("/session/new", json={"client": "tests"})
        assert r.status_code == 401
        assert "missing bearer token" in r.json()["detail"]


def test_http_rejects_wrong_token(persona_dir_for_auth: Path, monkeypatch):
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        r = c.post(
            "/session/new",
            json={"client": "tests"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert r.status_code == 401
        assert "invalid token" in r.json()["detail"]


def test_http_rejects_malformed_authorization_header(
    persona_dir_for_auth: Path, monkeypatch,
):
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        r = c.post(
            "/session/new",
            json={"client": "tests"},
            headers={"Authorization": "Basic c2VjcmV0OnNlY3JldA=="},
        )
        assert r.status_code == 401


def test_http_accepts_correct_token(persona_dir_for_auth: Path, monkeypatch):
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        r = c.post(
            "/session/new",
            json={"client": "tests"},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert r.status_code == 200
        assert "session_id" in r.json()


def test_http_health_also_protected(persona_dir_for_auth: Path, monkeypatch):
    """Even /health requires auth — Tauri's liveness probe reads the token."""
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        r_no = c.get("/health")
        assert r_no.status_code == 401
        r_yes = c.get("/health", headers={"Authorization": "Bearer secret-token"})
        assert r_yes.status_code == 200


# ---------- WebSocket auth ----------


def test_ws_no_auth_when_token_disabled(persona_dir_for_auth: Path, monkeypatch):
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token=None)
    with TestClient(app) as c:
        with c.websocket_connect("/events") as ws:
            f = ws.receive_json()
            assert f["type"] == "connected"


def test_ws_rejects_no_token(persona_dir_for_auth: Path, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with c.websocket_connect("/events"):
                pass
        assert exc_info.value.code == 4001


def test_ws_rejects_wrong_token(persona_dir_for_auth: Path, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with c.websocket_connect("/events?token=wrong-token"):
                pass
        assert exc_info.value.code == 4001


def test_ws_accepts_correct_token(persona_dir_for_auth: Path, monkeypatch):
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        with c.websocket_connect("/events?token=secret-token") as ws:
            f = ws.receive_json()
            assert f["type"] == "connected"


def test_ws_rejects_disallowed_origin(persona_dir_for_auth: Path, monkeypatch):
    """Origin not in allowlist → close 4001 even with valid token."""
    from starlette.websockets import WebSocketDisconnect

    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(
        persona_dir=persona_dir_for_auth,
        auth_token="secret-token",
        allowed_origins=("tauri://localhost",),  # exclude "null"
    )
    with TestClient(app) as c:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with c.websocket_connect("/events?token=secret-token"):
                pass
        assert exc_info.value.code == 4001
