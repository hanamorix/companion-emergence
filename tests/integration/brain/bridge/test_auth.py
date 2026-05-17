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
    (p / "persona_config.json").write_text('{"provider": "fake-auth", "searcher": "noop"}')
    (p / "emotion_vocabulary.json").write_text(_json.dumps({"version": 1, "emotions": []}))
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
    persona_dir_for_auth: Path,
    monkeypatch,
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


def test_http_cors_allows_tauri_dev_origin(persona_dir_for_auth: Path, monkeypatch):
    """Tauri dev uses http://localhost:1420; WebKit reports CORS blocks as 'Load failed'."""
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        r = c.options(
            "/persona/state",
            headers={
                "Origin": "http://localhost:1420",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "http://localhost:1420"
        assert "Authorization" in r.headers["access-control-allow-headers"]


def test_http_cors_allows_private_network_preflight_for_tauri_origin(
    persona_dir_for_auth: Path,
    monkeypatch,
):
    """Windows WebView2 can send Chromium Private Network Access preflights.

    Without Access-Control-Allow-Private-Network: true, Chromium reports the
    bridge request as a generic `Failed to fetch` even though the daemon is up
    and CLI health checks work.
    """
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        r = c.options(
            "/persona/state",
            headers={
                "Origin": "http://tauri.localhost",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization,content-type",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "http://tauri.localhost"
        assert r.headers["access-control-allow-private-network"] == "true"


def test_http_cors_still_rejects_untrusted_origin(persona_dir_for_auth: Path, monkeypatch):
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        r = c.options(
            "/persona/state",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert r.status_code == 400


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


def test_ws_rejects_wrong_subprotocol_token(persona_dir_for_auth: Path, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with c.websocket_connect("/events", subprotocols=["bearer", "wrong-token"]):
                pass
        assert exc_info.value.code == 4001


def test_ws_rejects_query_string_token(persona_dir_for_auth: Path, monkeypatch):
    """Bearer tokens should not be accepted from URL query strings."""
    from starlette.websockets import WebSocketDisconnect

    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with c.websocket_connect("/events?token=secret-token"):
                pass
        assert exc_info.value.code == 4001


def test_ws_accepts_correct_subprotocol_token(persona_dir_for_auth: Path, monkeypatch):
    _patch_provider(monkeypatch, _FakeProvider())
    app = build_app(persona_dir=persona_dir_for_auth, auth_token="secret-token")
    with TestClient(app) as c:
        with c.websocket_connect("/events", subprotocols=["bearer", "secret-token"]) as ws:
            assert ws.accepted_subprotocol == "bearer"
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
            with c.websocket_connect("/events", subprotocols=["bearer", "secret-token"]):
                pass
        assert exc_info.value.code == 4001
