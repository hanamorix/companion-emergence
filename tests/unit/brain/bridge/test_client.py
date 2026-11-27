"""Tests for brain.bridge.client.post_to_bridge: the one call from another process to the running bridge (#801 step 4).

The search tool's own outcome tests (tests/unit/brain/mcp_server/test_tools.py) go through it too; these pin what is not specific to search:
the route is the caller's, and so is the timeout.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from brain.bridge import client, state_file


def _state(port: int | None = 4321, token: str | None = "tok") -> state_file.BridgeState:
    return state_file.BridgeState(
        persona="p", pid=1, port=port, started_at="2026-01-01T00:00:00+00:00", stopped_at=None,
        shutdown_clean=True, client_origin="cli", auth_token=token,
    )


class _Resp:
    def __init__(self, status: int, body: object = None) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> object:
        return self._body


def test_the_route_the_timeout_the_body_and_the_token_are_the_callers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def post(url, *, json, headers, timeout, trust_env):
        seen.update(url=url, json=json, headers=headers, timeout=timeout, trust_env=trust_env)
        return _Resp(200, {"ok": True})

    monkeypatch.setattr(client.state_file, "read", lambda _pd: _state())
    monkeypatch.setattr(client.httpx, "post", post)
    assert client.post_to_bridge(tmp_path, "/vocabulary/lookup", {"names": ["joy"]}, timeout=7.5) == {"ok": True}
    assert seen == {
        "url": "http://127.0.0.1:4321/vocabulary/lookup",
        "json": {"names": ["joy"]},
        "headers": {"Authorization": "Bearer tok"},
        "timeout": 7.5,
        "trust_env": False,
    }


def test_no_token_sends_no_authorization_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(client.state_file, "read", lambda _pd: _state(token=None))
    monkeypatch.setattr(client.httpx, "post", lambda url, **kw: (seen.update(kw), _Resp(200, {}))[1])
    client.post_to_bridge(tmp_path, "/x", {}, timeout=1.0)
    assert seen["headers"] == {}


@pytest.mark.parametrize(
    ("state", "raises", "status", "expected"),
    [
        (None, None, None, {"error": "bridge unreachable"}),
        ("noport", None, None, {"error": "bridge unreachable"}),
        ("up", httpx.ConnectError("refused"), None, {"error": "bridge unreachable"}),
        ("up", httpx.ReadTimeout("slow"), None, {"error": "bridge timeout"}),
        ("up", None, 401, {"error": "bridge error 401"}),
        ("up", None, 500, {"error": "bridge error 500"}),
    ],
)
def test_the_three_failure_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state, raises, status, expected) -> None:
    monkeypatch.setattr(client.state_file, "read", lambda _pd: None if state is None else _state(port=None if state == "noport" else 4321))

    def post(*_a, **_k):
        if raises is not None:
            raise raises
        return _Resp(status)

    monkeypatch.setattr(client.httpx, "post", post)
    assert client.post_to_bridge(tmp_path, "/x", {}, timeout=1.0) == expected
