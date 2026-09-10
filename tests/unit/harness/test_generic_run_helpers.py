"""Unit checks for the helpers in the live example ``tests/harness/examples/test_generic_run.py``.

The example itself needs a live ``claude`` and is skipped in CI, so its private helpers rotted
unobserved (#234: ``/session/new`` posted with no body → 422). These run everywhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx

from tests.harness.examples import test_generic_run as example


def test_new_session_sends_a_json_body(monkeypatch) -> None:
    """``NewSessionReq`` is a required Pydantic body — a bare POST is a 422 (#234)."""
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(200, json={"session_id": "sid-1"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    server = SimpleNamespace(host="127.0.0.1", port=8931)

    assert example._new_session(server) == "sid-1"
    assert captured["url"].endswith("/session/new")
    assert captured.get("json") == {"client": "tests"}
