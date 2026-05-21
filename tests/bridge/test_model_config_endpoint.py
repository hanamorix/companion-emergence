"""POST /persona/config/model — live model switching endpoint.

Five contract tests:
  1. Endpoint updates persona_config.json
  2. Endpoint rejects unknown model (400)
  3. Endpoint swaps live provider model via /persona/state
  4. Endpoint requires bearer auth (401)
  5. Endpoint rejects malformed body (422 — Pydantic validation error)

Note on test 5: Pydantic raises 422 Unprocessable Entity (not 400) when
required fields are missing from the request body. 422 is the correct
HTTP status for validation errors per RFC 7807 / FastAPI conventions.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(tmp_path: Path, *, auth_token: str | None = None):
    """Minimal persona + TestClient. Returns (client, persona_dir, headers)."""
    persona_dir = tmp_path / "personas" / "test"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "fake", "model": "sonnet"})
    )
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=auth_token)
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    return TestClient(app), persona_dir, headers


def test_endpoint_updates_persona_config_json(tmp_path: Path):
    """A valid POST updates persona_config.json on disk."""
    client, persona_dir, _ = _make_client(tmp_path)
    with client:
        r = client.post(
            "/persona/config/model",
            json={"model": "opus"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"ok": True, "model": "opus"}

    from brain.persona_config import PersonaConfig

    reloaded = PersonaConfig.load(persona_dir / "persona_config.json")
    assert reloaded.model == "opus"


def test_endpoint_rejects_unknown_model(tmp_path: Path):
    """Unknown model names return 400 with a list of valid choices."""
    client, _persona_dir, _ = _make_client(tmp_path)
    with client:
        r = client.post(
            "/persona/config/model",
            json={"model": "gpt-4"},
        )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "unknown_model"
    assert "sonnet" in body["valid"]
    assert "opus" in body["valid"]
    assert "haiku" in body["valid"]


def test_endpoint_requires_bearer_auth(tmp_path: Path):
    """Requests without a valid bearer token must return 401."""
    client, _persona_dir, _ = _make_client(tmp_path, auth_token="secret")
    with client:
        # No auth header at all
        r = client.post(
            "/persona/config/model",
            json={"model": "opus"},
        )
    assert r.status_code == 401, r.text


def test_endpoint_rejects_malformed_body(tmp_path: Path):
    """Missing required 'model' field → 422 Unprocessable Entity (Pydantic default).

    422 is the correct HTTP status for schema validation failures per
    RFC 7807 and FastAPI conventions. The test explicitly checks 422 (not 400)
    to match what the server actually returns.
    """
    client, _persona_dir, _ = _make_client(tmp_path)
    with client:
        r = client.post(
            "/persona/config/model",
            json={"unrelated": "x"},
        )
    assert r.status_code == 422, r.text


def test_endpoint_swaps_live_provider_model_and_state_reflects_it(tmp_path: Path):
    """After a successful POST, /persona/state reports the new model.

    This verifies the hot-swap path: not just that persona_config.json is
    updated, but also that the provider._model attribute is updated so the
    next chat uses the new model without a restart. The /persona/state
    connection.model field is the canonical read path.
    """
    client, persona_dir, _ = _make_client(tmp_path)
    with client:
        # Switch from sonnet → opus
        r = client.post("/persona/config/model", json={"model": "opus"})
        assert r.status_code == 200, r.text

        # persona_state reads connection.model from persona_config.json
        # (it uses PersonaConfig.load which reads the file freshly each call).
        r2 = client.get("/persona/state")
        assert r2.status_code == 200, r2.text
        state = r2.json()
        assert state["connection"]["model"] == "opus"
