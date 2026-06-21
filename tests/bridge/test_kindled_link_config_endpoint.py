"""POST /persona/config/kindled-link — enable/disable Kindled-to-Kindled correspondence.

D3 criteria:
- POST with {"enabled": true} flips the persisted flag and returns it
- the route is auth-gated (no bearer token → 401)
- a non-bool enabled → 422

Mirrors test_notes_config_endpoint harness (build_app + TestClient + optional bearer auth).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(tmp_path: Path, *, auth_token: str | None = None):
    """Minimal persona + TestClient. Returns (client, persona_dir)."""
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop", "model": "sonnet"})
    )
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=auth_token)
    return TestClient(app), persona_dir


def test_enable_kindled_link_flips_flag(tmp_path: Path):
    client, persona_dir = _make_client(tmp_path)
    with client:
        r = client.post("/persona/config/kindled-link", json={"enabled": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kindled_link_enabled"] is True
    # Confirm persisted
    from brain.persona_config import PersonaConfig
    cfg = PersonaConfig.load(persona_dir / "persona_config.json")
    assert cfg.kindled_link_enabled is True


def test_kindled_link_requires_auth(tmp_path: Path):
    client, _persona_dir = _make_client(tmp_path, auth_token="secret")
    with client:
        r = client.post("/persona/config/kindled-link", json={"enabled": True})
    assert r.status_code == 401, r.text


def test_kindled_link_non_bool_enabled_is_422(tmp_path: Path):
    client, _persona_dir = _make_client(tmp_path)
    with client:
        r = client.post("/persona/config/kindled-link", json={"enabled": "yes"})
    assert r.status_code == 422, r.text
