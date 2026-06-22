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


def test_relay_url_persists_and_echoes(tmp_path: Path):
    """POST with relay_url persists it and echoes kindled_relay_url in response."""
    client, persona_dir = _make_client(tmp_path)
    with client:
        r = client.post(
            "/persona/config/kindled-link",
            json={"enabled": True, "relay_url": "http://127.0.0.1:9000"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kindled_link_enabled"] is True
    assert body["kindled_relay_url"] == "http://127.0.0.1:9000"
    # Confirm persisted to disk
    from brain.persona_config import PersonaConfig
    cfg = PersonaConfig.load(persona_dir / "persona_config.json")
    assert cfg.kindled_relay_url == "http://127.0.0.1:9000"


def test_relay_url_none_clears_it(tmp_path: Path):
    """POST with relay_url=null clears a previously set URL."""
    client, persona_dir = _make_client(tmp_path)
    with client:
        client.post(
            "/persona/config/kindled-link",
            json={"enabled": True, "relay_url": "http://127.0.0.1:9000"},
        )
        r = client.post(
            "/persona/config/kindled-link",
            json={"enabled": True, "relay_url": None},
        )
    assert r.status_code == 200, r.text
    assert r.json()["kindled_relay_url"] is None
    from brain.persona_config import PersonaConfig
    cfg = PersonaConfig.load(persona_dir / "persona_config.json")
    assert cfg.kindled_relay_url is None


def test_kindled_link_status_returns_relay_health(tmp_path: Path):
    """GET /kindled-link/status returns relay_health fields + recovered=False when no flag."""
    client, _persona_dir = _make_client(tmp_path)
    with client:
        r = client.get("/kindled-link/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "relay_ok" in body
    assert "last_poll_ts" in body
    assert "last_push_ts" in body
    assert "degraded_peers" in body
    assert body["recovered"] is False


def test_kindled_link_status_recovered_flag(tmp_path: Path):
    """GET /kindled-link/status returns recovered=True when flag file exists."""
    client, persona_dir = _make_client(tmp_path)
    flag_dir = persona_dir / "kindled_link"
    flag_dir.mkdir(parents=True, exist_ok=True)
    (flag_dir / "recovered.flag").touch()
    with client:
        r = client.get("/kindled-link/status")
    assert r.status_code == 200, r.text
    assert r.json()["recovered"] is True


def test_kindled_link_status_requires_auth(tmp_path: Path):
    """GET /kindled-link/status is auth-gated."""
    client, _persona_dir = _make_client(tmp_path, auth_token="secret")
    with client:
        r = client.get("/kindled-link/status")
    assert r.status_code == 401, r.text
