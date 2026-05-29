"""Bridge endpoint tests for POST /persona/config/thinking."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def _make_client(tmp_path: Path):
    """Build a minimal bridge TestClient with the persona dir wired up."""
    from brain.bridge.server import build_app

    persona_dir = tmp_path / "personas" / "test"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "fake", "model": "sonnet"})
    )
    app = build_app(persona_dir=persona_dir, client_origin="tests")
    return TestClient(app), persona_dir


def test_post_thinking_config_persists_budget(tmp_path: Path):
    client, persona_dir = _make_client(tmp_path)
    with client:
        resp = client.post("/persona/config/thinking", json={"thinking_budget_tokens": 8000})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    from brain.persona_config import PersonaConfig
    config = PersonaConfig.load(persona_dir / "persona_config.json")
    assert config.thinking_budget_tokens == 8000


def test_post_thinking_config_clears_budget(tmp_path: Path):
    client, persona_dir = _make_client(tmp_path)
    with client:
        client.post("/persona/config/thinking", json={"thinking_budget_tokens": 8000})
        resp = client.post("/persona/config/thinking", json={"thinking_budget_tokens": None})
    assert resp.status_code == 200

    from brain.persona_config import PersonaConfig
    config = PersonaConfig.load(persona_dir / "persona_config.json")
    assert config.thinking_budget_tokens is None


def test_post_thinking_config_rejects_invalid(tmp_path: Path):
    client, _persona_dir = _make_client(tmp_path)
    with client:
        for bad in [-1, 0, "a lot"]:
            resp = client.post("/persona/config/thinking", json={"thinking_budget_tokens": bad})
            assert resp.status_code == 422, f"expected 422 for {bad!r}, got {resp.status_code}"


def test_persona_state_includes_thinking_budget(tmp_path: Path):
    """GET /persona/state must expose thinking_budget_tokens from config."""
    from dataclasses import replace

    from brain.persona_config import PersonaConfig

    client, persona_dir = _make_client(tmp_path)
    cfg = PersonaConfig.load(persona_dir / "persona_config.json")
    updated = replace(cfg, thinking_budget_tokens=5000)
    updated.save(persona_dir / "persona_config.json")

    with client:
        resp = client.get("/persona/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["connection"]["thinking_budget_tokens"] == 5000


def test_persona_state_thinking_budget_none_by_default(tmp_path: Path):
    client, _persona_dir = _make_client(tmp_path)
    with client:
        resp = client.get("/persona/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["connection"]["thinking_budget_tokens"] is None
