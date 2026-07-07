"""Through-path test: bridge lifespan rewrites tunables.json defaults at boot.

Spec: docs/superpowers/specs/2026-07-04-ops-tunables-design.md

Contract: entering the real app lifespan (TestClient context manager) and
hitting /health causes $KINDLED_HOME/tunables.json to exist with "defaults"
and "overrides" keys — proving write_defaults_section() actually fires
through the FastAPI startup path, not just in isolation.

Mirrors test_kindled_link_config_endpoint.py's build_app + TestClient harness.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(tmp_path: Path):
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop", "model": "sonnet"})
    )
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=None)
    return TestClient(app), persona_dir


def test_bridge_boot_writes_tunables_defaults_section(tmp_path, monkeypatch):
    from brain import tunables

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()

    client, _persona_dir = _make_client(tmp_path)
    with client:
        r = client.get("/health")
        assert r.status_code == 200, r.text

    tunables_path = tmp_path / "tunables.json"
    assert tunables_path.exists(), "tunables.json was not written at boot"
    data = json.loads(tunables_path.read_text(encoding="utf-8"))
    assert "defaults" in data
    assert "overrides" in data

    tunables._reset_for_tests()
