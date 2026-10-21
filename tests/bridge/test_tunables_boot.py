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


def test_a_fresh_process_boot_writes_all_seven_model_keys_with_their_defaults(tmp_path):
    """A real bridge boot in a fresh interpreter that imports only the server: the defaults section lists every
    model.* job line with its shipped default. A subprocess, not this process: the registry here is cleared by other
    tests, and the point is that the boot itself (not a test's earlier import) registers the keys."""
    import subprocess
    import sys

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "noop"}))
    code = (
        "import sys; from pathlib import Path; from fastapi.testclient import TestClient; "
        "from brain.bridge.server import build_app; "
        "app = build_app(persona_dir=Path(sys.argv[1]), client_origin='tests', auth_token=None); "
        "c = TestClient(app); c.__enter__(); assert c.get('/health').status_code == 200; c.__exit__(None, None, None)"
    )
    env = {**__import__("os").environ, "KINDLED_HOME": str(tmp_path)}
    env.pop("NELLBRAIN_HOME", None)
    subprocess.run(
        [sys.executable, "-c", code, str(persona_dir)],
        cwd=Path(__file__).resolve().parents[2], env=env, check=True, capture_output=True, timeout=300,
    )
    defaults = json.loads((tmp_path / "tunables.json").read_text(encoding="utf-8"))["defaults"]
    model_defaults = {k: v for k, v in defaults.items() if k.startswith("model.")}
    assert model_defaults == {
        "model.interactive_chat": "claude-sonnet-5",
        "model.background_generative": "claude-sonnet-5",
        "model.background_classifier": "haiku",
        "model.background_housekeeping": "haiku",
        "model.compaction": "haiku",
        "model.self_model_articulate": "haiku",
        "model.attunement_detector": "claude-haiku-4-5",
    }
