"""Through-path test: the bridge lifespan runs the settings boot pass (``sync_file``).

Spec: docs/superpowers/specs/2026-07-04-ops-tunables-design.md (reshaped in #529)

Contract: entering the real app lifespan (TestClient context manager) and hitting /health causes
$KINDLED_HOME/tunables.toml to exist with one entry per registered setting, proving sync_file() actually fires through the
FastAPI startup path, not just in isolation. C56: the keys the REAL boot registers are the same as a fresh interpreter importing
every module that registers one, and the boot writes every one of them.

Mirrors test_kindled_link_config_endpoint.py's build_app + TestClient harness.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.utils import toml_io


def _make_client(tmp_path: Path):
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop"})
    )
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=None)
    return TestClient(app), persona_dir


def test_bridge_boot_writes_the_settings_file(tmp_path, monkeypatch):
    from brain import tunables

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()
    tunables.register("boot.test_key", 7, description="A key registered before boot.")

    client, _persona_dir = _make_client(tmp_path)
    with client:
        r = client.get("/health")
        assert r.status_code == 200, r.text

    tunables_path = tmp_path / "tunables.toml"
    assert tunables_path.exists(), "tunables.toml was not written at boot"
    data = toml_io.flatten_toml(toml_io.read_toml(tunables_path))
    assert data["boot.test_key"] == 7

    tunables._reset_for_tests()


def test_a_fresh_process_boot_writes_every_registered_key_with_its_default(tmp_path):
    """A real bridge boot in a fresh interpreter that imports only the server: the file holds EVERY key that a fresh
    interpreter importing every registering module registers, each at its default (C56, generalising the seven model keys).
    A subprocess, not this process: the registry here is cleared by other tests, and the point is that the boot itself (not
    a test's earlier import) registers the keys."""
    import ast
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
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
        cwd=repo, env=env, check=True, capture_output=True, timeout=300,
    )
    values = toml_io.flatten_toml(toml_io.read_toml(tmp_path / "tunables.toml"))
    assert {k: v for k, v in values.items() if k.startswith("model.")} == {
        "model.interactive_chat": "claude-sonnet-5",
        "model.background_generative": "claude-sonnet-5",
        "model.background_classifier": "haiku",
        "model.background_housekeeping": "haiku",
        "model.compaction": "haiku",
        "model.self_model_articulate": "haiku",
        "model.attunement_detector": "claude-haiku-4-5",
    }
    # Every literal key registered anywhere in brain/ is in the file the real boot wrote: the boot imports every module
    # that registers one (spec s12.3, C56), so no key waits for another process to add it.
    registered: set[str] = set()
    for path in (repo / "brain").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "register"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tunables"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                registered.add(node.args[0].value)
    assert "chat.idle_lull_seconds" in values and "bridge.stream_keepalive_seconds" in values
    missing = registered - set(values)
    assert missing == set(), f"the real boot did not register these keys, so sync_file did not write them: {sorted(missing)}"
