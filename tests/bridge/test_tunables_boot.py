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


def test_one_setting_module_that_cannot_be_imported_does_not_skip_the_boot_pass(tmp_path, monkeypatch, caplog):
    """Rev-4 review S1: each ``_SETTING_MODULES`` import is isolated, so one ImportError is logged and the file is still
    written (it used to share a ``try`` with ``sync_file()`` and skip the whole pass)."""
    import logging

    from brain import tunables
    from brain.bridge import server as server_mod

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()
    tunables.register("boot.test_key", 7, description="A key registered before boot.")
    monkeypatch.setattr(server_mod, "_SETTING_MODULES", ("brain.no_such_settings_module_zz", *server_mod._SETTING_MODULES))

    client, _persona_dir = _make_client(tmp_path)
    with caplog.at_level(logging.WARNING, logger="brain.bridge.server"), client:
        assert client.get("/health").status_code == 200

    data = toml_io.flatten_toml(toml_io.read_toml(tmp_path / "tunables.toml"))
    assert data["boot.test_key"] == 7  # the pass ran
    assert any("brain.no_such_settings_module_zz" in r.getMessage() for r in caplog.records), "the failed import must be logged"
    assert not any("boot pass skipped" in r.getMessage() for r in caplog.records)
    tunables._reset_for_tests()


def test_an_incomplete_registry_keeps_the_old_settings_file_for_the_next_start(tmp_path, monkeypatch):
    """When a settings module failed to import, the bridge's old-file fold must not discard the keys it cannot place."""
    import json as _json

    from brain import tunables
    from brain.bridge import server as server_mod

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()
    tunables.register("boot.test_key", 7, description="A key registered before boot.")
    legacy = tmp_path / "tunables.json"
    legacy.write_text(_json.dumps({"overrides": {"zz.only_another_module_knows": 5}}), encoding="utf-8")
    monkeypatch.setattr(server_mod, "_SETTING_MODULES", ("brain.no_such_settings_module_zz", *server_mod._SETTING_MODULES))

    client, persona_dir = _make_client(tmp_path)
    with client:
        assert client.get("/health").status_code == 200
    assert legacy.exists(), "the fold ran with an incomplete registry and discarded a key"

    monkeypatch.undo()  # the next start imports everything
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables.register("boot.test_key", 7, description="A key registered before boot.")
    from brain.bridge import shutdown_progress

    shutdown_progress.reset_for_tests()  # the next start is a new bridge process in production (the first stop sealed the registry)
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=None)
    with TestClient(app) as client:  # the key is no longer a setting, so it is discarded and the file archived
        assert client.get("/health").status_code == 200
    assert not legacy.exists() and any(p.name.startswith("tunables.json.migrated-") for p in tmp_path.iterdir())
    tunables._reset_for_tests()


def test_a_fresh_process_boot_writes_every_registered_key_with_its_default(tmp_path):
    """A real bridge boot in a fresh interpreter that imports only the server: the file holds EVERY key that a fresh
    interpreter importing every registering module registers, each at its default (C56, generalising the seven model keys).
    A subprocess, not this process: the registry here is cleared by other tests, and the point is that the boot itself (not
    a test's earlier import) registers the keys."""
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
    from tests.tunables_file import registered_literal_keys

    registered = registered_literal_keys(repo)  # follows aliased and `from brain.tunables import register` calls too
    assert "chat.idle_lull_seconds" in values and "bridge.stream_keepalive_seconds" in values
    missing = registered - set(values)
    assert missing == set(), f"the real boot did not register these keys, so sync_file did not write them: {sorted(missing)}"
