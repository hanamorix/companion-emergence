"""The REAL wiring (#529, C53 j, C37): server.py's lifespan folds a legacy ``tunables.json`` at KINDLED_HOME (never the
persona directory) into ``tunables.toml`` right after the boot pass, and the persona-model migration then writes into the same
file.

History: the ram-spike-fix INC-6 BLOCKER (caught before ship) was a first cut that passed ``persona_dir`` instead of
``paths.get_home()``, which silently no-oped every migration in production while every direct unit test passed because they
call the function with a fabricated path. This test exercises the REAL call site through ``build_app``'s lifespan instead.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.utils import toml_io


def _values(home: Path) -> dict:
    return toml_io.flatten_toml(toml_io.read_toml(home / "tunables.toml"))


def test_the_lifespan_folds_the_old_json_at_kindled_home_not_the_persona_dir(persona_dir: Path) -> None:
    kindled_home = Path(os.environ["KINDLED_HOME"])
    assert kindled_home != persona_dir, "sanity: fixtures must differ for this test to mean anything"
    kindled_home.mkdir(parents=True, exist_ok=True)
    (kindled_home / "tunables.json").write_text(
        json.dumps(
            {
                "_readme": "x",
                "defaults": {},
                "overrides": {"throttle.background_min_idle_seconds": 450.0, "model.interactive_chat": "claude-opus-5"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # A file at persona_dir/tunables.json must NOT exist beforehand: the fold cannot be "succeeding" against a stray copy.
    assert not (persona_dir / "tunables.json").exists()

    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass  # lifespan runs the boot pass and the fold synchronously on __enter__

    values = _values(kindled_home)
    assert values["chat.idle_lull_seconds"] == 450.0
    assert values["model.interactive_chat"] == "claude-opus-5"
    assert "throttle.background_min_idle_seconds" not in values
    assert not (kindled_home / "tunables.json").exists()
    assert len(list(kindled_home.glob("tunables.json.migrated-*"))) == 1
    assert not (persona_dir / "tunables.json").exists() and not (persona_dir / "tunables.toml").exists()


def test_a_second_boot_does_nothing_to_the_settings_file(persona_dir: Path) -> None:
    kindled_home = Path(os.environ["KINDLED_HOME"])
    (kindled_home).mkdir(parents=True, exist_ok=True)
    (kindled_home / "tunables.json").write_text(json.dumps({"overrides": {"chat.idle_lull_seconds": 120.0}}), encoding="utf-8")
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    toml = kindled_home / "tunables.toml"
    before, mtime = toml.read_bytes(), toml.stat().st_mtime_ns
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    assert toml.read_bytes() == before and toml.stat().st_mtime_ns == mtime


def _claude_persona(name: str, model: str) -> Path:
    """A throwaway persona under the sandbox home (the migration scans personas there) with a saved model."""
    from brain import paths

    d = paths.get_persona_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "active_conversations").mkdir(exist_ok=True)
    (d / "persona_config.json").write_text(
        json.dumps({"provider": "claude-cli", "searcher": "noop", "model": model}), encoding="utf-8"
    )
    return d


def test_the_persona_models_land_in_the_toml_through_the_real_lifespan() -> None:
    kindled_home = Path(os.environ["KINDLED_HOME"])
    persona = _claude_persona("wirer", "opus")
    with TestClient(build_app(persona_dir=persona, client_origin="tests")):
        pass
    assert _values(kindled_home)["model.interactive_chat"] == "claude-opus-5"
    assert "model" not in json.loads((persona / "persona_config.json").read_text(encoding="utf-8"))


def test_the_boot_with_an_old_json_and_a_saved_persona_model_leaves_the_toml_holding_the_chat_value() -> None:
    """C53 (j): old JSON with a model.interactive_chat entry plus a throwaway persona with a saved model, one real boot."""
    kindled_home = Path(os.environ["KINDLED_HOME"])
    kindled_home.mkdir(parents=True, exist_ok=True)
    (kindled_home / "tunables.json").write_text(
        json.dumps({"overrides": {"model.interactive_chat": "claude-opus-5"}}), encoding="utf-8"
    )
    persona = _claude_persona("wirer", "claude-sonnet-4-6")
    with TestClient(build_app(persona_dir=persona, client_origin="tests")):
        pass
    # the old file's setting is the install's; the persona's saved model is discarded (the install's value was already changed)
    assert _values(kindled_home)["model.interactive_chat"] == "claude-opus-5"
    assert "model" not in json.loads((persona / "persona_config.json").read_text(encoding="utf-8"))
    assert not (kindled_home / "tunables.json").exists()
