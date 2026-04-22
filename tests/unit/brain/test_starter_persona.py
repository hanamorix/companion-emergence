"""Validation tests: shipped starter persona files exist and parse cleanly."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from brain import config


def _find_repo_root() -> Path:
    """Walk up from this file until we find pyproject.toml.

    More robust than counting `.parent` calls — survives future
    reorganisation of the tests/ directory tree (e.g. adding an
    integration/ subdirectory between unit/ and brain/).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find repo root (no pyproject.toml found)")


_REPO_ROOT = _find_repo_root()
_STARTER_THOUGHTFUL = _REPO_ROOT / "examples" / "starter-thoughtful"


def test_starter_thoughtful_dir_exists() -> None:
    """The starter-thoughtful example directory is present."""
    assert _STARTER_THOUGHTFUL.is_dir()


def test_starter_thoughtful_persona_toml_parses() -> None:
    """persona.toml is valid TOML."""
    toml_path = _STARTER_THOUGHTFUL / "persona.toml"
    assert toml_path.exists()
    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)
    assert data["persona"]["type"] == "starter"


@pytest.mark.parametrize(
    "filename",
    [
        "personality.json",
        "soul.json",
        "self_model.json",
        "voice.json",
        "emotions/extensions.json",
    ],
)
def test_starter_thoughtful_json_files_parse(filename: str) -> None:
    """All shipped JSON files in the starter persona are valid JSON."""
    json_path = _STARTER_THOUGHTFUL / filename
    assert json_path.exists()
    with json_path.open() as fh:
        json.load(fh)


def test_starter_thoughtful_loads_via_brain_config(
    clean_env: None,
) -> None:
    """brain.config.load_config accepts the starter persona dir."""
    result = config.load_config(_STARTER_THOUGHTFUL)
    assert result.persona_name == "starter-thoughtful"
    assert result.bridge_bind == "127.0.0.1:8765"
    assert result.provider == "ollama"
    # Confirm the TOML was actually read (guards against a silent fallback
    # to defaults if the file fails to parse or the bridge section is empty).
    assert result.source_trace["BRIDGE_BIND"] == "persona.toml"
