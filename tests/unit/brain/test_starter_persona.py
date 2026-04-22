"""Integration test: the shipped starter personas load cleanly."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from brain import config

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
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
