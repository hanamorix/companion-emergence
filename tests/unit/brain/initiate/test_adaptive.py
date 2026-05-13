"""Tests for brain.initiate.adaptive — Bundle C adaptive-D layer."""
from __future__ import annotations

from pathlib import Path

from brain.initiate.adaptive import load_d_mode


def test_load_d_mode_missing_file_returns_stateless(tmp_path: Path):
    persona = tmp_path / "fresh"
    persona.mkdir()
    assert load_d_mode(persona) == "stateless"


def test_load_d_mode_stateless_explicit(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "d_mode.json").write_text('{"mode": "stateless"}')
    assert load_d_mode(persona) == "stateless"


def test_load_d_mode_adaptive(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "d_mode.json").write_text('{"mode": "adaptive"}')
    assert load_d_mode(persona) == "adaptive"


def test_load_d_mode_invalid_json_falls_back_to_stateless(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "d_mode.json").write_text("not json{")
    assert load_d_mode(persona) == "stateless"


def test_load_d_mode_unknown_value_falls_back_to_stateless(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "d_mode.json").write_text('{"mode": "experimental"}')
    assert load_d_mode(persona) == "stateless"


def test_load_d_mode_non_dict_json_falls_back_to_stateless(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "d_mode.json").write_text('"adaptive"')
    assert load_d_mode(persona) == "stateless"
