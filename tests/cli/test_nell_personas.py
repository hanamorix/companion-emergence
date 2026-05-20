"""Tests for `nell personas` subcommand.

Runs `uv run nell personas ...` as a subprocess with KINDLED_HOME pointing at
a temp dir. Bridge state is never active in these tests, so all personas show
"not running".
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _nell(tmp_home: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "KINDLED_HOME": str(tmp_home)}
    return subprocess.run(
        ["uv", "run", "nell", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).parent.parent.parent,  # repo root
    )


def _setup_persona(tmp_home: Path, name: str) -> Path:
    pd = tmp_home / "personas" / name
    pd.mkdir(parents=True, exist_ok=True)
    return pd


def test_personas_no_personas_exits_zero(tmp_path):
    """Zero personas installed — friendly empty message, not an error."""
    (tmp_path / "personas").mkdir(parents=True, exist_ok=True)
    result = _nell(tmp_path, "personas")
    assert result.returncode == 0, result.stderr


def test_personas_lists_installed_names(tmp_path):
    _setup_persona(tmp_path, "alice")
    _setup_persona(tmp_path, "bob")
    result = _nell(tmp_path, "personas")
    assert result.returncode == 0, result.stderr
    assert "alice" in result.stdout
    assert "bob" in result.stdout


def test_personas_json_is_valid(tmp_path):
    _setup_persona(tmp_path, "testpersona")
    result = _nell(tmp_path, "personas", "--json")
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert isinstance(rows, list)
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "testpersona"
    assert "bridge_running" in row
    assert row["bridge_running"] is False
