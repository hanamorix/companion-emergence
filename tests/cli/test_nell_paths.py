"""Tests for `nell paths` subcommand.

Runs `uv run nell paths ...` as a subprocess with KINDLED_HOME pointed at
a temp dir. Each test installs exactly one persona so _resolve_persona_or_exit
takes the silent-single path. ~2s startup overhead per uv run call is expected.
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


def _setup_persona(tmp_home: Path, name: str = "testpersona") -> Path:
    """Create a minimal persona directory so _resolve_persona_or_exit succeeds."""
    pd = tmp_home / "personas" / name
    pd.mkdir(parents=True, exist_ok=True)
    return pd


def test_paths_table_exits_zero(tmp_path):
    _setup_persona(tmp_path)
    result = _nell(tmp_path, "paths")
    assert result.returncode == 0, result.stderr


def test_paths_table_contains_home_key(tmp_path):
    _setup_persona(tmp_path)
    result = _nell(tmp_path, "paths")
    assert result.returncode == 0, result.stderr
    assert "home" in result.stdout


def test_paths_table_contains_persona_dir_key(tmp_path):
    _setup_persona(tmp_path)
    result = _nell(tmp_path, "paths")
    assert result.returncode == 0, result.stderr
    assert "persona_dir" in result.stdout


def test_paths_json_is_valid_json(tmp_path):
    _setup_persona(tmp_path)
    result = _nell(tmp_path, "paths", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "home" in payload
    assert "persona_dir" in payload
    # Each value has path + exists keys
    assert "path" in payload["home"]
    assert "exists" in payload["home"]


def test_paths_single_key_prints_one_line(tmp_path):
    _setup_persona(tmp_path)
    result = _nell(tmp_path, "paths", "home")
    assert result.returncode == 0, result.stderr
    # Output should be a single path (one non-empty line)
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1


def test_paths_single_key_home_matches_kindled_home(tmp_path):
    _setup_persona(tmp_path)
    result = _nell(tmp_path, "paths", "home")
    assert result.returncode == 0, result.stderr
    printed = result.stdout.strip()
    assert printed == str(tmp_path)


def test_paths_unknown_key_exits_nonzero(tmp_path):
    _setup_persona(tmp_path)
    result = _nell(tmp_path, "paths", "nonexistent_key_xyz")
    assert result.returncode != 0


def test_paths_all_iterates_multiple_personas(tmp_path):
    _setup_persona(tmp_path, "alpha")
    _setup_persona(tmp_path, "beta")
    result = _nell(tmp_path, "paths", "--all")
    assert result.returncode == 0, result.stderr
    # Both persona names should appear in the output
    assert "alpha" in result.stdout
    assert "beta" in result.stdout
