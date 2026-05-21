"""End-to-end CLI persona resolution against a tmp KINDLED_HOME.

Drives `uv run nell status` as a subprocess with 0 / 1 / many installed
personas. Verifies the _resolve_persona_or_exit policy at the binary
boundary, not just the unit level.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _nell(args: list[str], kindled_home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "nell", *args],
        env={**os.environ, "KINDLED_HOME": str(kindled_home)},
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_status_zero_personas_exits_2(tmp_path: Path) -> None:
    (tmp_path / "personas").mkdir()
    r = _nell(["status"], tmp_path)
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "no persona" in r.stderr.lower()


def test_status_single_persona_runs_silently(tmp_path: Path) -> None:
    (tmp_path / "personas" / "mira").mkdir(parents=True)
    r = _nell(["status"], tmp_path)
    # exit 0 happy path; 1 is acceptable too if status itself errors on a
    # half-empty persona dir (no memories.db yet, no persona_config.json).
    # _status_handler returns 1 and prints limited output when persona_exists
    # is False, but the persona: line is always printed before the dir check.
    assert r.returncode in (0, 1), (r.stdout, r.stderr)
    # The resolved name MUST surface in stdout — _status_handler prints
    # "persona: {args.persona}" unconditionally before the existence check.
    assert "persona: mira" in r.stdout, (r.stdout, r.stderr)


def test_status_many_personas_lists_them(tmp_path: Path) -> None:
    for n in ["alex", "mira", "nell"]:
        (tmp_path / "personas" / n).mkdir(parents=True)
    r = _nell(["status"], tmp_path)
    assert r.returncode == 2, (r.stdout, r.stderr)
    for n in ["alex", "mira", "nell"]:
        assert n in r.stderr, (r.stdout, r.stderr)
