"""Every leak-guard denied path must also be gitignored (#163).

A denied path that is not gitignored is a trap: it shows as untracked,
can be staged, and is only rejected by the pre-push guard.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DENIED = _REPO / "hooks" / "denied-paths.txt"


def _denied_paths() -> list[str]:
    lines = _DENIED.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


@pytest.mark.parametrize("path", _denied_paths())
def test_denied_path_is_gitignored(path: str) -> None:
    probe = f"{path.rstrip('/')}/probe.txt"
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe], cwd=_REPO, capture_output=True, text=True
    )
    assert result.returncode == 0, f"{path} is in denied-paths.txt but not .gitignore"
