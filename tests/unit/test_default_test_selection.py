"""A plain `pytest` must not spend subscription usage on a machine that has `claude` (#680)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MODULE = "tests/integration/brain/attunement/test_adversarial_corpus.py"
_REAL_CALL_TEST = "test_adversarial_corpus_does_not_crystallise_any_pattern"
_OFFLINE_TEST = "test_validate_grounded_gate_is_wired_into_merge_path"


def _collected(*args: str) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:randomly", *args, _MODULE],
        cwd=_REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return done.stdout


@pytest.mark.no_base_failure(reason="the fix is in pyproject.toml, which the base check keeps at the PR's version")
def test_real_call_tests_only_run_when_asked_for():
    plain = _collected()
    assert _OFFLINE_TEST in plain
    assert _REAL_CALL_TEST not in plain
    assert _REAL_CALL_TEST in _collected("-m", "requires_claude_cli")
