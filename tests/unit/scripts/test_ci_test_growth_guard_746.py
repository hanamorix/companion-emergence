"""`.github/workflows/test.yml` runs the test-growth guard on pull requests, report-only for now (#746).

These pin what a YAML or pyproject edit could quietly drop: the job exists and cannot block yet, it calls the guard
without `--enforce`, and the `large_grid` marker the guard reads is registered.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
WF = yaml.safe_load((REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
JOBS = WF["jobs"]

# The base check keeps .github/ and pyproject.toml at the PR head, so a test that reads only those passes there by
# construction; each of these fails against main's files (checked by hand, #746).
_HEAD_ONLY = pytest.mark.no_base_failure(
    reason="reads only test.yml and pyproject.toml, which the base check keeps at the PR head; fails against main's (#746)"
)


def _run_commands(job):
    return [s["run"] for s in job["steps"] if "run" in s]


@_HEAD_ONLY
def test_the_guard_job_exists_and_cannot_block_during_the_trial():
    job = JOBS["test-growth-guard"]
    assert job["name"] == "test growth guard"
    assert job["if"] == "github.event_name == 'pull_request'"
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 15
    assert job["continue-on-error"] is True


@_HEAD_ONLY
def test_the_guard_job_runs_the_script_without_enforce():
    job = JOBS["test-growth-guard"]
    runs = "\n".join(_run_commands(job))
    assert "scripts/guard_test_growth.py" in runs
    assert "--enforce" not in runs
    assert '--base "$base"' in runs and "--head HEAD" in runs
    assert job["steps"][0]["with"]["fetch-depth"] == 0


@_HEAD_ONLY
def test_a_branch_without_the_script_gets_a_notice_and_a_crash_cannot_turn_the_check_red():
    (step,) = [s for s in JOBS["test-growth-guard"]["steps"] if "guard_test_growth.py" in s.get("run", "")]
    first = step["run"].lstrip().splitlines()[0]
    assert first == (
        'if [ ! -f scripts/guard_test_growth.py ]; then echo "::notice::this branch predates the test-growth guard;'
        ' merge main to get its report"; exit 0; fi'
    )
    assert step["continue-on-error"] is True


@_HEAD_ONLY
def test_the_large_grid_marker_is_registered_with_a_mandatory_reason():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    entries = [m for m in markers if m.startswith("large_grid(reason)")]
    assert len(entries) == 1
    assert "reason is mandatory" in entries[0]
