"""`.github/workflows/test.yml`'s Python 3.14 job (#740).

`requires-python` has no upper bound, so a dev machine on 3.14 builds its `.venv` there while CI ran 3.12 (and 3.13
for the bridge tests only): a break that only shows on 3.14 passed CI twice (#678, #700). These pin what a YAML edit
could quietly drop: the job exists, runs on 3.14 with every extra, runs the whole suite, and leaves the CPU budget to
the main matrix's Ubuntu leg.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
WF = yaml.safe_load((REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
JOBS = WF["jobs"]

# The base check keeps .github/ at the PR head (HEAD_ONLY in scripts/new_tests_fail_on_base.py), so a test that reads
# only test.yml passes there by construction. All four fail against main's test.yml (checked by hand on 2026-11-19).
_HEAD_ONLY = pytest.mark.no_base_failure(
    reason="reads only .github/workflows/test.yml, which the base check keeps at the PR head; fails against main's (#740)"
)


def _job():
    assert "python-314" in JOBS, "test.yml has no python-314 job"
    return JOBS["python-314"]


@_HEAD_ONLY
def test_the_job_runs_on_ubuntu_for_pushes_and_prs():
    job = _job()
    assert job["name"] == "python 3.14 (full suite)"
    assert job["runs-on"] == "ubuntu-latest"
    assert job["if"] == "github.event_name != 'workflow_dispatch'"  # a manual run picks its own OS and tests
    assert job["timeout-minutes"] == JOBS["python"]["timeout-minutes"]


def _step(job, name):
    matches = [s for s in job["steps"] if s.get("name") == name]
    assert len(matches) == 1, f"expected one step named {name!r}, got {len(matches)}"
    return matches[0]


@_HEAD_ONLY
def test_the_job_installs_python_314_with_every_extra():
    job = _job()
    assert _step(job, "Set up Python 3.14")["run"] == "uv python install 3.14"
    assert _step(job, "Install dependencies")["run"] == "uv sync --all-extras --python 3.14"


@_HEAD_ONLY
def test_the_job_runs_the_whole_suite_with_the_main_matrix_options():
    """No path argument (a subset would miss breaks like #678's), and the same xdist options as the main matrix."""
    import shlex

    args = shlex.split(_step(_job(), "Run pytest")["run"])
    assert args[:5] == ["uv", "run", "--python", "3.14", "pytest"]
    pytest_args = args[5:]
    values = {"auto", "loadgroup", "no:tdd-guard", "short"}
    assert [a for a in pytest_args if not a.startswith("-") and a not in values] == [], "a path would run a subset"
    main = shlex.split(_step(JOBS["python"], "Run pytest")["run"])
    for opt in ("-n", "--dist", "-p"):
        assert pytest_args[pytest_args.index(opt) + 1] == main[main.index(opt) + 1], opt


@_HEAD_ONLY
def test_the_job_leaves_the_cpu_budget_to_the_main_matrix():
    """A slow runner must not fail the CPU budget twice: only the main matrix's Ubuntu leg switches it on."""
    job = _job()
    envs = [job.get("env") or {}] + [s.get("env") or {} for s in job["steps"]]
    assert not any("KINDLED_SLOW_BUDGET" in e for e in envs)
