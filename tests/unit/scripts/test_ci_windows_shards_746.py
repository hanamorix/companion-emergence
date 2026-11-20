"""`.github/workflows/test.yml` runs the Windows python suite as N shards on separate runners (#746).

These pin what a YAML edit could quietly drop: every shard 1..N runs with `--shard i/N` (so the shards together run
every test once), one `python windows-latest` check still says whether Windows passed, and the per-shard artifacts don't
collide.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
WF = yaml.safe_load((REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
JOBS = WF["jobs"]

# The base check keeps .github/ at the PR head (HEAD_ONLY in scripts/new_tests_fail_on_base.py), so a test that reads
# only test.yml passes there by construction; each of these fails against main's test.yml (checked by hand, #746).
_HEAD_ONLY = pytest.mark.no_base_failure(
    reason="reads only .github/workflows/test.yml, which the base check keeps at the PR head; fails against main's (#746)"
)


def _step(job, name):
    matches = [s for s in job["steps"] if s.get("name") == name]
    assert len(matches) == 1, f"expected one step named {name!r}, got {len(matches)}"
    return matches[0]


@_HEAD_ONLY
def test_every_shard_from_1_to_n_runs_the_whole_suite_command_with_its_shard():
    job = JOBS["python-windows"]
    shards = job["strategy"]["matrix"]["shard"]
    count = len(shards)
    assert count >= 2 and shards == list(range(1, count + 1))
    assert job["strategy"]["fail-fast"] is False
    assert job["runs-on"] == "windows-latest"
    assert job["if"] == "github.event_name != 'workflow_dispatch'"
    assert job["name"] == f"python windows-latest (${{{{ matrix.shard }}}}/{count})"
    # GitHub fills in ${{ }} before any shell sees the command.
    args = shlex.split(_step(job, "Run pytest")["run"].replace("${{ matrix.shard }}", "2"))
    main = shlex.split(_step(JOBS["python"], "Run pytest")["run"])
    assert args[:3] == ["uv", "run", "pytest"] and "--shard" not in " ".join(main)
    assert f"--shard=2/{count}" in args
    assert [a for a in args if not a.startswith("--shard") and a not in ("-n", "logical", "auto")] == [
        a for a in main if a not in ("-n", "logical", "auto")
    ], "a shard runs the same command as the other systems, plus its shard"


@_HEAD_ONLY
def test_the_main_matrix_leaves_windows_to_the_shards_except_on_a_manual_run():
    import json
    import re

    expr = JOBS["python"]["strategy"]["matrix"]["os"]
    lists = [x for x in re.findall(r"'(\[[^']*\])'", expr) if "{0}" not in x]  # not format()'s one-OS template
    *manual_all, push_and_pr = lists  # the last list: the OSes of a push or a PR
    assert json.loads(push_and_pr) == ["ubuntu-latest", "macos-latest"]
    # `os: all` on a manual run still runs all three, Windows unsharded.
    assert [json.loads(x) for x in manual_all] == [["ubuntu-latest", "macos-latest", "windows-latest"]]
    # A manual run still picks any one OS, Windows included, and runs it unsharded for a faithful reproduction.
    assert "inputs.os != 'all'" in expr and "format('[\"{0}\"]', inputs.os)" in expr
    assert "windows-latest" in WF[True]["workflow_dispatch"]["inputs"]["os"]["options"]  # bare `on:` is True


@_HEAD_ONLY
def test_one_python_windows_latest_check_passes_only_when_every_shard_passed():
    """One name for people and scripts to read. Skipped, not failed, when a shard was cancelled: the runner-loss re-run
    (rerun-runner-loss.yml) re-runs a run only when every job that didn't pass was lost, and a failing aggregate would
    stop it; a re-run of the cancelled shard re-runs this job too."""
    (job,) = [j for j in JOBS.values() if j.get("name") == "python windows-latest"]
    assert job["needs"] == "python-windows"
    assert job["if"] == (
        "always() && github.event_name != 'workflow_dispatch' && needs.python-windows.result != 'cancelled'"
    )
    assert job["runs-on"] == "ubuntu-latest"
    (step,) = job["steps"]
    assert step["env"] == {"RESULT": "${{ needs.python-windows.result }}"}
    script = step["run"]
    assert 'test "$RESULT" = success' in script


@_HEAD_ONLY
def test_each_shard_uploads_its_reports_under_its_own_name():
    """upload-artifact refuses a second artifact of the same name in one run, so a shared name would lose all but one
    shard's junit report and CPU log (the timings scripts/refresh_shard_durations.py reads)."""
    count = len(JOBS["python-windows"]["strategy"]["matrix"]["shard"])
    names = {}
    for job_id in ("python", "python-windows"):
        for step in JOBS[job_id]["steps"]:
            if str(step.get("uses", "")).startswith("actions/upload-artifact"):
                names.setdefault(job_id, []).append(step["with"]["name"])
    assert sorted(names["python-windows"]) == [
        f"pytest-junit-windows-latest-${{{{ matrix.shard }}}}-of-{count}",
        f"slow-budget-cpu-windows-latest-${{{{ matrix.shard }}}}-of-{count}",
    ]
    assert sorted(names["python"]) == ["pytest-junit-${{ matrix.os }}", "slow-budget-cpu-${{ matrix.os }}"]
