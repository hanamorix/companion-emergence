"""scripts/stack_guard.sh + .github/workflows/stack-guard.yml — #282.

Two jobs, both ADVISORY (required checks apply to a PR's base branch, and a stacked
PR's base is an unprotected feature branch, so no check on main can block the merge
that goes wrong):
- **check** (every open PR): red when the PR's live base isn't main, naming the PR
  below and what to do.
- **merged** (a PR merged into anything but main): comments on it — its changes are
  not on main, and if the base's own PR already merged or closed they are stranded
  (the F2 incident, recovered by #281).
The real guard is delete-branch-on-merge (on since 2026-09-25; off during F2): GitHub
deletes a merged PR's branch and retargets the next PR in the stack to main.
See CONTRIBUTING.md, "Stacked pull requests".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "stack_guard.sh"
WF_PATH = REPO / ".github" / "workflows" / "stack-guard.yml"

bash_only = pytest.mark.skipif(sys.platform == "win32", reason="bash script run by a GitHub ubuntu runner")
needs_jq = pytest.mark.skipif(shutil.which("jq") is None, reason="the fake gh applies the real --jq with jq")


def _fake_gh(tmp_path: Path, prs, live_base, fail_list, fail_api, fail_comment, comments=()) -> Path:
    """A `gh` on PATH. `pr list --jq EXPR` answers over a fixture of PRs with the real
    jq (so the script's own expression is tested); `api .../pulls/N` answers the live
    base; `pr comment` records the body. Every call is logged."""
    (tmp_path / "prs.json").write_text(json.dumps(prs or []), encoding="utf-8")
    (tmp_path / "view.json").write_text(json.dumps({"comments": [{"body": b} for b in comments]}), encoding="utf-8")
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir(exist_ok=True)
    gh = fakebin / "gh"
    gh.write_text(
        "#!/bin/bash\n"
        f"echo \"$@\" >> '{tmp_path}/gh.log'\n"
        'case "$1 $2" in\n'
        "  'pr list')\n"
        + ('    echo "gh: HTTP 502" >&2; exit 1;;\n' if fail_list else
           '    expr=""; while [ $# -gt 0 ]; do case "$1" in --jq) expr="$2"; shift 2;; *) shift;; esac; done\n'
           f"    jq -r \"$expr\" '{tmp_path}/prs.json';;\n")
        + "  'pr view')\n"
        + '    expr=""; while [ $# -gt 0 ]; do case "$1" in --jq) expr="$2"; shift 2;; *) shift;; esac; done\n'
        + f"    jq -r \"$expr\" '{tmp_path}/view.json';;\n"
        + "  'pr comment')\n"
        + ('    echo "gh: HTTP 403" >&2; exit 1;;\n' if fail_comment else
           f"    cat > '{tmp_path}/comment.md';;\n")
        + "  api*)\n"
        + ('    echo "gh: HTTP 502" >&2; exit 1;;\n' if fail_api else f"    echo '{live_base}';;\n")
        + "esac\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    return fakebin


MERGED_AT = "2026-09-24T13:38:44Z"  # when "this" PR merged, in merged-mode tests


def _run(tmp_path: Path, base: str, prs=None, *, mode: str = "check", live_base: str | None = None,
         fail_list=False, fail_api=False, fail_comment=False, comments=()):
    fakebin = _fake_gh(tmp_path, prs, live_base if live_base is not None else base,
                       fail_list, fail_api, fail_comment, comments)
    env = {**os.environ, "PATH": f"{fakebin}{os.pathsep}{os.environ['PATH']}", "MODE": mode,
           "BASE_REF": base, "DEFAULT_BRANCH": "main", "REPO": "o/r", "PR_NUMBER": "7",
           "MERGED_AT": MERGED_AT}
    cp = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, encoding="utf-8")
    log = tmp_path / "gh.log"
    comment = tmp_path / "comment.md"
    return (cp, log.read_text(encoding="utf-8") if log.exists() else "",
            comment.read_text(encoding="utf-8") if comment.exists() else None)


# ── check: every open PR ─────────────────────────────────────────────────────

@bash_only
def test_a_pr_into_main_passes(tmp_path):
    cp, gh_calls, _ = _run(tmp_path, "main")
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "pr list" not in gh_calls


@bash_only
@needs_jq
def test_stacked_on_an_open_pr_fails_and_says_merge_it_first(tmp_path):
    cp, gh_calls, _ = _run(tmp_path, "hana/slice-2", [{"number": 305, "state": "OPEN"}])
    assert cp.returncode == 1
    assert "::error" in cp.stdout and "#305" in cp.stdout and "Merge #305" in cp.stdout
    assert "--head hana/slice-2" in gh_calls and "--repo o/r" in gh_calls


@bash_only
@needs_jq
def test_the_f2_case_below_already_merged_still_fails(tmp_path):
    """#274 in F2: the PR below (#271) had reached main 36 s earlier and its branch
    wasn't deleted; the sketch's "PR below merged" rule would have been green."""
    cp, _, _ = _run(tmp_path, "tot/embedding-column-259", [{"number": 271, "state": "MERGED"}])
    assert cp.returncode == 1
    assert "#271" in cp.stdout and "already merged" in cp.stdout and "Retarget" in cp.stdout


@bash_only
@needs_jq
def test_below_closed_without_merging_fails_with_retarget(tmp_path):
    cp, _, _ = _run(tmp_path, "abandoned", [{"number": 9, "state": "CLOSED"}])
    assert cp.returncode == 1
    assert "#9" in cp.stdout and "closed without merging" in cp.stdout


@bash_only
@needs_jq
def test_prefers_the_open_pr_when_a_branch_name_has_several(tmp_path):
    cp, _, _ = _run(tmp_path, "reused", [{"number": 40, "state": "MERGED"}, {"number": 41, "state": "OPEN"}])
    assert cp.returncode == 1
    assert "Merge #41" in cp.stdout


@bash_only
@needs_jq
def test_a_base_branch_with_no_pr_still_fails(tmp_path):
    cp, _, _ = _run(tmp_path, "some-feature", [])
    assert cp.returncode == 1
    assert "some-feature" in cp.stdout and "Retarget" in cp.stdout


@bash_only
def test_a_github_lookup_failure_still_fails_the_check(tmp_path):
    cp, _, _ = _run(tmp_path, "hana/slice-2", fail_list=True)
    assert cp.returncode == 1
    assert "::error" in cp.stdout and "Retarget" in cp.stdout


@bash_only
@needs_jq
def test_reads_the_live_base_so_a_rerun_after_a_retarget_goes_green(tmp_path):
    """A re-run replays the old event, whose payload still names the old base."""
    cp, gh_calls, _ = _run(tmp_path, "hana/slice-2", [{"number": 305, "state": "MERGED"}], live_base="main")
    assert cp.returncode == 0, cp.stdout
    assert "api repos/o/r/pulls/7" in gh_calls


@bash_only
@needs_jq
def test_the_payload_base_is_the_fallback_when_the_live_lookup_fails(tmp_path):
    cp, _, _ = _run(tmp_path, "hana/slice-2", [{"number": 305, "state": "OPEN"}], fail_api=True)
    assert cp.returncode == 1
    assert "Merge #305" in cp.stdout


@bash_only
@needs_jq
def test_a_percent_in_a_branch_name_cannot_break_the_annotation(tmp_path):
    """Workflow-command messages must escape % (then CR/LF) or the annotation garbles."""
    cp, _, _ = _run(tmp_path, "fix-100%", [])
    assert cp.returncode == 1
    assert "fix-100%25" in cp.stdout


@bash_only
@needs_jq
def test_works_without_repo_on_old_bash(tmp_path):
    """No REPO → no --repo args: an empty array under `set -u` is fatal on bash 3.2."""
    fakebin = _fake_gh(tmp_path, [{"number": 3, "state": "OPEN"}], "x", False, False, False)
    env = {**os.environ, "PATH": f"{fakebin}{os.pathsep}{os.environ['PATH']}",
           "BASE_REF": "x", "DEFAULT_BRANCH": "main"}
    env.pop("REPO", None)
    cp = subprocess.run(["/bin/bash", str(SCRIPT)], env=env, capture_output=True, text=True, encoding="utf-8")
    assert cp.returncode == 1 and "unbound variable" not in cp.stderr, cp.stderr
    assert "Merge #3" in cp.stdout


# ── merged: the post-merge alarm ─────────────────────────────────────────────

@bash_only
def test_merged_into_main_is_silent(tmp_path):
    cp, gh_calls, comment = _run(tmp_path, "main", mode="merged")
    assert cp.returncode == 0
    assert comment is None and gh_calls == ""


@bash_only
@needs_jq
def test_merged_into_an_already_merged_branch_says_stranded_and_how_to_recover(tmp_path):
    """The F2 alarm: #274 merged into #271's branch after #271 had reached main."""
    cp, _, comment = _run(tmp_path, "tot/embedding-column-259", [
        {"number": 271, "state": "MERGED", "mergedAt": "2026-09-24T13:38:23Z", "baseRefName": "main"}], mode="merged")
    assert cp.returncode == 1
    assert "::error" in cp.stdout
    assert comment is not None
    assert "not on main" in comment and "#271" in comment and "tot/embedding-column-259" in comment
    assert "pull request from" in comment  # the recovery: a PR from that branch to main


@bash_only
@needs_jq
def test_merged_into_a_branch_whose_pr_closed_unmerged_is_stranded(tmp_path):
    cp, _, comment = _run(tmp_path, "abandoned", [{"number": 9, "state": "CLOSED"}], mode="merged")
    assert cp.returncode == 1
    assert "not on main" in comment and "#9" in comment and "closed without merging" in comment


@bash_only
@needs_jq
def test_merged_into_a_branch_with_no_pr_is_stranded(tmp_path):
    cp, _, comment = _run(tmp_path, "some-feature", [], mode="merged")
    assert cp.returncode == 1
    assert "not on main" in comment and "some-feature" in comment


@bash_only
@needs_jq
def test_merged_into_an_open_prs_branch_warns_but_is_not_stranded_yet(tmp_path):
    cp, _, comment = _run(tmp_path, "hana/slice-2", [{"number": 305, "state": "OPEN"}], mode="merged")
    assert cp.returncode == 0
    assert "::warning" in cp.stdout
    assert "#305" in comment and "reach main only when #305 merges" in comment


@bash_only
@needs_jq
def test_the_alarm_still_fails_loudly_when_it_cannot_comment(tmp_path):
    cp, _, _ = _run(tmp_path, "tot/x", [
        {"number": 1, "state": "MERGED", "mergedAt": "2026-09-24T10:00:00Z", "baseRefName": "main"}],
        mode="merged", fail_comment=True)
    assert cp.returncode == 1
    assert "::error" in cp.stdout


@bash_only
@needs_jq
def test_a_base_pr_that_merged_later_carried_the_changes_to_main(tmp_path):
    """The alarm runs seconds after the merge. In a quick top-down collapse (C into B,
    then B into main) B has merged by then — but AFTER C, so C's changes went with it."""
    cp, _, comment = _run(tmp_path, "hana/b", [
        {"number": 12, "state": "MERGED", "mergedAt": "2026-09-24T13:38:59Z", "baseRefName": "main"}], mode="merged")
    assert cp.returncode == 0, cp.stdout
    assert comment is None


@bash_only
@needs_jq
def test_carried_on_to_another_stacked_branch_warns_to_follow_it(tmp_path):
    """F2's #277: merged into #274's branch, which then merged (later) into #271's branch."""
    cp, _, comment = _run(tmp_path, "tot/f2a", [
        {"number": 274, "state": "MERGED", "mergedAt": "2026-09-24T13:38:59Z",
         "baseRefName": "tot/embedding-column-259"}], mode="merged")
    assert cp.returncode == 0
    assert "::warning" in cp.stdout
    assert "#274" in comment and "tot/embedding-column-259" in comment


@bash_only
@needs_jq
def test_a_rerun_does_not_post_the_alarm_twice(tmp_path):
    prior = "<!-- stack-guard -->\n🚨 **stack guard: these changes are not on main.** ..."
    cp, gh_calls, comment = _run(tmp_path, "tot/x", [
        {"number": 1, "state": "MERGED", "mergedAt": "2026-09-24T10:00:00Z", "baseRefName": "main"}],
        mode="merged", comments=[prior])
    assert cp.returncode == 1 and "::error" in cp.stdout
    assert comment is None and "pr comment" not in gh_calls


@bash_only
@needs_jq
def test_the_alarm_comment_carries_the_marker(tmp_path):
    _, _, comment = _run(tmp_path, "abandoned", [{"number": 9, "state": "CLOSED"}], mode="merged")
    assert comment.startswith("<!-- stack-guard -->")


@bash_only
def test_an_unknown_mode_is_refused(tmp_path):
    cp, gh_calls, _ = _run(tmp_path, "hana/x", mode="merge")
    assert cp.returncode == 2 and gh_calls == ""


# ── the workflow ─────────────────────────────────────────────────────────────

WF = yaml.safe_load(WF_PATH.read_text(encoding="utf-8")) if WF_PATH.exists() else {}
ON = WF.get(True, {})  # YAML 1.1 parses bare `on:` as boolean True
CHECK = WF.get("jobs", {}).get("stack-guard", {})
ALARM = WF.get("jobs", {}).get("stranded-after-merge", {})


def _script_step(job):
    steps = [s for s in job["steps"] if "stack_guard.sh" in s.get("run", "")]
    assert len(steps) == 1
    return steps[0]


def _checkout(job):
    steps = [s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout@")]
    assert len(steps) == 1
    return steps[0]


def test_runs_on_every_pr_base_and_reruns_when_the_base_changes():
    pr = ON["pull_request"]
    assert "branches" not in pr, "a base filter would skip exactly the stacked PRs"
    # `edited` fires on a base change, incl. GitHub retargeting a stacked PR to main
    assert {"opened", "reopened", "synchronize", "edited", "closed"} <= set(pr["types"])
    assert set(ON) == {"pull_request"}, "no pull_request_target: a PR's own copy runs, no secrets"


def test_only_the_alarm_can_write_and_only_to_pull_requests():
    assert WF["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert "permissions" not in CHECK
    assert ALARM["permissions"] == {"contents": "read", "pull-requests": "write"}


def test_the_check_skips_closed_prs_and_the_alarm_runs_only_after_a_merge():
    assert "github.event.action != 'closed'" in CHECK["if"]
    cond = ALARM["if"]
    assert "github.event.action == 'closed'" in cond
    assert "github.event.pull_request.merged" in cond


def test_the_check_reads_the_live_base_and_never_interpolates_the_branch_name():
    step = _script_step(CHECK)
    assert "${{" not in step["run"], "interpolating a branch name into bash is script injection"
    env = step["env"]
    assert env["BASE_REF"] == "${{ github.event.pull_request.base.ref }}"
    assert env["PR_NUMBER"] == "${{ github.event.pull_request.number }}"
    assert env["DEFAULT_BRANCH"] == "${{ github.event.repository.default_branch }}"
    assert env.get("MODE", "check") == "check"


def test_the_alarm_runs_in_merged_mode_from_the_merge_it_reports_on():
    step = _script_step(ALARM)
    assert "${{" not in step["run"]
    assert step["env"]["MODE"] == "merged"
    assert step["env"]["BASE_REF"] == "${{ github.event.pull_request.base.ref }}"
    assert step["env"]["MERGED_AT"] == "${{ github.event.pull_request.merged_at }}"
    # the merge's own commit on the base branch: always exists (the head branch may be
    # auto-deleted already) and carries the script wherever the base came from
    assert _checkout(ALARM)["with"]["ref"] == "${{ github.event.pull_request.merge_commit_sha }}"


def test_both_jobs_check_out_exactly_the_script():
    for job in (CHECK, ALARM):
        assert _checkout(job)["with"]["sparse-checkout"] == "scripts/stack_guard.sh"


def test_overlapping_check_runs_on_one_pr_cancel_the_older():
    conc = CHECK["concurrency"]
    assert "github.event.pull_request.number" in conc["group"]
    assert conc["cancel-in-progress"] is True
