"""scripts/ci_rerun_runner_loss.py + .github/workflows/rerun-runner-loss.yml — #758.

GitHub's macOS arm64 pool is short of runners; a job that no runner picks up is cancelled with the annotation "The job
was not acquired by Runner of type hosted even after multiple attempts", and the run reads as failed although no test
failed. The watcher re-runs the failed jobs only when every job that didn't pass is such a cancellation, never a real
failure. Every test here uses a fake `gh`; nothing calls GitHub.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("ci_rerun_runner_loss_under_test", REPO / "scripts" / "ci_rerun_runner_loss.py")
rr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rr
_spec.loader.exec_module(rr)

LOSS = "The job was not acquired by Runner of type hosted even after multiple attempts"


def _job(job_id, name, conclusion):
    return {"id": job_id, "name": name, "conclusion": conclusion}


class FakeGh:
    """A scripted run: `attempts[i]` is attempt i+1's `(jobs, annotations, conclusion)`. `run rerun` starts the next
    attempt, which reads as in progress for one status poll before it completes. Every call is kept."""

    def __init__(self, attempts):
        self.attempts = attempts
        self.attempt = 1
        self.polls_since_rerun = 0
        self.calls = []

    def __call__(self, args):
        import json

        self.calls.append(args)
        jobs, notes, conclusion = self.attempts[self.attempt - 1]
        if args[:2] == ["run", "rerun"]:
            self.attempt += 1
            self.polls_since_rerun = 0
            return ""
        path = args[1]
        if path.endswith("/annotations"):
            job_id = int(path.split("/")[-2])
            return json.dumps([{"message": m} for m in notes.get(job_id, [])])
        if "/jobs" in path:
            return json.dumps({"jobs": jobs})
        self.polls_since_rerun += 1
        done = self.attempt == 1 or self.polls_since_rerun > 1
        return json.dumps({"run_attempt": self.attempt, "status": "completed" if done else "in_progress",
                           "conclusion": conclusion if done else None})

    def reruns(self):
        return sum(1 for c in self.calls if c[:2] == ["run", "rerun"])


def test_a_lost_job_is_re_run_after_a_pause_and_the_watch_ends_when_the_attempt_passes():
    lost = ([_job(1, "python ubuntu-latest", "success"), _job(2, "tauri (cargo check + test)", "cancelled")], {2: [LOSS]}, "failure")
    green = ([_job(1, "python ubuntu-latest", "success"), _job(7, "tauri (cargo check + test)", "success")], {}, "success")
    gh = FakeGh([lost, green])
    slept = []
    assert rr.watch(123, "o/r", gh=gh, sleep=slept.append, pause_s=180, poll_s=30) == 1
    assert gh.reruns() == 1 and ["run", "rerun", "123", "--failed", "-R", "o/r"] in gh.calls
    assert slept[0] == 180  # the pause comes before the re-run, so the queue can clear
    assert 30 in slept  # and the new attempt was waited for until it completed


def test_a_loss_that_keeps_happening_is_re_run_three_times_then_left_for_a_person(capsys):
    lost = ([_job(2, "tauri (cargo check + test)", "cancelled")], {2: [LOSS]}, "failure")
    gh = FakeGh([lost] * 5)
    assert rr.watch(123, "o/r", gh=gh, sleep=lambda s: None) == rr.MAX_RETRIES == 3
    assert gh.reruns() == 3
    assert "leaving it for a person" in capsys.readouterr().out


def test_a_real_failure_is_never_re_run():
    failed = ([_job(2, "tauri (cargo check + test)", "cancelled"), _job(4, "python windows-latest", "failure")],
              {2: [LOSS], 4: ["Process completed with exit code 1."]}, "failure")
    gh = FakeGh([failed])
    assert rr.watch(123, "o/r", gh=gh, sleep=lambda s: None) == 0
    assert gh.reruns() == 0


def test_main_watches_the_run_the_workflow_names_with_the_real_gh_and_clock(monkeypatch):
    import time

    seen = {}
    monkeypatch.setenv("RUN_ID", "44320147167")
    monkeypatch.setenv("REPO", "o/r")
    monkeypatch.setattr(rr, "watch", lambda run_id, repo, *, gh, sleep: seen.update(run_id=run_id, repo=repo, gh=gh, sleep=sleep) or 0)
    assert rr.main() == 0
    assert seen["run_id"] == 44320147167 and seen["repo"] == "o/r"
    assert seen["gh"] is rr.run_gh and seen["sleep"] is time.sleep


def test_the_workflow_watches_failed_first_attempts_of_test_with_least_privilege_and_no_pr_code():
    import yaml

    wf = yaml.safe_load((REPO / ".github" / "workflows" / "rerun-runner-loss.yml").read_text(encoding="utf-8"))
    test_wf = yaml.safe_load((REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
    assert wf[True]["workflow_run"] == {"workflows": [test_wf["name"]], "types": ["completed"]}  # bare `on:` is True
    assert wf["permissions"] == {"actions": "write", "checks": "read", "contents": "read"}
    (job,) = wf["jobs"].values()
    # failures only, and attempt 1 only: one watcher owns all of a run's retries
    assert "github.event.workflow_run.conclusion == 'failure'" in job["if"]
    assert "github.event.workflow_run.run_attempt == 1" in job["if"]
    assert job["runs-on"] == "ubuntu-latest" and 0 < job["timeout-minutes"] <= 180
    assert job["concurrency"]["group"] == "rerun-runner-loss-${{ github.event.workflow_run.id }}"
    # workflow_run runs with the base repo's token: it must never check out or run a PR's code
    checkout = [s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")]
    assert checkout and all("ref" not in (s.get("with") or {}) for s in checkout)
    (run_step,) = [s for s in job["steps"] if "run" in s]
    assert run_step["run"].strip() == "python3 scripts/ci_rerun_runner_loss.py"
    assert run_step["env"] == {"RUN_ID": "${{ github.event.workflow_run.id }}", "REPO": "${{ github.repository }}",
                               "GH_TOKEN": "${{ github.token }}"}


def test_only_the_jobs_that_need_macos_wait_for_a_macos_runner():
    # The frontend build is OS-neutral (release.yml's validate gate runs it on Ubuntu); cargo compiles the macOS-only
    # #[cfg] halves and the wheel smoke stays CPU-torch-sized on macOS, so those two keep the macOS runner.
    import yaml

    jobs = yaml.safe_load((REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))["jobs"]
    assert jobs["frontend"]["runs-on"] == "ubuntu-latest"
    assert jobs["tauri"]["runs-on"] == "macos-latest"
    assert jobs["wheel-smoke"]["runs-on"] == "macos-latest"


def test_only_runner_losses_are_retried():
    passed = _job(1, "python ubuntu-latest", "success")
    lost = _job(2, "tauri (cargo check + test)", "cancelled")
    notes = {2: [LOSS]}
    assert rr.runner_loss_only([passed, lost, _job(3, "frontend", "skipped")], notes) is True
    # a real failure beside a loss: never retried
    assert rr.runner_loss_only([lost, _job(4, "python windows-latest", "failure")], {**notes, 4: ["Process completed with exit code 1."]}) is False
    # cancelled for another reason (a person, a timeout, a newer push): not a runner loss
    assert rr.runner_loss_only([_job(5, "python macos-latest", "cancelled")], {5: ["The operation was canceled."]}) is False
    assert rr.runner_loss_only([_job(6, "wheel smoke", "timed_out")], {6: [LOSS]}) is False
    # nothing failed: nothing to retry
    assert rr.runner_loss_only([passed], {}) is False
