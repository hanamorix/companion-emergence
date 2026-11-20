"""Re-run a `test` run's failed jobs when GitHub lost them for want of a runner, and only then (#758).

GitHub's macOS arm64 pool is short of runners. A job that no runner picks up in time is cancelled with the annotation
"The job was not acquired by Runner of type hosted even after multiple attempts", and the run reads as failed although no
test failed. Run by `.github/workflows/rerun-runner-loss.yml` when a `test` run completes with a failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping

RUNNER_LOSS = "was not acquired by Runner"  # the stable part of GitHub's cancellation annotation

MAX_RETRIES = 3  # automatic re-runs of one run, each after PAUSE_S
PAUSE_S = 180  # before each re-run, so the runner queue can clear
POLL_S = 60  # between checks on a re-run attempt

Gh = Callable[[list[str]], str]  # runs `gh <args>` and returns its stdout; raises on failure

_PASSING = frozenset({"success", "skipped", "neutral"})

# Jobs that only report on their shards, named "<gate> (i/N)" (#746). A gate fails when a shard was cancelled, because a
# skipped check counts as passing (#774 review), so it says nothing about why: its shards do. Re-running a lost shard
# re-runs its gate too.
SHARD_GATES = frozenset({"python windows-latest"})


def _is_shard_gate(job: Mapping, names: set[str]) -> bool:
    """A gate is only a gate in a run that has its shards; in a manual run the job of that name runs the tests itself."""
    name = job.get("name", "")
    return name in SHARD_GATES and any(n.startswith(f"{name} (") for n in names)


def runner_loss_only(jobs: Iterable[Mapping], annotations: Mapping[int, list[str]]) -> bool:
    """True when at least one job didn't pass and every such job was cancelled with the runner-loss annotation.

    `jobs` are the run's jobs as the API gives them (`id`, `name`, `conclusion`); `annotations` maps a job id to its
    annotation messages. A real failure, a timeout, or a cancellation for any other reason (a person, a newer push) is
    never a runner loss, so a run holding one is not retried. A shard gate (SHARD_GATES) is left out: its shards decide.
    """
    jobs = list(jobs)
    names = {job.get("name", "") for job in jobs}
    failed = [job for job in jobs if job.get("conclusion") not in _PASSING and not _is_shard_gate(job, names)]
    return bool(failed) and all(
        job.get("conclusion") == "cancelled" and any(RUNNER_LOSS in note for note in annotations.get(job.get("id"), ()))
        for job in failed
    )


def _latest_jobs(gh: Gh, run_id: int, repo: str) -> tuple[list[dict], dict[int, list[str]]]:
    """The jobs of the run's latest attempt, and the annotation messages of each one that didn't pass."""
    jobs = json.loads(gh(["api", f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"]))["jobs"]
    notes = {
        job["id"]: [a.get("message", "") for a in json.loads(gh(["api", f"repos/{repo}/check-runs/{job['id']}/annotations"]))]
        for job in jobs
        if job.get("conclusion") not in _PASSING
    }
    return jobs, notes


def _run(gh: Gh, run_id: int, repo: str) -> dict:
    return json.loads(gh(["api", f"repos/{repo}/actions/runs/{run_id}"]))


def watch(run_id: int, repo: str, *, gh: Gh, sleep: Callable[[float], object],
          max_retries: int = MAX_RETRIES, pause_s: float = PAUSE_S, poll_s: float = POLL_S) -> int:
    """Re-run the run's failed jobs while its latest attempt failed only by runner losses, at most `max_retries` times.

    Each re-run waits `pause_s` first, then the new attempt is followed until it completes. Returns the number of
    re-runs made. One watcher owns all of a run's retries (the workflow starts it for attempt 1 only).
    """
    reruns = 0
    while True:
        jobs, notes = _latest_jobs(gh, run_id, repo)
        if not runner_loss_only(jobs, notes):
            print(f"run {run_id}: nothing lost to a runner shortage; {reruns} re-run(s) made")
            return reruns
        if reruns == max_retries:
            print(f"run {run_id}: still lost to the runner shortage after {reruns} re-runs; leaving it for a person")
            return reruns
        before = _run(gh, run_id, repo).get("run_attempt", 1)
        sleep(pause_s)
        gh(["run", "rerun", str(run_id), "--failed", "-R", repo])
        reruns += 1
        print(f"run {run_id}: re-ran the jobs lost to the runner shortage ({reruns} of {max_retries})")
        while True:
            run = _run(gh, run_id, repo)
            if run.get("run_attempt", 1) > before and run.get("status") == "completed":
                break
            sleep(poll_s)
        if run.get("conclusion") == "success":
            print(f"run {run_id}: attempt {run['run_attempt']} passed")
            return reruns


def run_gh(args: list[str]) -> str:
    """`gh <args>`'s stdout; a failed call raises (the workflow step then fails and shows why)."""
    return subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8", check=True).stdout


def main() -> int:
    """The workflow's entry point: watch the run in `RUN_ID` of `REPO` (`GH_TOKEN` authenticates `gh`)."""
    watch(int(os.environ["RUN_ID"]), os.environ["REPO"], gh=run_gh, sleep=time.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
