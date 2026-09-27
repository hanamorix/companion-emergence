"""ram-spike-fix INC-11 — the central cadence function publishes the running
gated job to `background_jobs` (spec §6, S15/S23/S38; criterion C11(a)).

Kept in its own file (not `test_central_cadence.py`) so this increment's
build doesn't touch a file another concurrent builder in this worktree may
have in-flight changes to.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.bridge import background_jobs
from brain.bridge.central_cadence import GatedJob, JobOutcome, run_central_pass

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def setup_function() -> None:
    background_jobs._reset_for_tests()


def teardown_function() -> None:
    background_jobs._reset_for_tests()


def _pass(persona_dir: Path, jobs, *, idle=True, slot=True, now=NOW):
    return run_central_pass(
        persona_dir,
        jobs,
        is_idle=(lambda: idle),
        slot_available=(lambda: slot),
        now_wall=(lambda: now),
    )


def test_job_is_published_while_run_executes_and_cleared_after(tmp_path: Path) -> None:
    seen_during_run: list[str] = []

    def run() -> None:
        seen_during_run.append(",".join(background_jobs.snapshot()))

    job = GatedJob("maintenance", run=run, has_work=lambda: True)
    assert background_jobs.snapshot() == []

    _pass(tmp_path, [job])

    assert seen_during_run == ["maintenance"], "job must be published for the run() call itself"
    assert background_jobs.snapshot() == [], "must clear once run() returns"


def test_job_is_cleared_even_when_run_raises(tmp_path: Path) -> None:
    """S44: a raising job still resolves (COMPLETED, cadence advances) — the
    registry must not leak an entry when that happens."""

    def run() -> None:
        raise RuntimeError("boom")

    job = GatedJob("compaction", run=run, has_work=lambda: True)

    decisions = _pass(tmp_path, [job])

    assert ("compaction", "completed") in [(d.job, d.action) for d in decisions]
    assert background_jobs.snapshot() == []


def test_job_is_cleared_when_run_returns_paused(tmp_path: Path) -> None:
    """S38: a paused job is not running. `run()` returning PAUSED still
    exits the wrapping `with` normally, so the registry must already be
    clear by the time the central function's PAUSED handling runs."""
    seen_during_run: list[str] = []

    def run() -> JobOutcome:
        seen_during_run.append(",".join(background_jobs.snapshot()))
        return JobOutcome.PAUSED

    job = GatedJob("pass2", run=run, has_work=lambda: True)

    decisions = _pass(tmp_path, [job])

    assert seen_during_run == ["pass2"]
    assert ("pass2", "paused") in [(d.job, d.action) for d in decisions]
    assert background_jobs.snapshot() == []


def test_only_the_currently_running_job_is_published_not_the_whole_due_set(tmp_path: Path) -> None:
    """Two due jobs run one after another (S55 order via GATED_JOB_ORDER):
    while the first runs, only its name is published; by the time the
    second runs, the first has already cleared."""
    seen: list[list[str]] = []

    def run_a() -> None:
        seen.append(background_jobs.snapshot())

    def run_b() -> None:
        seen.append(background_jobs.snapshot())

    job_a = GatedJob("session_snapshot_prune", run=run_a, has_work=lambda: True)
    job_b = GatedJob("emotion_backfill", run=run_b, has_work=lambda: True)

    _pass(tmp_path, [job_b, job_a])  # order_jobs sorts into GATED_JOB_ORDER

    assert seen == [["session_snapshot_prune"], ["emotion_backfill"]]
    assert background_jobs.snapshot() == []


def test_a_skipped_or_not_due_job_never_appears(tmp_path: Path) -> None:
    ran = False

    def run() -> None:
        nonlocal ran
        ran = True

    not_due = GatedJob("clustering", run=run, has_work=lambda: False)
    _pass(tmp_path, [not_due])
    assert ran is False
    assert background_jobs.snapshot() == []

    denied_slot = GatedJob("clustering", run=run, has_work=lambda: True)
    _pass(tmp_path, [denied_slot], slot=False)
    assert ran is False
    assert background_jobs.snapshot() == []
