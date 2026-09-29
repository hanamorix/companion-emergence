"""The central cadence function (ram-spike-fix INC-9, spec §3, S40/S43/S55).

One function, called once per supervisor loop pass, decides for every gated
background job whether it runs now. A gated job runs only when BOTH:

* the chat is idle (``cli_throttle.is_chat_idle()`` — asked again before every
  job, S43; the first "not idle" stops the whole sequence and the jobs not
  reached wait for the next lull), and
* the job is due: an interval job's persisted ``next_at`` has passed
  (S23/S34), or a no-interval job has work (pass 2, session snapshot/prune,
  emotion backfill, embedding backfill: S53/S66), or a predicate job's own
  condition holds (deploy recalibration: the stored floor is stale, S70/S73;
  self-model articulation: its own cadence module, S29).

Jobs run one after another in the fixed S55 order (``GATED_JOB_ORDER``), with
deploy recalibration immediately before daily calibration (S70/S73).

Cadence rules (interval jobs, owned here):

* A predicate job's own condition holds (the two floor-bootstrap RETRY jobs,
  name-recall fix S85 revised: no calibrated row, no cached bootstrap, chat
  has happened since the failed attempt, the process-start computation not
  running), like deploy recalibration.
* A missing OR corrupt cadence file is created as "last ran now"
  (``next_at = now + interval``) and the job does not run on that pass
  (S22/S69). A present file with a past ``next_at`` is simply overdue (S34),
  so an overdue job runs at the first lull, including the bridge-start lull.
* The cadence advances only when the job completes (S20/S36): not when it is
  skipped for lack of a lull, not when the concurrency slot is denied, not
  when it pauses, and not when the process dies mid-job (nothing is written
  until the job returns). A job that raises still advances (S44, the
  existing anti-hammering cadence invariant) — whether its own handler caught
  the exception or this function's outer guard did.

There is no startup catch-up: the bridge starting is just the first pass of
this function (S23/S34).

The heartbeat is NOT a gated job (S16/S21): it keeps its own 15-minute timer
in the supervisor. The supervisor passes its heartbeat check as
``between_jobs``; this function calls it after every job that ran, so a
heartbeat that fell due while a job was running runs only once that job has
returned, and the next job starts only after the whole heartbeat pass
(single supervisor thread; S41/S65). INC-10's per-item pause protocol calls
the same hook between a job's items.

Slot: before running a due job this function peeks
``cli_throttle.slot_available()`` (read-only, never takes the semaphore). A
denied peek skips the job without advancing its cadence (S20). Jobs that make
background CLI calls keep their own authoritative acquire inside ``run``; a
lost race there returns ``JobOutcome.SKIPPED`` (no advance).
"""
from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge import background_jobs, cli_throttle, persisted_cadence

logger = logging.getLogger(__name__)

# S55 order, with deploy recalibration (S70/S73) immediately before daily
# calibration and, after embedding backfill, the once-per-process cosine floor
# bootstrap (name-recall fix S85; it needs the embedder, off the recall hot
# path). Names are the job table's keys and the log's `job=` values.
GATED_JOB_ORDER: tuple[str, ...] = (
    "pass2",
    "session_snapshot_prune",
    "emotion_backfill",
    "embedding_backfill",
    "cosine_floor_bootstrap",
    "rerank_floor_bootstrap",
    "maintenance",
    "interest_sweep",
    "self_model_articulation",
    "compaction",
    "clustering",
    "deploy_recalibration",
    "daily_calibration",
    "weekly_selftune",
    "finalize",
    "initiate_review",
)


class JobOutcome(enum.Enum):
    """What a gated job's ``run`` reports back."""

    COMPLETED = "completed"  # ran (or raised): an interval job's cadence advances
    SKIPPED = "skipped"  # did no work (e.g. lost the slot race): no advance
    PAUSED = "paused"  # stopped between items for chat (INC-10): no advance, stop the sequence


@dataclass(frozen=True)
class GatedJob:
    """One row of the job table.

    Exactly one "due" source applies:

    * ``cadence_file`` + ``interval_s`` → an interval job; this module owns
      its persisted cadence (missing/corrupt init, advance on completion).
    * ``has_work`` → a no-interval or predicate job (no cadence file owned
      here); due iff ``has_work()`` is True. A job with its own internal
      cadence (self-model articulation, S29) supplies it through
      ``has_work`` and saves its own cadence inside ``run``.

    ``run`` returns a ``JobOutcome`` (``None`` counts as ``COMPLETED``).
    """

    name: str
    run: Callable[[], JobOutcome | None]
    cadence_file: str | None = None
    interval_s: float | None = None
    has_work: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        interval_job = self.cadence_file is not None
        if interval_job and self.interval_s is None:
            raise ValueError(f"gated job {self.name!r}: cadence_file needs interval_s")
        if interval_job == (self.has_work is not None):
            raise ValueError(
                f"gated job {self.name!r}: give exactly one of cadence_file or has_work"
            )


@dataclass(frozen=True)
class JobDecision:
    """One logged decision (the event log C5/C6/C15 read)."""

    job: str
    action: str  # run | completed | skipped | paused | skip-not-due | skip-no-lull | skip-slot | init-cadence


def order_jobs(jobs: Sequence[GatedJob]) -> list[GatedJob]:
    """Return ``jobs`` sorted into ``GATED_JOB_ORDER``. Unknown names are an
    error (every gated job has a fixed slot, S55)."""
    rank = {name: i for i, name in enumerate(GATED_JOB_ORDER)}
    for job in jobs:
        if job.name not in rank:
            raise ValueError(f"unknown gated job {job.name!r} (not in GATED_JOB_ORDER)")
    return sorted(jobs, key=lambda j: rank[j.name])


def _load_interval_cadence(
    persona_dir: Path, job: GatedJob, now: datetime
) -> tuple[persisted_cadence.CadenceState, bool]:
    """(state, initialised) for an interval job. A missing/corrupt file is
    written as ``now + interval`` (S22/S69)."""
    assert job.cadence_file is not None and job.interval_s is not None
    return persisted_cadence.load_or_init_cadence(
        persona_dir, job.cadence_file, now=now, interval_s=job.interval_s
    )


def run_central_pass(
    persona_dir: Path,
    jobs: Sequence[GatedJob],
    *,
    is_idle: Callable[[], bool] = cli_throttle.is_chat_idle,
    slot_available: Callable[[], bool] = cli_throttle.slot_available,
    now_wall: Callable[[], datetime] = lambda: datetime.now(UTC),
    between_jobs: Callable[[], None] | None = None,
) -> list[JobDecision]:
    """Run one pass of the central cadence function; return its decisions.

    ``jobs`` is the job table (any order; sorted into ``GATED_JOB_ORDER``
    here). ``is_idle``/``slot_available``/``now_wall`` are injectable for
    deterministic tests; production uses the cli_throttle functions and the
    wall clock. ``between_jobs`` is the heartbeat hook (see module docstring).
    """
    persona_dir = Path(persona_dir)
    decisions: list[JobDecision] = []

    def _decide(job: str, action: str) -> None:
        decisions.append(JobDecision(job, action))
        # skip-not-due fires for most jobs on most passes (16 lines a minute
        # while idle); keep it out of the INFO log. Every other decision is INFO.
        level = logging.DEBUG if action == "skip-not-due" else logging.INFO
        logger.log(level, "central cadence: job=%s action=%s", job, action)

    ordered = order_jobs(jobs)

    # S22/S69: load every interval cadence once per pass, creating any
    # missing/corrupt file as "last ran now", chat idle or not, so the
    # one-full-interval clock starts when the file is first found missing
    # (not at the first lull). The loaded state is reused below: this
    # function is the only writer of these files (single supervisor thread).
    initialised: set[str] = set()
    states: dict[str, persisted_cadence.CadenceState] = {}
    for job in ordered:
        if job.cadence_file is None:
            continue
        try:
            state, created = _load_interval_cadence(persona_dir, job, now_wall())
        except Exception:  # noqa: BLE001 — a cadence read fault must not kill the pass
            logger.exception("central cadence: job=%s cadence load raised", job.name)
            continue
        states[job.name] = state
        if created:
            initialised.add(job.name)
            _decide(job.name, "init-cadence")

    for job in ordered:
        # S43: ask is-chat-idle again before EVERY job; the first "not idle"
        # ends the sequence — the rest wait for the next lull.
        if not is_idle():
            _decide(job.name, "skip-no-lull")
            break

        if job.name in initialised:
            _decide(job.name, "skip-not-due")
            continue
        try:
            if job.cadence_file is not None:
                state = states.get(job.name)
                due = state is not None and persisted_cadence.is_due(state, now=now_wall())
            else:
                assert job.has_work is not None
                due = bool(job.has_work())
        except Exception:  # noqa: BLE001 — a failing due probe skips the job, never the pass
            logger.exception("central cadence: job=%s due check raised", job.name)
            due = False
        if not due:
            _decide(job.name, "skip-not-due")
            continue

        if not slot_available():
            # S20: a denied slot is a skip, not a run — cadence untouched.
            _decide(job.name, "skip-slot")
            continue

        _decide(job.name, "run")
        try:
            # INC-11 (S15/S23/S38): publish this job as running for the
            # duration of `run()` only — a paused job has already returned
            # (JobOutcome.PAUSED below) and so is no longer a member.
            with background_jobs.running(job.name):
                outcome = job.run()
        except Exception:  # noqa: BLE001 — S44: a job that raises still advances
            logger.exception("central cadence: job=%s raised", job.name)
            outcome = JobOutcome.COMPLETED
        if outcome is None:
            outcome = JobOutcome.COMPLETED

        if outcome is JobOutcome.COMPLETED and job.cadence_file is not None:
            assert job.interval_s is not None
            persisted_cadence.save_cadence(
                persona_dir,
                job.cadence_file,
                persisted_cadence.advance(now=now_wall(), interval_s=job.interval_s),
            )
        _decide(job.name, outcome.value)

        # S41/S65: the heartbeat runs between jobs, never during one.
        if between_jobs is not None:
            try:
                between_jobs()
            except Exception:  # noqa: BLE001 — the hook must never kill the pass
                logger.exception("central cadence: between-jobs hook raised")

        if outcome is JobOutcome.PAUSED:
            # INC-10 seam: a job that paused for chat ends the sequence (S43).
            break

    return decisions
