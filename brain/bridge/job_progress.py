"""Generic per-job pause/resume progress store (ram-spike-fix INC-10, spec §4).

A pausable gated job that has no cursor/state file of its own (today: only
the forgetting sub-step of the maintenance job — every other pausable job
reuses an existing cursor file named in 2-plan.md's S32 table, e.g.
``emotion_backfill_state.json``, the embedding-backfill cursor, the
``calibration_log`` table itself, or a session's own extraction cursor)
saves how far it got here: ``<persona_dir>/cadence/<job>_progress.json``,
written atomically (temp file + rename — same pattern as
``persisted_cadence.save_cadence``, C31: a concurrent reader sees either the
old complete JSON or the new complete JSON, never a torn write).

The SAME file is the crash-resume point (S36): a killed process leaves
whatever was last saved here, and the next lull resumes from it — there is
no separate crash-recovery code path.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

from brain.paths import cadence_state_path

logger = logging.getLogger(__name__)


def _progress_path(persona_dir: Path, job: str) -> Path:
    return cadence_state_path(persona_dir, f"{job}_progress.json")


def save_progress(persona_dir: Path, job: str, progress: dict) -> None:
    """Atomically persist ``progress`` for ``job`` (temp file + rename).

    Best-effort + fail-soft, mirroring ``persisted_cadence.save_cadence``: a
    failed save only means the job resumes further back (or from scratch)
    on the next lull, never a crash of the caller.
    """
    path = _progress_path(persona_dir, job)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(progress), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.warning(
            "job_progress.save_progress: could not persist %s progress (best-effort)",
            job,
            exc_info=True,
        )
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()


def load_progress(persona_dir: Path, job: str) -> dict:
    """Load ``job``'s saved progress. Missing/corrupt -> ``{}`` (fail toward
    redoing a little work from scratch, never toward silently skipping
    some)."""
    try:
        data = json.loads(_progress_path(persona_dir, job).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def clear_progress(persona_dir: Path, job: str) -> None:
    """Remove ``job``'s saved progress on completion, so a finished pass
    never leaves a stale resume point for the next one to misread."""
    path = _progress_path(persona_dir, job)
    with contextlib.suppress(OSError):
        path.unlink()
