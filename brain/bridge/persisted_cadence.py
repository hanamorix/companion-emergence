"""Generic persisted wall-clock cadence for supervisor ticks.

The supervisor's monotonic timers (``time.monotonic()``) reset on every restart
and do not advance during system sleep, so any cadence whose interval exceeds a
typical session under-fires on a desktop app. This helper persists each
cadence's NEXT-due time as a wall-clock timestamp, so a due-in-the-past state
fires on a fresh process regardless of how long the interval is.

Simpler than ``brain/soul/cadence.py``: pure next-due pacing, no failure-backoff
or backlog-drain semantics (the cadences using this — voice reflection,
forgetting+narrative maintenance, finalize — have no per-item backlog; finalize
owns its own F-011 extraction backoff). See
``docs/superpowers/specs/2026-06-23-persisted-cadence-design.md``.
"""
from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CadenceState:
    next_at: datetime | None  # None => due now (never run / corrupt / reset)


def _state_path(persona_dir: Path, filename: str) -> Path:
    return persona_dir / filename


def _parse_ts(raw: object) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def load_cadence(persona_dir: Path, filename: str) -> CadenceState:
    """Load a persisted cadence. Missing/corrupt → due-now (fail toward running:
    a bad state file must never wedge a cadence into silence)."""
    try:
        data = json.loads(_state_path(persona_dir, filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CadenceState(next_at=None)
    if not isinstance(data, dict):
        return CadenceState(next_at=None)
    return CadenceState(next_at=_parse_ts(data.get("next_at")))


def save_cadence(persona_dir: Path, filename: str, state: CadenceState) -> None:
    """Atomically persist the cadence (temp file + rename).

    Best-effort + fail-soft: every call site is a supervisor ``finally``/
    end-of-block advance step, so a raised ``OSError`` (disk full, EACCES, a
    Windows AV/indexer holding the ``.tmp`` target during the atomic rename)
    would escape ``run_folded`` and kill the whole supervisor loop. A failed
    save only means the cadence re-fires sooner (load_cadence returns due-now
    on a missing/corrupt file), so swallowing the error is strictly safer than
    propagating it. Mirrors ``load_cadence``'s fail-toward-running posture.
    """
    path = _state_path(persona_dir, filename)
    payload = {"next_at": state.next_at.isoformat() if state.next_at is not None else None}
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.warning("save_cadence: could not persist %s (best-effort)", filename, exc_info=True)
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()


def is_due(state: CadenceState, *, now: datetime) -> bool:
    """True when the cadence should run now (never-run, or wall-clock due)."""
    return state.next_at is None or now >= state.next_at


def advance(*, now: datetime, interval_s: float) -> CadenceState:
    """Next-due = now + interval."""
    return CadenceState(next_at=now + timedelta(seconds=interval_s))
