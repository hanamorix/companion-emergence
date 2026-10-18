"""Kindled-link supervisor tick cadence (Phase 7a T5).

Persisted wall-clock cadence in ``kindled_link/tick_cadence.json``.
Mirrors brain/kindled_link/relationship.py cadence idiom.

Functions accept an explicit ``now`` — no internal clock.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from brain.health.json_io import read_json, write_json

_CADENCE_FILE = "tick_cadence.json"
_DEFAULT_INTERVAL_MINUTES = 5.0


def load_tick_cadence(persona_dir) -> dict:
    """Load the persisted tick cadence state. Returns ``{"last_run": iso|None}``."""
    p = Path(persona_dir) / "kindled_link" / _CADENCE_FILE
    return read_json(p, default=lambda: {"last_run": None}, expect=dict)  # fail-soft: never-run


def tick_is_due(
    persona_dir,
    now: datetime,
    *,
    interval_minutes: float = _DEFAULT_INTERVAL_MINUTES,
) -> bool:
    """True if a tick is due (no prior run, or elapsed ≥ interval)."""
    last = load_tick_cadence(persona_dir).get("last_run")
    if not last:
        return True
    try:
        elapsed_m = (now - datetime.fromisoformat(last)).total_seconds() / 60.0
    except (ValueError, TypeError):
        return True
    return elapsed_m >= interval_minutes


def save_tick_cadence(persona_dir, now: datetime) -> None:
    """Persist the tick cadence state (last_run timestamp)."""
    write_json(
        Path(persona_dir) / "kindled_link" / _CADENCE_FILE,
        {"last_run": now.isoformat()},
        indent=None,
        fsync=False,
    )
