"""Daily Haiku-call budget tracker for the attunement subsystem.

Per autonomous-behaviour recipe item 2 + 3: hard cap, cap-hit → defer
without failing the reply path.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from brain.attunement.schemas import DAILY_BUDGET_DEFAULT
from brain.health.json_io import JsonFileError, Recovery, read_json, write_json

log = logging.getLogger(__name__)

_BUDGET_FILE = "daily_budget.json"


class _BudgetCorruptError(Exception):
    """Raised by _load when the budget file exists but cannot be parsed."""


def _budget_path(persona_dir: Path) -> Path:
    return persona_dir / "attunement" / _BUDGET_FILE


def _today_local_str(now: datetime) -> str:
    """Local-date string (user-local midnight is the reset boundary)."""
    return now.astimezone().strftime("%Y-%m-%d")


def _load(persona_dir: Path) -> dict:
    """Return the parsed budget state dict.

    MISSING file → {} (fresh-allow — no record yet).
    CORRUPT file → try the atomic .tmp sibling; if that also fails, raise
    _BudgetCorruptError so callers fail CLOSED rather than silently resetting.
    """
    path = _budget_path(persona_dir)
    try:
        # RESTORE: a corrupt file is restored from the previous save (or a legacy .tmp sibling) instead of
        # failing closed forever; with no good copy it raises, and the caller still fails CLOSED. A missing
        # file with no backup is "no record yet".
        return read_json(
            path,
            default=dict,
            recover=Recovery.RESTORE,
            expect=dict,
            fallbacks=(path.with_suffix(".tmp"),),
        )
    except JsonFileError as exc:
        if exc.kind == "io":
            raise  # an unreadable file propagated before; it still does
        log.warning("attunement budget: corrupt file — failing closed (deny) for today")
        raise _BudgetCorruptError() from exc


def _save(persona_dir: Path, state: dict) -> None:
    write_json(_budget_path(persona_dir), state, backups=1, indent=None, fsync=False)


def get_remaining(
    persona_dir: Path, *, now: datetime, cap: int = DAILY_BUDGET_DEFAULT
) -> int:
    try:
        state = _load(persona_dir)
    except _BudgetCorruptError:
        return 0
    today = _today_local_str(now)
    if state.get("date") != today:
        return cap
    return max(0, cap - int(state.get("count", 0)))


def consume_call(
    persona_dir: Path, *, now: datetime, cap: int = DAILY_BUDGET_DEFAULT
) -> bool:
    """Return True if call permitted (and decrements counter); False if cap reached."""
    # PASS2-AT-LEAST-ONCE: a pass-2 crash/restart repeat (see
    # pass2_queue.py's drain_all_locked pop-after-process step) can call
    # this twice for one logical item, consuming one extra daily-budget
    # count. Owner-accepted 2026-09-26: "the odds of it happening are tiny,
    # the consequence if it happens are small." See #240 (emotion-system
    # untangle) for the durable fix.
    try:
        state = _load(persona_dir)
    except _BudgetCorruptError:
        return False
    today = _today_local_str(now)
    if state.get("date") != today:
        state = {"date": today, "count": 0}
    if int(state.get("count", 0)) >= cap:
        return False
    state["count"] = int(state.get("count", 0)) + 1
    _save(persona_dir, state)
    return True
