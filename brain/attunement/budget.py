"""Daily Haiku-call budget tracker for the attunement subsystem.

Per autonomous-behaviour recipe item 2 + 3: hard cap, cap-hit → defer
without failing the reply path.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from brain.attunement.schemas import DAILY_BUDGET_DEFAULT

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
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError) as exc:
        # Try the atomic .tmp sibling written by _save before rename
        tmp = path.with_suffix(".tmp")
        if tmp.exists():
            try:
                return json.loads(tmp.read_text())
            except (json.JSONDecodeError, ValueError):
                pass
        log.warning("attunement budget: corrupt file — failing closed (deny) for today")
        raise _BudgetCorruptError() from exc


def _save(persona_dir: Path, state: dict) -> None:
    path = _budget_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


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
