"""brain.maker.budget — daily making cap (mirrors emotion_backfill budget)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from brain.health.json_io import read_json, write_json

_BUDGET_FILE = "maker_budget.json"


def _budget_path(persona_dir: Path) -> Path:
    return persona_dir / _BUDGET_FILE


def _today_str(now: datetime) -> str:
    return now.date().isoformat()


def consume_budget(persona_dir: Path, *, now: datetime, cap: int) -> bool:
    """Return True and decrement if under cap today; False if exhausted.
    Corrupt/missing file resets (fail-safe-permissive)."""
    path = _budget_path(persona_dir)
    raw = read_json(path, default=dict, expect=dict)
    today = _today_str(now)
    if raw.get("date") != today:
        raw = {"date": today, "count": 0}
    if int(raw.get("count", 0)) >= cap:
        return False
    raw["count"] = int(raw.get("count", 0)) + 1
    write_json(path, raw, indent=None, fsync=False)
    return True
