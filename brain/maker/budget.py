"""brain.maker.budget — daily making cap (mirrors emotion_backfill budget)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_BUDGET_FILE = "maker_budget.json"


def _budget_path(persona_dir: Path) -> Path:
    return persona_dir / _BUDGET_FILE


def _today_str(now: datetime) -> str:
    return now.date().isoformat()


def consume_budget(persona_dir: Path, *, now: datetime, cap: int) -> bool:
    """Return True and decrement if under cap today; False if exhausted.
    Corrupt/missing file resets (fail-safe-permissive)."""
    path = _budget_path(persona_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, ValueError, OSError):
        raw = {}
    today = _today_str(now)
    if raw.get("date") != today:
        raw = {"date": today, "count": 0}
    if int(raw.get("count", 0)) >= cap:
        return False
    raw["count"] = int(raw.get("count", 0)) + 1
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw), encoding="utf-8")
    tmp.replace(path)
    return True
