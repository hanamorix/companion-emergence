"""brain.notes.state — notes cadence persistence.

`notes_state.json` holds the last-note timestamp (cooldown gate); `notes_budget.json`
holds the per-day cap. Both fail-safe on corrupt/missing (mirrors
brain/maker/charge.py + brain/maker/budget.py).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

_STATE_FILE = "notes_state.json"
_BUDGET_FILE = "notes_budget.json"


@dataclass
class NotesState:
    last_note_at: str | None = None


def _state_path(persona_dir: Path) -> Path:
    return persona_dir / _STATE_FILE


def load_notes_state(persona_dir: Path) -> NotesState:
    """Load persisted notes state; cold default on missing/corrupt (fail-safe)."""
    path = _state_path(persona_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        last = raw.get("last_note_at")
        return NotesState(last_note_at=last if isinstance(last, str) and last else None)
    except (OSError, ValueError, TypeError):
        return NotesState(last_note_at=None)


def save_notes_state(persona_dir: Path, state: NotesState) -> None:
    path = _state_path(persona_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state)), encoding="utf-8")
    tmp.replace(path)


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
