"""brain.notes.config — notes engine constants + the cross-platform folder resolver."""
from __future__ import annotations

from pathlib import Path

from platformdirs import user_documents_dir

_NOTES_AWAY_HOURS = 12.0
_NOTES_COOLDOWN_HOURS = 24.0
_NOTES_DAILY_CAP = 1


def resolve_notes_folder(persona_name: str) -> Path:
    """The per-OS notes folder: <Documents>/<Persona> Notes. platformdirs gives
    the right Documents dir on macOS/Windows/Linux."""
    return Path(user_documents_dir()) / f"{persona_name} Notes"
