"""Read/write app_config.json — NellFace selection state.

app_config.json lives at $KINDLED_HOME/app_config.json (next to personas/).
NellFace reads it on boot to decide which Kindled to wake. CLI write_if_missing
keeps the file in sync for users who set up via `nell init` rather than the wizard.
"""
from __future__ import annotations

import json

from brain.paths import get_home


def _path():
    return get_home() / "app_config.json"


def write_if_missing(persona: str) -> None:
    """Write app_config.json with selected_persona=persona iff the file does not exist.

    Per spec §Open Questions: existing files (even with selected_persona = null)
    are left untouched. The Tauri boot path handles the "exists but null" case
    via list_personas autodetect.
    """
    target = _path()
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({
            "selected_persona": persona,
            "always_on_top": False,
            "reduced_motion": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
