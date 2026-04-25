"""Per-persona user preferences — the GUI-surfaceable cadence knobs.

Lives at `{persona_dir}/user_preferences.json`. This is the *only* file the
end-user GUI reads or writes. Everything else (heartbeat_config.json,
persona_config.json, the SQLite stores) is brain-internal.

Per principle audit 2026-04-25 (PR-C): the user surfaces are name, cadence
(this file), face/body, and reading generated documents. heartbeat_config.json
holds developer-only internal calibration (decay rates, GC thresholds, gating
thresholds) that the GUI must never expose.

Currently ships only `dream_every_hours`. Future GUI cadence knobs land here
(e.g., the Phase 2a `growth_every_hours` once that lands).

When both heartbeat_config.json and user_preferences.json define the same
cadence field, user_preferences.json wins — the GUI is authoritative.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DREAM_EVERY_HOURS = 24.0


@dataclass
class UserPreferences:
    """GUI-surfaceable cadence preferences.

    Hand-edited corruption degrades to defaults rather than crashing —
    same UX policy as HeartbeatConfig and PersonaConfig.
    """

    dream_every_hours: float = DEFAULT_DREAM_EVERY_HOURS

    @classmethod
    def load(cls, path: Path) -> UserPreferences:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()
        try:
            return cls(
                dream_every_hours=float(
                    data.get("dream_every_hours", DEFAULT_DREAM_EVERY_HOURS)
                ),
            )
        except (TypeError, ValueError):
            return cls()

    def save(self, path: Path) -> None:
        """Atomic save via .new + os.replace."""
        payload = {"dream_every_hours": self.dream_every_hours}
        tmp = path.with_suffix(path.suffix + ".new")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def read_raw_keys(path: Path) -> set[str]:
    """Return the set of keys explicitly present in user_preferences.json.

    Used by HeartbeatConfig.load to distinguish "file omits this field, fall
    back to heartbeat_config.json" from "file sets this field to its default
    value, override heartbeat_config.json". Returns empty set on any error
    (missing file, corrupt JSON, non-object payload).
    """
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(data, dict):
        return set()
    return set(data.keys())
