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
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.health.anomaly import BrainAnomaly

DEFAULT_DREAM_EVERY_HOURS = 24.0

logger = logging.getLogger(__name__)


def _default_user_preferences_dict() -> dict:
    return {"dream_every_hours": DEFAULT_DREAM_EVERY_HOURS}


@dataclass
class UserPreferences:
    """GUI-surfaceable cadence preferences.

    Hand-edited corruption degrades to defaults rather than crashing —
    same UX policy as HeartbeatConfig and PersonaConfig.
    """

    dream_every_hours: float = DEFAULT_DREAM_EVERY_HOURS

    @classmethod
    def _parse_data(cls, data: object) -> UserPreferences:
        """Build instance from already-parsed JSON data (dict expected)."""
        if not isinstance(data, dict):
            return cls()
        try:
            return cls(
                dream_every_hours=float(data.get("dream_every_hours", DEFAULT_DREAM_EVERY_HOURS)),
            )
        except (TypeError, ValueError):
            return cls()

    @classmethod
    def load_with_anomaly(cls, path: Path) -> tuple[UserPreferences, BrainAnomaly | None]:
        """Load with self-healing from .bak rotation if corrupt.

        Returns (instance, anomaly_or_None). Missing file → defaults, no anomaly.
        Corrupt file → quarantine + restore from .bak1/.bak2/.bak3 or reset.
        """
        from brain.health.attempt_heal import attempt_heal

        data, anomaly = attempt_heal(path, _default_user_preferences_dict)
        return cls._parse_data(data), anomaly

    @classmethod
    def load(cls, path: Path) -> UserPreferences:
        instance, anomaly = cls.load_with_anomaly(path)
        if anomaly is not None:
            logger.warning(
                "UserPreferences anomaly detected: %s action=%s file=%s",
                anomaly.kind,
                anomaly.action,
                anomaly.file,
            )
        return instance

    def save(self, path: Path) -> None:
        """Atomic save via .bak rotation (save_with_backup)."""
        from brain.health.adaptive import compute_treatment
        from brain.health.attempt_heal import save_with_backup

        payload = {"dream_every_hours": self.dream_every_hours}
        treatment = compute_treatment(path.parent, path.name)
        save_with_backup(path, payload, backup_count=treatment.backup_count)
        if treatment.verify_after_write:
            self._verify_after_write(path)

    def _verify_after_write(self, path: Path) -> None:
        """Re-read the written file; if corrupt, restore from .bak1."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("non-dict payload after write")
        except (json.JSONDecodeError, ValueError, OSError):
            logger.error(
                "UserPreferences verify_after_write failed for %s; restoring from .bak1", path
            )
            bak1 = path.with_name(path.name + ".bak1")
            if bak1.exists():
                shutil.copy2(bak1, path)


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
