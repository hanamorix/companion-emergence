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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from brain.health.json_io import Recovery, read_json, read_json_result, write_json

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
        res = read_json_result(path, default=_default_user_preferences_dict, recover=Recovery.HEAL)
        return cls._parse_data(res.data), res.anomaly

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
        """Atomic save via .bak rotation (write_json)."""
        from brain.health.adaptive import compute_treatment

        payload = {"dream_every_hours": self.dream_every_hours}
        treatment = compute_treatment(path.parent, path.name)
        write_json(path, payload, backups=treatment.backup_count, verify=treatment.verify_after_write)


def read_raw_keys(path: Path) -> set[str]:
    """Return the set of keys explicitly present in user_preferences.json.

    Used by HeartbeatConfig.load to distinguish "file omits this field, fall
    back to heartbeat_config.json" from "file sets this field to its default
    value, override heartbeat_config.json". Returns empty set for a missing file,
    corrupt JSON or a non-object payload.
    """
    # CONTENT: a missing, corrupt or non-object file is the empty set; an unreadable one still raises.
    return set(read_json(path, default=dict, recover=Recovery.CONTENT, expect=dict).keys())
