"""Reflex — autonomous emotional-threshold creative expression.

See docs/superpowers/specs/2026-04-24-week-4-reflex-engine-design.md.
This module ships the types, loaders, and engine scaffold. run_tick
evaluation + firing logic lands in Task 2.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


# ---------- Types ----------


@dataclass(frozen=True)
class ReflexArc:
    """Definition of one emotional-threshold-triggered arc."""

    name: str
    description: str
    trigger: Mapping[str, float]
    days_since_human_min: float
    cooldown_hours: float
    action: str
    output_memory_type: str
    prompt_template: str

    @classmethod
    def from_dict(cls, data: dict) -> ReflexArc:
        """Construct an arc from a dict. Raises KeyError/ValueError on invalid input."""
        required = (
            "name",
            "description",
            "trigger",
            "days_since_human_min",
            "cooldown_hours",
            "action",
            "output_memory_type",
            "prompt_template",
        )
        for key in required:
            if key not in data:
                raise KeyError(f"ReflexArc missing required key: {key!r}")

        trigger_raw = data["trigger"]
        if not isinstance(trigger_raw, dict) or not trigger_raw:
            raise ValueError(f"ReflexArc {data.get('name')!r}: trigger must be non-empty dict")
        trigger = {str(k): float(v) for k, v in trigger_raw.items()}

        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            trigger=trigger,
            days_since_human_min=float(data["days_since_human_min"]),
            cooldown_hours=float(data["cooldown_hours"]),
            action=str(data["action"]),
            output_memory_type=str(data["output_memory_type"]),
            prompt_template=str(data["prompt_template"]),
        )


@dataclass(frozen=True)
class ArcFire:
    """Record of one arc firing."""

    arc_name: str
    fired_at: datetime  # tz-aware UTC
    trigger_state: Mapping[str, float]
    output_memory_id: str | None  # None for dry_run

    def to_dict(self) -> dict:
        return {
            "arc": self.arc_name,
            "fired_at": _iso_utc(self.fired_at),
            "trigger_state": dict(self.trigger_state),
            "output_memory_id": self.output_memory_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArcFire:
        return cls(
            arc_name=str(data["arc"]),
            fired_at=_parse_iso_utc(data["fired_at"]),
            trigger_state={str(k): float(v) for k, v in data.get("trigger_state", {}).items()},
            output_memory_id=data.get("output_memory_id"),
        )


@dataclass(frozen=True)
class ArcSkipped:
    """Record of one arc evaluated-but-not-fired."""

    arc_name: str
    reason: str  # trigger_not_met | days_since_human_too_low | cooldown_active | single_fire_cap | no_arcs_defined


@dataclass(frozen=True)
class ReflexResult:
    """Outcome of a single reflex evaluation pass."""

    arcs_fired: tuple[ArcFire, ...]
    arcs_skipped: tuple[ArcSkipped, ...]
    would_fire: str | None  # dry-run only
    dry_run: bool
    evaluated_at: datetime


# ---------- Storage ----------


@dataclass(frozen=True)
class ReflexArcSet:
    """Loaded set of ReflexArc definitions."""

    arcs: tuple[ReflexArc, ...]

    @classmethod
    def load(cls, path: Path, *, default_path: Path) -> ReflexArcSet:
        """Load arcs from path, falling back to default_path on corrupt/missing.

        Per-arc validation failures skip that arc, log warning, keep others.
        """
        source_path = path if path.exists() else default_path
        if source_path != path:
            logger.warning("reflex arcs file %s not found, using defaults", path)

        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("reflex arcs load failed (%s), falling back to defaults", exc)
            data = json.loads(default_path.read_text(encoding="utf-8"))

        if not isinstance(data, dict) or "arcs" not in data or not isinstance(data["arcs"], list):
            logger.warning(
                "reflex arcs schema invalid at %s, falling back to defaults", source_path
            )
            data = json.loads(default_path.read_text(encoding="utf-8"))

        arcs: list[ReflexArc] = []
        for raw in data["arcs"]:
            try:
                arcs.append(ReflexArc.from_dict(raw))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("reflex arc %r failed to load: %s", raw.get("name"), exc)
                continue
        return cls(arcs=tuple(arcs))


@dataclass(frozen=True)
class ReflexLog:
    """Fire-history log for one persona."""

    fires: tuple[ArcFire, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, path: Path) -> ReflexLog:
        """Load the log; return empty on corrupt/missing."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            fires_raw = data.get("fires", [])
            if not isinstance(fires_raw, list):
                return cls()
            fires = tuple(ArcFire.from_dict(f) for f in fires_raw if isinstance(f, dict))
            return cls(fires=fires)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return cls()

    def save(self, path: Path) -> None:
        """Atomic save via write-to-.new + os.replace."""
        payload = {
            "version": 1,
            "fires": [f.to_dict() for f in self.fires],
        }
        tmp = path.with_suffix(path.suffix + ".new")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def last_fire_for_arc(self, arc_name: str) -> datetime | None:
        """Return the most recent fired_at for the given arc, or None."""
        most_recent: datetime | None = None
        for fire in self.fires:
            if fire.arc_name != arc_name:
                continue
            if most_recent is None or fire.fired_at > most_recent:
                most_recent = fire.fired_at
        return most_recent

    def appended(self, fire: ArcFire) -> ReflexLog:
        """Return a new ReflexLog with `fire` appended."""
        return ReflexLog(fires=self.fires + (fire,))


# ---------- Engine ----------


@dataclass
class ReflexEngine:
    """Autonomous emotional-threshold creative expression engine.

    run_tick() implementation ships in Task 2.
    """

    store: MemoryStore
    provider: LLMProvider
    persona_name: str
    persona_system_prompt: str
    arcs_path: Path
    log_path: Path
    default_arcs_path: Path

    def run_tick(self, *, trigger: str = "manual", dry_run: bool = False) -> ReflexResult:
        raise NotImplementedError("run_tick body lands in Task 2")


# ---------- Helpers ----------


def _iso_utc(dt: datetime) -> str:
    """ISO-8601 with Z suffix. Requires tz-aware datetime."""
    if dt.tzinfo is None:
        raise ValueError("_iso_utc requires a tz-aware datetime")
    return dt.isoformat().replace("+00:00", "Z")


def _parse_iso_utc(s: str) -> datetime:
    """Parse ISO-8601 Z-suffix timestamp back to tz-aware datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
