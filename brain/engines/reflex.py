"""Reflex — autonomous emotional-threshold creative expression.

See docs/superpowers/specs/2026-04-24-week-4-reflex-engine-design.md.
This module ships the types, loaders, and engine scaffold. run_tick
evaluation + firing logic lands in Task 2.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.emotion.aggregate import aggregate_state
from brain.memory.store import Memory, MemoryStore

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
    """Autonomous emotional-threshold creative expression engine."""

    store: MemoryStore
    provider: LLMProvider
    persona_name: str
    persona_system_prompt: str
    arcs_path: Path
    log_path: Path
    default_arcs_path: Path

    def run_tick(self, *, trigger: str = "manual", dry_run: bool = False) -> ReflexResult:
        """Evaluate arcs against current state, fire at most one per tick."""
        now = datetime.now(UTC)

        arc_set = ReflexArcSet.load(self.arcs_path, default_path=self.default_arcs_path)
        log = ReflexLog.load(self.log_path)

        if not arc_set.arcs:
            return ReflexResult(
                arcs_fired=(),
                arcs_skipped=(ArcSkipped(arc_name="", reason="no_arcs_defined"),),
                would_fire=None,
                dry_run=dry_run,
                evaluated_at=now,
            )

        all_mems = self.store.search_text("", active_only=True, limit=None)
        state = aggregate_state(all_mems)
        days_since_human = _compute_days_since_human(self.store, now)

        eligible, skipped = self._evaluate(arc_set.arcs, state.emotions, days_since_human, log, now)

        if not eligible:
            return ReflexResult(
                arcs_fired=(),
                arcs_skipped=tuple(skipped),
                would_fire=None,
                dry_run=dry_run,
                evaluated_at=now,
            )

        winner = self._rank(eligible, state.emotions, log, now)
        losers = [a for a in eligible if a.name != winner.name]
        skipped.extend(ArcSkipped(arc_name=a.name, reason="single_fire_cap") for a in losers)

        if dry_run:
            return ReflexResult(
                arcs_fired=(),
                arcs_skipped=tuple(skipped),
                would_fire=winner.name,
                dry_run=True,
                evaluated_at=now,
            )

        fire = self._fire(winner, state.emotions, days_since_human, all_mems, now)
        new_log = log.appended(fire)
        new_log.save(self.log_path)

        return ReflexResult(
            arcs_fired=(fire,),
            arcs_skipped=tuple(skipped),
            would_fire=None,
            dry_run=False,
            evaluated_at=now,
        )

    def _evaluate(
        self,
        arcs: tuple[ReflexArc, ...],
        emotions: Mapping[str, float],
        days_since_human: float,
        log: ReflexLog,
        now: datetime,
    ) -> tuple[list[ReflexArc], list[ArcSkipped]]:
        eligible: list[ReflexArc] = []
        skipped: list[ArcSkipped] = []

        for arc in arcs:
            if not _trigger_met(arc, emotions):
                skipped.append(ArcSkipped(arc_name=arc.name, reason="trigger_not_met"))
                continue
            if days_since_human < arc.days_since_human_min:
                skipped.append(ArcSkipped(arc_name=arc.name, reason="days_since_human_too_low"))
                continue
            last = log.last_fire_for_arc(arc.name)
            if last is not None:
                hours_since = (now - last).total_seconds() / 3600.0
                if hours_since < arc.cooldown_hours:
                    skipped.append(ArcSkipped(arc_name=arc.name, reason="cooldown_active"))
                    continue
            eligible.append(arc)

        return eligible, skipped

    def _rank(
        self,
        eligible: list[ReflexArc],
        emotions: Mapping[str, float],
        log: ReflexLog,
        now: datetime,
    ) -> ReflexArc:
        """Highest aggregate threshold-excess wins; ties broken by longest-since-fire."""

        def key(arc: ReflexArc) -> tuple[float, float]:
            excess = sum(emotions.get(e, 0.0) - t for e, t in arc.trigger.items())
            last = log.last_fire_for_arc(arc.name)
            seconds_since = (now - last).total_seconds() if last is not None else float("inf")
            return (excess, seconds_since)

        return max(eligible, key=key)

    def _fire(
        self,
        arc: ReflexArc,
        emotions: Mapping[str, float],
        days_since_human: float,
        all_mems: list,
        now: datetime,
    ) -> ArcFire:
        """Render prompt → call LLM → write memory → return ArcFire."""
        context: dict = defaultdict(lambda: "0")
        context["persona_name"] = self.persona_name
        context["days_since_human"] = f"{days_since_human:.1f}"
        context["emotion_summary"] = _format_emotion_summary(emotions)
        context["memory_summary"] = _format_memory_summary(all_mems)
        for name, value in emotions.items():
            context[name] = f"{value:.1f}"

        prompt = arc.prompt_template.format_map(context)
        raw = self.provider.generate(prompt, system=self.persona_system_prompt)

        trigger_state = {e: emotions.get(e, 0.0) for e in arc.trigger}

        mem = Memory.create_new(
            content=raw,
            memory_type=arc.output_memory_type,
            domain="us",
            emotions={},
            metadata={
                "arc_name": arc.name,
                "trigger_state": trigger_state,
                "fired_at": _iso_utc(now),
                "provider": self.provider.name(),
            },
        )
        self.store.create(mem)

        return ArcFire(
            arc_name=arc.name,
            fired_at=now,
            trigger_state=trigger_state,
            output_memory_id=mem.id,
        )


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


def _trigger_met(arc: ReflexArc, emotions: Mapping[str, float]) -> bool:
    for name, threshold in arc.trigger.items():
        if emotions.get(name, 0.0) < threshold:
            return False
    return True


def _compute_days_since_human(store: MemoryStore, now: datetime) -> float:
    """Days since the most recent `conversation` memory. 999.0 if none exist."""
    convos = store.list_by_type("conversation", active_only=True, limit=1)
    if not convos:
        return 999.0
    latest = convos[0].created_at
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=UTC)
    return (now - latest).total_seconds() / 86400.0


def _format_emotion_summary(emotions: Mapping[str, float]) -> str:
    top = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return "\n".join(f"- {name}: {value:.1f}/10" for name, value in top)


def _format_memory_summary(memories: list) -> str:
    top = list(memories)[:3]
    return "\n".join(f"- {m.content[:140]}" for m in top)
