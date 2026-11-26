"""FeltTimeState — persisted snapshot of felt-time across bridge restarts.

Spec §2 `state.py` + §3 recovery model.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from brain import dev_constants
from brain.health.attempt_heal import save_with_backup
from brain.health.json_io import Recovery, read_json
from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt
from brain.state_compat import from_known_fields

ANCHOR_TYPES = ("dream", "growth", "soul", "arc")


@dataclass(frozen=True)
class Anchor:
    """A single felt-time anchor — a sparse semantic event marking time."""

    type: str
    ts: str  # ISO 8601 UTC
    label: str
    source_ref: str  # e.g. "dreams.log.jsonl:42" or "growth.log.jsonl:128"
    event_type: str = ""  # populated for arc anchors: "arc_opened" | "arc_closed"


@dataclass
class PressureCounters:
    heartbeats: int = 0
    chat_turns: int = 0
    reflex_firings: int = 0
    wall_clock_s: float = 0.0


@dataclass
class HorizonBucket:
    """Pressure counters for a wall-clock horizon window (week or month).

    counters: running totals for the current (incomplete) period.
    prev_counters: snapshot of the last fully-completed period.
    period_start_ts: ISO 8601 UTC when the current period began.
    """

    counters: PressureCounters = field(default_factory=PressureCounters)
    prev_counters: PressureCounters = field(default_factory=PressureCounters)
    period_start_ts: str | None = None


@dataclass
class FeltTimeState:
    lived_age_hours: float = 0.0
    anchors: dict[str, Anchor] = field(default_factory=dict)  # type -> most recent
    pressure: PressureCounters = field(default_factory=PressureCounters)
    last_tick_ts: str | None = None  # ISO 8601 UTC of last tick()
    first_tick_ts: str | None = None  # ISO 8601 UTC of the FIRST tick (felt-time start anchor). None => not yet seeded.
    weather_baselines: dict[str, dict] = field(default_factory=dict)  # per-channel rolling baseline
    replayed: bool = False  # True iff state was rebuilt from JSONLs, not loaded from state file.
    horizon_pressure: dict[str, HorizonBucket] = field(default_factory=dict)
    arc_anchors: list[Anchor] = field(default_factory=list)
    # Body energy, a level carried from one heartbeat to the next (#650, spec 1.1, 1.3). A file written before these existed
    # loads the defaults. `energy_long_avg` is not read while `energy_long_weight` is 0 (no long average yet).
    energy_level: float = field(default_factory=lambda: dev_constants.BODY_ENERGY_BASELINE)
    energy_long_avg: float = 0.0
    energy_long_weight: float = 0.0
    # The recent averages of the lift-list and drain-list feelings as the previous heartbeat saw them (spec 1.10). None means
    # the field is absent from the file ("missing": the next heartbeat's feeling effect is 0 and it then writes the values).
    energy_feelings: dict[str, float] | None = None

    @classmethod
    def cold_start(cls) -> FeltTimeState:
        return cls()


STATE_FILENAME = "felt_time_state.json"


def persist(state: FeltTimeState, persona_dir: Path) -> None:
    """Atomic save via .bak rotation (save_with_backup) — same pattern as persona_config + user_preferences."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    target = persona_dir / STATE_FILENAME
    payload = {
        "lived_age_hours": state.lived_age_hours,
        "anchors": {k: asdict(v) for k, v in state.anchors.items()},
        "pressure": asdict(state.pressure),
        "last_tick_ts": state.last_tick_ts,
        "first_tick_ts": state.first_tick_ts,
        "weather_baselines": state.weather_baselines,
        "replayed": state.replayed,
        "horizon_pressure": {
            k: {
                "counters": asdict(v.counters),
                "prev_counters": asdict(v.prev_counters),
                "period_start_ts": v.period_start_ts,
            }
            for k, v in state.horizon_pressure.items()
        },
        "arc_anchors": [asdict(a) for a in state.arc_anchors],
        "energy_level": state.energy_level,
        "energy_long_avg": state.energy_long_avg,
        "energy_long_weight": state.energy_long_weight,
    }
    if state.energy_feelings is not None:  # absent from the file means "missing" (spec 1.10)
        payload["energy_feelings"] = state.energy_feelings
    save_with_backup(target, payload)


def _finite_number(value: object) -> float | None:
    """The value as a float when it is a real finite number (not a bool, a string, null, NaN or infinity), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _energy_feelings_of(raw: object) -> dict[str, float] | None:
    """The saved feeling averages when the whole map is sound (string names, finite numbers), else None (absent)."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for name, value in raw.items():
        number = _finite_number(value)
        if not isinstance(name, str) or number is None:
            return None  # a partial map would make the feeling effect from only some of the feelings
        out[name] = number
    return out


def _energy_level_of(data: dict) -> float:
    """The saved level in the file's data: the number as it is (no hold, no rounding), the baseline when absent or unsound."""
    level = _finite_number(data.get("energy_level"))
    return dev_constants.BODY_ENERGY_BASELINE if level is None else level


def read_energy_level(persona_dir: Path) -> float:
    """The saved energy level, unrounded: one light read of `felt_time_state.json` (spec 1.6).

    Readers use this, not `load_or_recover` (which scans four log files on every call). Keeps no copy of the file and never
    rewrites it (`Recovery.DEFAULT`). A missing file, an unreadable one, or a field that is absent or not a finite number
    gives the baseline.
    """
    return _energy_level_of(read_json(persona_dir / STATE_FILENAME, default=dict, recover=Recovery.DEFAULT, expect=dict))


LOG_SOURCES = ("dreams.log.jsonl", "growth.log.jsonl", "soul.log.jsonl")


def _newest_log_ts(persona_dir: Path) -> str | None:
    """Return the newest 'ts' field across the source JSONLs, or None."""
    newest: str | None = None
    for name in LOG_SOURCES:
        p = persona_dir / name
        if not p.exists():
            continue
        for row in iter_jsonl_skipping_corrupt(p):
            ts = row.get("ts")
            if ts and (newest is None or ts > newest):
                newest = ts
    return newest


def load_or_recover(persona_dir: Path) -> tuple[FeltTimeState, bool]:
    """Returns (state, recovered_from_logs)."""
    state_file = persona_dir / STATE_FILENAME
    if not state_file.exists():
        if _newest_log_ts(persona_dir) is not None:
            from brain.felt_time import _replay_from_logs  # local import avoids cycle

            return _replay_from_logs(persona_dir), True
        return FeltTimeState.cold_start(), True

    try:
        data = json.loads(state_file.read_text())
    except (OSError, json.JSONDecodeError):
        return FeltTimeState.cold_start(), True

    newest_log = _newest_log_ts(persona_dir)
    last_tick = data.get("last_tick_ts")
    if newest_log and (last_tick is None or newest_log > last_tick):
        from brain.felt_time import _replay_from_logs  # local import avoids cycle

        return _replay_from_logs(persona_dir), True

    # Deserialise horizon_pressure
    raw_hp = data.get("horizon_pressure") or {}
    horizon_pressure = {
        k: HorizonBucket(
            counters=from_known_fields(PressureCounters, v.get("counters") or {}),
            prev_counters=from_known_fields(PressureCounters, v.get("prev_counters") or {}),
            period_start_ts=v.get("period_start_ts"),
        )
        for k, v in raw_hp.items()
    }

    # Deserialise arc_anchors
    raw_arcs = data.get("arc_anchors") or []
    arc_anchors = [from_known_fields(Anchor, a) for a in raw_arcs]

    anchors = {k: from_known_fields(Anchor, v) for k, v in (data.get("anchors") or {}).items()}

    # Seed arc_anchors from anchors["arc"] when upgrading from old state format.
    if not arc_anchors and "arc" in anchors:
        arc_anchors = [anchors["arc"]]

    return FeltTimeState(
        lived_age_hours=float(data.get("lived_age_hours", 0.0)),
        anchors=anchors,
        pressure=from_known_fields(PressureCounters, data.get("pressure") or {}),
        last_tick_ts=data.get("last_tick_ts"),
        first_tick_ts=data.get("first_tick_ts"),
        weather_baselines=data.get("weather_baselines") or {},
        replayed=bool(data.get("replayed", False)),
        horizon_pressure=horizon_pressure,
        arc_anchors=arc_anchors,
        energy_level=_energy_level_of(data),
        energy_long_avg=_finite_number(data.get("energy_long_avg")) or 0.0,
        energy_long_weight=_finite_number(data.get("energy_long_weight")) or 0.0,
        energy_feelings=_energy_feelings_of(data.get("energy_feelings")),
    ), False
