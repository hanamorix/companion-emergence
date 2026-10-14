"""FeltTimeState — persisted snapshot of felt-time across bridge restarts.

Spec §2 `state.py` + §3 recovery model.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from brain.health.attempt_heal import save_with_backup
from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt
from brain.state_compat import from_known_fields

ANCHOR_TYPES = ("dream", "growth", "soul", "weather_shift", "arc")


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
    }
    save_with_backup(target, payload)


LOG_SOURCES = ("dreams.log.jsonl", "growth.log.jsonl", "soul.log.jsonl", "weather_shifts.log.jsonl")


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


def _is_real(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_counters(pc: PressureCounters) -> None:
    for name in ("heartbeats", "chat_turns", "reflex_firings", "wall_clock_s"):
        if not _is_real(getattr(pc, name)):
            raise TypeError(f"pressure counter {name} is not a finite number")


def _check_anchor(a: Anchor) -> None:
    for name in ("type", "ts", "label", "source_ref", "event_type"):
        if not isinstance(getattr(a, name), str):
            raise TypeError(f"anchor field {name} is not a string")


def _check_iso(value: object, key: str) -> None:
    """None or an ISO-8601 string that datetime.fromisoformat accepts (tick() parses it)."""
    from datetime import datetime

    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{key} is not a string")
    datetime.fromisoformat(value)


def _validate_state(state: FeltTimeState) -> None:
    """Raise TypeError/ValueError if any leaf of a deserialised state has the wrong type
    or an impossible value (a wrong-typed field counts as corrupt). Pure."""
    if not _is_real(state.lived_age_hours) or state.lived_age_hours < 0:
        raise TypeError("lived_age_hours is not a finite non-negative number")
    _check_iso(state.last_tick_ts, "last_tick_ts")
    _check_iso(state.first_tick_ts, "first_tick_ts")
    _check_counters(state.pressure)
    for a in (*state.anchors.values(), *state.arc_anchors):
        _check_anchor(a)
    for bucket in state.horizon_pressure.values():
        _check_counters(bucket.counters)
        _check_counters(bucket.prev_counters)
        if bucket.period_start_ts is not None and not isinstance(bucket.period_start_ts, str):
            raise TypeError("period_start_ts is not a string")


def _state_from_data(data: object) -> FeltTimeState:
    """Build a FeltTimeState from parsed JSON. Raises ValueError, TypeError,
    AttributeError or KeyError when the data is not a well-typed state (a wrong-typed
    field counts as corrupt). Pure: touches no files."""
    if not isinstance(data, dict):
        raise TypeError("felt-time state is not a JSON object")
    weather_baselines = data.get("weather_baselines") or {}
    if not isinstance(weather_baselines, dict):
        raise TypeError("weather_baselines is not an object")

    raw_hp = data.get("horizon_pressure") or {}
    horizon_pressure = {
        k: HorizonBucket(
            counters=from_known_fields(PressureCounters, v.get("counters") or {}),
            prev_counters=from_known_fields(PressureCounters, v.get("prev_counters") or {}),
            period_start_ts=v.get("period_start_ts"),
        )
        for k, v in raw_hp.items()
    }

    raw_arcs = data.get("arc_anchors") or []
    arc_anchors = [from_known_fields(Anchor, a) for a in raw_arcs]

    anchors = {k: from_known_fields(Anchor, v) for k, v in (data.get("anchors") or {}).items()}

    # Seed arc_anchors from anchors["arc"] when upgrading from old state format.
    if not arc_anchors and "arc" in anchors:
        arc_anchors = [anchors["arc"]]

    state = FeltTimeState(
        lived_age_hours=data.get("lived_age_hours", 0.0),
        anchors=anchors,
        pressure=from_known_fields(PressureCounters, data.get("pressure") or {}),
        last_tick_ts=data.get("last_tick_ts"),
        first_tick_ts=data.get("first_tick_ts"),
        weather_baselines=weather_baselines,
        replayed=bool(data.get("replayed", False)),
        horizon_pressure=horizon_pressure,
        arc_anchors=arc_anchors,
    )
    _validate_state(state)
    state.lived_age_hours = float(state.lived_age_hours)
    return state


def _read_state_file(path: Path) -> FeltTimeState | None:
    """Read-only: the well-typed state in `path`, or None when it is unreadable,
    unparseable or wrong-typed."""
    try:
        return _state_from_data(json.loads(path.read_text()))
    except (OSError, ValueError, TypeError, AttributeError, KeyError, OverflowError, RecursionError):
        return None


BACKUP_COUNT = 3  # save_with_backup's default rotation: .bak1 (newest) .. .bak3


def load_or_recover(persona_dir: Path) -> tuple[FeltTimeState, bool]:
    """Returns (state, recovered). recovered is False only for a clean load of the
    live state file. Read-only: nothing here writes (only FeltTime.tick persists).

    - State file parses and is well-typed: loaded as-is. Newer anchor-log rows are
      NOT a reason to discard it; the next FeltTime.tick folds them in (#384).
    - State file missing or corrupt (unparseable or a wrong-typed field): walk .bak1,
      .bak2, .bak3 without touching them; the first good one is used as saved.
    - No good state file or backup, anchor logs exist: replayed from the logs (lived
      age 0.0, the recovery honesty contract).
    - Nothing to recover from (a brand-new persona): cold start.
    """
    state_file = persona_dir / STATE_FILENAME
    # _read_state_file is None for a missing, unreadable or corrupt file alike. A missing
    # main file is not "no state": save_with_backup renames live -> .bak1 just before the
    # new file lands, so a reader in that window must still find the backups.
    state = _read_state_file(state_file)
    if state is not None:
        return state, False
    for i in range(1, BACKUP_COUNT + 1):
        bak = state_file.with_name(f"{state_file.name}.bak{i}")
        if (state := _read_state_file(bak)) is not None:
            return state, True

    if _newest_log_ts(persona_dir) is not None:
        from brain.felt_time import _replay_from_logs  # local import avoids cycle

        return _replay_from_logs(persona_dir), True
    return FeltTimeState.cold_start(), True
