"""FeltTimeState — persisted snapshot of felt-time across bridge restarts.

Spec §2 `state.py` + §3 recovery model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from brain.health.attempt_heal import save_with_backup
from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

ANCHOR_TYPES = ("dream", "growth", "soul", "weather_shift")


@dataclass(frozen=True)
class Anchor:
    """A single felt-time anchor — a sparse semantic event marking time."""

    type: str
    ts: str  # ISO 8601 UTC
    label: str
    source_ref: str  # e.g. "dreams.log.jsonl:42" or "growth.log.jsonl:128"


@dataclass
class PressureCounters:
    heartbeats: int = 0
    chat_turns: int = 0
    reflex_firings: int = 0
    wall_clock_s: float = 0.0


@dataclass
class FeltTimeState:
    lived_age_hours: float = 0.0
    anchors: dict[str, Anchor] = field(default_factory=dict)  # type -> most recent
    pressure: PressureCounters = field(default_factory=PressureCounters)
    last_tick_ts: str | None = None  # ISO 8601 UTC of last tick()
    weather_baselines: dict[str, dict] = field(default_factory=dict)  # per-channel rolling baseline

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
        "weather_baselines": state.weather_baselines,
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

    return FeltTimeState(
        lived_age_hours=float(data.get("lived_age_hours", 0.0)),
        anchors={k: Anchor(**v) for k, v in (data.get("anchors") or {}).items()},
        pressure=PressureCounters(**(data.get("pressure") or {})),
        last_tick_ts=data.get("last_tick_ts"),
        weather_baselines=data.get("weather_baselines") or {},
    ), False
