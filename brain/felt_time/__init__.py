"""brain.felt_time — felt-time texture for Nell.

Public surface:
    FeltTime          — orchestrator. tick(ctx) on the supervisor cadence;
                        get_state() to read; persists automatically.
    TickContext       — what the supervisor passes on each tick.

Spec: docs/superpowers/specs/2026-05-17-felt-time-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from brain.felt_time.anchors import scan_since
from brain.felt_time.lived_age import IntensityDrivers, advance
from brain.felt_time.pressure import TickInput, apply_tick
from brain.felt_time.state import (
    Anchor,
    FeltTimeState,
    PressureCounters,
    load_or_recover,
    persist,
)


@dataclass(frozen=True)
class TickContext:
    now_iso: str  # ISO 8601 UTC of this tick
    heartbeats_in_tick: int
    chat_turns_in_tick: int
    reflex_firings_in_tick: int
    wall_clock_s_in_tick: float
    drivers: IntensityDrivers


def _iso_to_seconds_delta(a: str | None, b: str) -> float:
    """Return wall-clock seconds from a → b. 0.0 when a is None."""
    if a is None:
        return 0.0
    from datetime import datetime

    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()


class FeltTime:
    """Composes state + anchors + pressure + lived_age behind one tick()."""

    def __init__(self, *, persona_dir: Path):
        self.persona_dir = persona_dir
        self._state, self._recovered = load_or_recover(persona_dir)

    @classmethod
    def from_logs(cls, *, persona_dir: Path) -> FeltTime:
        """Rebuild state purely from existing JSONL logs. Used by load_or_recover."""
        inst = cls.__new__(cls)
        inst.persona_dir = persona_dir
        inst._state = _replay_from_logs(persona_dir)
        inst._recovered = True
        return inst

    @property
    def recovered_from_logs(self) -> bool:
        return self._recovered

    def get_state(self) -> FeltTimeState:
        return self._state

    def tick(self, ctx: TickContext) -> None:
        new_anchors = scan_since(self.persona_dir, self._state.last_tick_ts)

        # Update anchors map — newest of each type wins.
        anchors = dict(self._state.anchors)
        for a in new_anchors:
            existing = anchors.get(a.type)
            if existing is None or a.ts > existing.ts:
                anchors[a.type] = a

        # Pressure aggregation.
        pressure = apply_tick(
            self._state.pressure,
            tick=TickInput(
                heartbeats=ctx.heartbeats_in_tick,
                chat_turns=ctx.chat_turns_in_tick,
                reflex_firings=ctx.reflex_firings_in_tick,
                wall_clock_s_delta=ctx.wall_clock_s_in_tick,
            ),
            new_anchors=new_anchors,
        )

        # Lived-age advancement.
        # When last_tick_ts is None (cold start or first tick after replay),
        # use wall_clock_s_in_tick as the dt — the supervisor knows the real
        # elapsed wall time even when we have no prior timestamp baseline.
        if self._state.last_tick_ts is None:
            dt_s = ctx.wall_clock_s_in_tick
        else:
            dt_s = _iso_to_seconds_delta(self._state.last_tick_ts, ctx.now_iso)
        lived = advance(
            prev_lived_hours=self._state.lived_age_hours,
            dt_seconds=dt_s,
            drivers=ctx.drivers,
        )

        self._state = FeltTimeState(
            lived_age_hours=lived,
            anchors=anchors,
            pressure=pressure,
            last_tick_ts=ctx.now_iso,
            weather_baselines=self._state.weather_baselines,
            replayed=False,  # first real tick clears the recovery banner
        )
        persist(self._state, self.persona_dir)


def _replay_from_logs(persona_dir: Path) -> FeltTimeState:
    """Reconstruct FeltTimeState from JSONLs only. Does NOT advance lived_age
    (we don't have intensity samples back in history). lived_age starts at 0
    after a replay — this is honest: a recovered brain doesn't get to fabricate
    its accumulated age."""
    all_anchors = scan_since(persona_dir, since_ts=None)
    anchors_by_type: dict[str, Anchor] = {}
    for a in all_anchors:
        existing = anchors_by_type.get(a.type)
        if existing is None or a.ts > existing.ts:
            anchors_by_type[a.type] = a

    # last_tick_ts stays None so the first real tick after replay will call
    # scan_since(None) and pick up all anchors as "new", giving it a chance
    # to reset pressure counters correctly. _iso_to_seconds_delta handles
    # None → uses wall_clock_s_in_tick from the TickContext instead.
    return FeltTimeState(
        lived_age_hours=0.0,
        anchors=anchors_by_type,
        pressure=PressureCounters(),
        last_tick_ts=None,
        weather_baselines={},
        replayed=True,
    )


__all__ = ["FeltTime", "TickContext", "IntensityDrivers"]
