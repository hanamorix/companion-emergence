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

from brain.felt_time.anchors import extract_all, scan_since
from brain.felt_time.lived_age import _MIGRATION_SEED_RATE, IntensityDrivers, advance
from brain.felt_time.pressure import TickInput, apply_horizon_tick, apply_tick
from brain.felt_time.state import (
    Anchor,
    FeltTimeState,
    HorizonBucket,
    PressureCounters,
    load_or_recover,
    persist,
)

_ARC_ANCHOR_CAP = 20


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


def _compute_first_tick_ts(prev: FeltTimeState, now_iso: str) -> str:
    """Seed/carry the felt-time start anchor (#67).

    - already set → carry it forward;
    - last_tick None (fresh persona, true first tick) → now;
    - else (existing persona that predates the anchor) → back-date at the
      midpoint rate so the computed rate starts at ~_MIGRATION_SEED_RATE and
      self-corrects as new ticks accumulate.
    """
    from datetime import datetime, timedelta

    if prev.first_tick_ts is not None:
        return prev.first_tick_ts
    if prev.last_tick_ts is None:
        return now_iso
    wall_hours = prev.lived_age_hours / _MIGRATION_SEED_RATE
    return (
        datetime.fromisoformat(now_iso) - timedelta(hours=wall_hours)
    ).isoformat()


def _arc_log_index(anchor: Anchor) -> int | None:
    """Position of an arc anchor in arcs.log.jsonl, from its source_ref
    ("arcs.log.jsonl:N", N = 1-based valid-entry index, i.e. append order).
    None when the ref has no parseable index."""
    _, _, tail = anchor.source_ref.rpartition(":")
    return int(tail) if tail.isdigit() else None


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
        # Parse the logs once. new_anchors is exactly scan_since(last_tick_ts):
        # strictly newer than the last tick; it drives the per-type anchors map and
        # the pressure reset.
        all_anchors = extract_all(self.persona_dir)
        last_tick_ts = self._state.last_tick_ts
        new_anchors = [a for a in all_anchors if last_tick_ts is None or a.ts > last_tick_ts]

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

        # Arc anchor list — append arc events not yet folded in, cap at _ARC_ANCHOR_CAP.
        # Selected by position in arcs.log.jsonl, not by timestamp: an arc_opened row is
        # stamped with its SEED anchor's time, which is at or before the last tick, so a
        # timestamp scan would never see it. Folding by position is idempotent (an event
        # is folded once, never re-added after the cap evicts it) and does not reset
        # pressure (only the timestamp-newer new_anchors above do).
        arc_anchors = list(self._state.arc_anchors)
        folded_upto = max((i for i in map(_arc_log_index, arc_anchors) if i is not None), default=0)
        new_arc_anchors = sorted(
            (a for a in all_anchors if a.type == "arc" and (_arc_log_index(a) or 0) > folded_upto),
            key=lambda a: _arc_log_index(a) or 0,  # log order, so the cap evicts the oldest-appended
        )
        # Known limit: if arcs.log.jsonl were ever replaced by a shorter log, folding pauses
        # until it grows past folded_upto. Nothing in brain/ rotates or truncates it.
        arc_anchors.extend(new_arc_anchors)
        if len(arc_anchors) > _ARC_ANCHOR_CAP:
            arc_anchors = arc_anchors[-_ARC_ANCHOR_CAP:]
        # Keep anchors["arc"] in sync with the most recent arc anchor.
        if arc_anchors:
            anchors["arc"] = arc_anchors[-1]

        # Horizon buckets — accumulate on wall-clock cadence (not anchor-reset).
        horizon_pressure = apply_horizon_tick(
            self._state.horizon_pressure,
            tick=TickInput(
                heartbeats=ctx.heartbeats_in_tick,
                chat_turns=ctx.chat_turns_in_tick,
                reflex_firings=ctx.reflex_firings_in_tick,
                wall_clock_s_delta=ctx.wall_clock_s_in_tick,
            ),
            now_ts=ctx.now_iso,
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

        first_tick_ts = _compute_first_tick_ts(self._state, ctx.now_iso)
        self._state = FeltTimeState(
            lived_age_hours=lived,
            anchors=anchors,
            pressure=pressure,
            last_tick_ts=ctx.now_iso,
            first_tick_ts=first_tick_ts,
            weather_baselines=self._state.weather_baselines,
            replayed=False,  # first real tick clears the recovery banner
            horizon_pressure=horizon_pressure,
            arc_anchors=arc_anchors,
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

    # Seed arc_anchors from all arc events found in logs.
    arc_anchors_from_logs = [a for a in all_anchors if a.type == "arc"]
    if len(arc_anchors_from_logs) > _ARC_ANCHOR_CAP:
        arc_anchors_from_logs = arc_anchors_from_logs[-_ARC_ANCHOR_CAP:]

    # last_tick_ts stays None so the first real tick after replay will call
    # scan_since(None) and pick up all anchors as "new", giving it a chance
    # to reset pressure counters correctly. _iso_to_seconds_delta handles
    # None → uses wall_clock_s_in_tick from the TickContext instead.
    return FeltTimeState(
        lived_age_hours=0.0,
        anchors=anchors_by_type,
        pressure=PressureCounters(),
        last_tick_ts=None,
        first_tick_ts=None,
        weather_baselines={},
        replayed=True,
        horizon_pressure={},
        arc_anchors=arc_anchors_from_logs,
    )


__all__ = ["FeltTime", "TickContext", "IntensityDrivers", "HorizonBucket"]
