"""pressure.py — tick aggregation of pressure-since-anchor counters.

Spec §2 pressure.py. Pure functions; the FeltTime orchestrator wires
the inputs. Counters reset to zero on any new anchor — see §3 data
flow: pressure_since_anchor[type] = ticks accumulated since the latest
anchor of *any* type.
"""

from __future__ import annotations

from dataclasses import dataclass

from brain.felt_time.state import Anchor, PressureCounters


@dataclass(frozen=True)
class TickInput:
    heartbeats: int
    chat_turns: int
    reflex_firings: int
    wall_clock_s_delta: float


def apply_tick(
    before: PressureCounters,
    *,
    tick: TickInput,
    new_anchors: list[Anchor],
) -> PressureCounters:
    """Returns the new PressureCounters after one supervisor tick.

    If new_anchors is non-empty, the counters reset to zero (the latest
    anchor in the batch is "now-zero"). Otherwise increment by the tick.
    """
    if new_anchors:
        # An anchor in this tick zeroes everything. Whether we got one
        # anchor or several, we land at the latest one.
        return PressureCounters()

    return PressureCounters(
        heartbeats=before.heartbeats + tick.heartbeats,
        chat_turns=before.chat_turns + tick.chat_turns,
        reflex_firings=before.reflex_firings + tick.reflex_firings,
        wall_clock_s=before.wall_clock_s + tick.wall_clock_s_delta,
    )
