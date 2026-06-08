"""pressure.py — tick aggregation of pressure-since-anchor counters.

Spec §2 pressure.py. Pure functions; the FeltTime orchestrator wires
the inputs. Counters reset to zero on any new anchor — see §3 data
flow: pressure_since_anchor[type] = ticks accumulated since the latest
anchor of *any* type.
"""

from __future__ import annotations

from dataclasses import dataclass

from brain.felt_time.state import Anchor, HorizonBucket, PressureCounters


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


_HORIZON_PERIOD_S: dict[str, float] = {
    "week": 7 * 24 * 3600.0,
    "month": 30 * 24 * 3600.0,
}


def apply_horizon_tick(
    buckets: dict[str, HorizonBucket],
    *,
    tick: TickInput,
    now_ts: str,
) -> dict[str, HorizonBucket]:
    """Return updated horizon buckets after one supervisor tick.

    Buckets accumulate the same deltas as the stretch-pressure counter
    but reset only on wall-clock period rollover — never on anchors.
    Fails open on a malformed now_ts (emits a warning, returns buckets
    unchanged so a timestamp bug never silently drops temporal context).
    """
    import warnings
    from datetime import datetime

    try:
        now = datetime.fromisoformat(now_ts)
    except ValueError:
        warnings.warn(
            f"apply_horizon_tick: malformed now_ts {now_ts!r}, skipping tick",
            stacklevel=2,
        )
        return dict(buckets)

    result: dict[str, HorizonBucket] = {}
    for key, period_s in _HORIZON_PERIOD_S.items():
        bucket = buckets.get(key) or HorizonBucket(period_start_ts=now_ts)

        # Rollover check — skip gracefully if period_start_ts is malformed.
        if bucket.period_start_ts is not None:
            try:
                start = datetime.fromisoformat(bucket.period_start_ts)
                if (now - start).total_seconds() >= period_s:
                    bucket = HorizonBucket(
                        counters=PressureCounters(),
                        prev_counters=PressureCounters(
                            heartbeats=bucket.counters.heartbeats,
                            chat_turns=bucket.counters.chat_turns,
                            reflex_firings=bucket.counters.reflex_firings,
                            wall_clock_s=bucket.counters.wall_clock_s,
                        ),
                        period_start_ts=now_ts,
                    )
            except ValueError:
                pass  # Malformed period_start_ts — accumulate without rolling.

        result[key] = HorizonBucket(
            counters=PressureCounters(
                heartbeats=bucket.counters.heartbeats + tick.heartbeats,
                chat_turns=bucket.counters.chat_turns + tick.chat_turns,
                reflex_firings=bucket.counters.reflex_firings + tick.reflex_firings,
                wall_clock_s=bucket.counters.wall_clock_s + tick.wall_clock_s_delta,
            ),
            prev_counters=bucket.prev_counters,
            period_start_ts=bucket.period_start_ts or now_ts,
        )

    return result
