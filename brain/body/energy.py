"""The per-heartbeat body energy step (#650, spec section 2): one pure function over a carried level.

Energy is a number carried from one heartbeat to the next (kept in felt time's state file, `brain.felt_time`). This module is
the arithmetic only: no file is read or written here. `brain.body.energy_inputs` gathers the inputs, `FeltTime.tick` calls
`advance_energy` and saves what it returns.

One heartbeat, worked out from the level `E` at its start and held once at the end (spec 2.3)::

    E_new = hold(E + MOOD + MSG + DRIFT)

The funnel scales a rise by `E / 5` and a drop by `5 / E`. The "used to" multipliers (spec 2.2) are 1 until the kindled is
older than 30 days.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from brain import dev_constants
from brain.emotion import engine_utils


@dataclass(frozen=True)
class EnergyMessage:
    """One message of the interval: its speaker's weight and its length factor (spec 2.3)."""

    weight: float
    length_factor: float


@dataclass(frozen=True)
class EnergyInputs:
    """Everything one tick's energy step reads, gathered by `brain.body.energy_inputs`.

    `interval_start` is the previous step's time (felt time's `last_tick_ts`, or now less the elapsed wall time when there is
    none). `short` and `long` are the feelings' recent and long averages from one `read()` (empty while a recovery rebuild is
    pending). `high` holds the lift-list and drain-list feelings the engine calls high now. `messages` are the interval's
    messages, `persona_start_at` is the kindled's start for the 30-day gate (None closes it).
    """

    interval_start: datetime
    short: Mapping[str, float]
    long: Mapping[str, float]
    high: frozenset[str]
    messages: tuple[EnergyMessage, ...]
    persona_start_at: datetime | None


@dataclass(frozen=True)
class EnergyCarry:
    """The carried numbers: the level and its long average with that average's weight (weight 0: no long average yet)."""

    level: float
    long_avg: float
    long_weight: float


@dataclass(frozen=True)
class EnergyResult:
    """The new carry, and the feelings to save (None when `read()` gave nothing: the saved ones stay)."""

    carry: EnergyCarry
    feelings: dict[str, float] | None


@dataclass(frozen=True)
class UsedTo:
    """The "used to" multipliers (spec 2.2); all 1 until they apply."""

    u_rise: float = 1.0
    u_drop: float = 1.0
    v_lift: float = 1.0
    v_drain: float = 1.0


def hold(level: float) -> float:
    """The level held to the 1 to 10 scale, never rounded (spec 1.4)."""
    return min(dev_constants.BODY_ENERGY_MAX, max(dev_constants.BODY_ENERGY_MIN, level))


def rise(level: float) -> float:
    """The funnel's factor on a rise: higher energy rises faster."""
    return level / dev_constants.BODY_ENERGY_FUNNEL_PIVOT


def drop(level: float) -> float:
    """The funnel's factor on a drop: lower energy drops faster."""
    return dev_constants.BODY_ENERGY_FUNNEL_PIVOT / level


def length_factor(length: int, average: float) -> float:
    """A message's length against its speaker's average: as it is up to the average, half the excess over it (spec 2.3).

    A speaker with no earlier message (average 0) gets 1, so there is no division by zero.
    """
    if average <= 0.0:
        return 1.0
    ratio = length / average
    if ratio <= 1.0:
        return ratio
    return 1.0 + (ratio - 1.0) * dev_constants.BODY_ENERGY_LENGTH_EXCESS_SHARE


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def used_to(long_avg: float, long_weight: float, long: Mapping[str, float], gate_open: bool) -> UsedTo:
    """The multipliers for one heartbeat (spec 2.2). `gate_open` is the kindled's 30-day test at that heartbeat's time.

    `U_*` compare the level's long average with the baseline and need a long average to exist (weight above 0). `V_*` compare
    the mean long average of the drain feelings with that of the lift feelings, a feeling the kindled lacks counting 0 and
    each mean floored at `BODY_ENERGY_MEAN_FLOOR`.
    """
    if not gate_open:
        return UsedTo()
    u_rise = u_drop = 1.0
    if long_weight > 0.0:
        average = hold(long_avg)
        baseline = dev_constants.BODY_ENERGY_BASELINE
        u_rise = average / baseline
        u_drop = baseline / average
    floor = dev_constants.BODY_ENERGY_MEAN_FLOOR
    drain_mean = _mean(long.get(name, 0.0) for name in dev_constants.BODY_ENERGY_DRAIN_FEELINGS)
    lift_mean = _mean(long.get(name, 0.0) for name in dev_constants.BODY_ENERGY_LIFT_FEELINGS)
    v_lift = max(floor, drain_mean) / max(floor, lift_mean)
    return UsedTo(u_rise=u_rise, u_drop=u_drop, v_lift=v_lift, v_drain=1.0 / v_lift)


def feeling_movement(short: Mapping[str, float], saved: Mapping[str, float] | None) -> float:
    """X: how far the lift and drain feelings' recent averages moved since the saved ones, drain moves subtracted, over 10.

    No saved values (a first heartbeat, a file from before #650) or nothing readable now gives 0. A feeling absent on either
    side counts 0 on that side (spec 1.10, 2.3).
    """
    if saved is None or not short:
        return 0.0
    lift = sum(short.get(name, 0.0) - saved.get(name, 0.0) for name in dev_constants.BODY_ENERGY_LIFT_FEELINGS)
    drain = sum(short.get(name, 0.0) - saved.get(name, 0.0) for name in dev_constants.BODY_ENERGY_DRAIN_FEELINGS)
    return (lift - drain) / dev_constants.EMOTION_VALUE_CAP


def step_heartbeat(
    level: float,
    *,
    short: Mapping[str, float],
    high: frozenset[str],
    messages: tuple[EnergyMessage, ...],
    x: float,
    multipliers: UsedTo,
) -> float:
    """One heartbeat: mood, each message's bracket and the drift, all from the level at its start, held once (spec 2.3)."""
    cap = dev_constants.EMOTION_VALUE_CAP
    cost = dev_constants.BODY_ENERGY_STEP
    start = hold(level)  # a stored level outside the scale (step 2 does not clamp on load) must not reach the funnel's division
    rising = rise(start) * multipliers.u_rise
    dropping = drop(start) * multipliers.u_drop

    mood = 0.0
    for name in dev_constants.BODY_ENERGY_LIFT_FEELINGS:
        if name in high:
            mood += (short.get(name, 0.0) / cap) * cost * rising * multipliers.v_lift
    for name in dev_constants.BODY_ENERGY_DRAIN_FEELINGS:
        if name in high:
            mood -= (short.get(name, 0.0) / cap) * cost * dropping * multipliers.v_drain

    # One bracket per message, funnelled by its sign: V scales the feeling part X only, then the base cost comes off.
    if x > 0.0:
        feeling_part = x * multipliers.v_lift
    elif x < 0.0:
        feeling_part = x * multipliers.v_drain
    else:
        feeling_part = 0.0
    bracket = feeling_part - cost
    if bracket > 0.0:
        per_message = bracket * rising
    elif bracket < 0.0:
        per_message = bracket * dropping
    else:
        per_message = 0.0
    messages_effect = sum(m.length_factor * m.weight * per_message for m in messages)

    baseline = dev_constants.BODY_ENERGY_BASELINE
    gap = baseline - start
    calm = max((short.get(name, 0.0) for name in dev_constants.BODY_ENERGY_CALM_FEELINGS), default=0.0)
    pace = dev_constants.BODY_ENERGY_DRIFT_STEP * (1.0 + calm / cap) * (rising if start < baseline else dropping)
    drift = math.copysign(min(abs(gap), pace), gap) if gap != 0.0 else 0.0  # the fixed step stops at the baseline

    return hold(start + mood + messages_effect + drift)


def _beat(
    carry: EnergyCarry,
    *,
    inputs: EnergyInputs,
    at: datetime,
    hours: float,
    messages: tuple[EnergyMessage, ...],
    x: float,
) -> EnergyCarry:
    """Step one heartbeat at time `at`, then fold the new level into the long average (spec 1.8, 2.6).

    The average's weight is aged by `hours` (the time since the previous fold, never negative) before the fold, with the
    engine's 30-day half-life taken in hours; the fold itself has weight 1, and the first one sets the average to the level.
    """
    gate_open = engine_utils.persona_past_long_window(inputs.persona_start_at, at)
    multipliers = used_to(carry.long_avg, carry.long_weight, inputs.long, gate_open)
    level = step_heartbeat(
        carry.level, short=inputs.short, high=inputs.high, messages=messages, x=x, multipliers=multipliers
    )
    half_life_hours = dev_constants.EMOTION_LONG_HALF_LIFE_D * 24.0
    average = engine_utils.Average(carry.long_avg, carry.long_weight * engine_utils.decay(max(0.0, hours), half_life_hours))
    engine_utils.fold_average(average, level, 1.0)
    return EnergyCarry(level=level, long_avg=average.avg, long_weight=average.weight)


def advance_energy(
    carry: EnergyCarry,
    *,
    prev_feelings: Mapping[str, float] | None,
    inputs: EnergyInputs,
    now: datetime,
) -> EnergyResult:
    """The caught-up heartbeats, then the current one (spec 2.5).

    Every whole heartbeat length since the previous step, less the heartbeat being taken now, is a caught-up heartbeat: closed
    time, sleep and a skipped heartbeat all count, a gap over six hours is not skipped, and there is no cap. Each has the mood
    and the drift and no messages (so X does not enter it), takes its own time for the 30-day gate and folds the long average
    aged by one heartbeat length. The current heartbeat has the interval's messages and X, and ages the average by the time
    since the last caught-up heartbeat.
    """
    beat = dev_constants.BODY_ENERGY_HEARTBEAT_S
    elapsed = (now - inputs.interval_start).total_seconds()
    caught_up = max(0, int(elapsed // beat) - 1)
    previous = inputs.interval_start
    for k in range(1, caught_up + 1):
        at = inputs.interval_start + timedelta(seconds=k * beat)
        carry = _beat(carry, inputs=inputs, at=at, hours=beat / 3600.0, messages=(), x=0.0)
        previous = at
    carry = _beat(
        carry,
        inputs=inputs,
        at=now,
        hours=(now - previous).total_seconds() / 3600.0,
        messages=inputs.messages,
        x=feeling_movement(inputs.short, prev_feelings),
    )
    feelings: dict[str, float] | None = None
    if inputs.short:
        names = dev_constants.BODY_ENERGY_LIFT_FEELINGS + dev_constants.BODY_ENERGY_DRAIN_FEELINGS
        feelings = {name: inputs.short[name] for name in names if name in inputs.short}
    return EnergyResult(carry=carry, feelings=feelings)
