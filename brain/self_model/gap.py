"""Gap — the emotion engine's shifted channels, as a self-model gap.

`gap_from_read(read, *, unnamed_pressure) -> Gap`

Pure compute, no I/O, no LLM.

A channel is in the gap when the engine's `read()` reports it as shifted (`shift != 0`): it is valued at its signed
`difference` (short average minus long average) and carries the shift as its direction (+1 rising, -1 fading).
The engine's read already lists only registered names and the shift rule already decided what counts, so there is
no name test and no noise floor here.

magnitude = sum(abs(v) for v in per_channel.values())

Gap dataclass fields per spec §4:
  per_channel        — {channel: signed difference} for each shifted channel
  magnitude          — sum of absolute differences
  unnamed_pressure   — the body's residual signal, passed in by the caller
  note               — optional Haiku-articulated note (None until Task 4)
  status             — "open" | "acknowledged" | "dismissed" | "resolved"
  first_seen_ts      — ISO-8601 UTC string, set by the caller / cadence layer
  last_seen_ts       — ISO-8601 UTC string, set by the caller / cadence layer
  sustained_ticks    — integer count, incremented by the cadence layer
  channel_cooldowns  — {channel: ISO-8601 UTC string} — set by reconcile layer
  directions         — {channel: +1 | -1} the engine's shift direction per channel
                       (rising / fading)
  reset_seen         — the engine's `status().reset_at` as the tick saw it second,
                       as an ISO-8601 UTC string; `UNSET` for a gap saved before
                       the field existed, None when the engine had never reset
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.emotion.engine import Reading


class _Unset:
    """The type of `UNSET`: "this field was never stored" (distinct from a stored None)."""

    _instance: _Unset | None = None

    def __new__(cls) -> _Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


@dataclass
class Gap:
    """The divergence between a companion's declared and derived emotional reads.

    See module docstring for invariants.

    Attributes:
        per_channel: {channel_name: delta} where delta is the engine's signed difference
            (short average minus long average) for each shifted channel.
        magnitude: sum(abs(delta) for delta in per_channel.values()).
        unnamed_pressure: residual body signal that maps to no present channel.
        note: optional human-readable articulation from the Haiku articulate layer
            (Task 4). None until articulated.
        status: lifecycle marker — "open" | "acknowledged" | "dismissed" | "resolved".
        first_seen_ts: ISO-8601 UTC timestamp set when the gap is first persisted.
        last_seen_ts: ISO-8601 UTC timestamp updated on each cadence tick.
        sustained_ticks: number of consecutive cadence ticks this gap has been open.
        channel_cooldowns: {channel: ISO-8601 UTC expiry} set by the reconcile layer
            after a self-authored revision; no new gap surfaced for that channel
            until the expiry passes (R-B2).
        directions: {channel: +1 | -1}, the engine's shift direction for each channel
            of `per_channel` (+1 rising, -1 fading).
        reset_seen: the engine's `reset_at` as the tick saw it second, an ISO-8601 UTC
            string. `UNSET` means "never stored" (a gap saved before this field);
            None is a stored "no reset yet".
    """

    per_channel: dict[str, float]
    magnitude: float
    unnamed_pressure: float
    note: str | None = None
    status: str = "open"
    first_seen_ts: str | None = None
    last_seen_ts: str | None = None
    sustained_ticks: int = 0
    channel_cooldowns: dict[str, str] = field(default_factory=dict)
    directions: dict[str, int] = field(default_factory=dict)
    reset_seen: str | None | _Unset = UNSET


def gap_from_read(read: Mapping[str, Reading], *, unnamed_pressure: float) -> Gap:
    """The gap from the emotion engine's `read()`.

    A channel opens when the engine reads it as shifted (`shift != 0`); it is valued at its signed `difference`
    (0.0 allowed) and carries the shift as its direction (+1 rising, -1 fading). No name test (`read()` already
    lists only registered names) and no noise floor (the shift rule decided). `magnitude` is the sum of the
    absolute differences. Persistence fields stay at their defaults for the cadence layer.
    """
    per_channel: dict[str, float] = {}
    directions: dict[str, int] = {}
    for name, reading in read.items():
        if reading.shift != 0:
            per_channel[name] = reading.difference
            directions[name] = reading.shift
    return Gap(
        per_channel=per_channel,
        magnitude=sum(abs(v) for v in per_channel.values()),
        unnamed_pressure=unnamed_pressure,
        directions=directions,
    )
