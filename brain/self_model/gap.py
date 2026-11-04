"""Gap — short-minus-long (recent-vs-baseline) emotional divergence.

`compute_gap(short, long) -> Gap`

Pure compute, no I/O, no LLM.

The gap is the vector difference between the short (recent, felt-lately) and the
long (baseline) normalized recency-weighted mean reads, restricted to REGISTERED
channels. Unregistered channel names — whether from stale vocabulary or synthetic
tests — are silently dropped (vocab-flood guard, R-A adjacent).

per_channel[c] = short.channels.get(c, 0) − long.channels.get(c, 0)

for every channel c that:
  1. appears in EITHER short.channels or long.channels, AND
  2. is registered in the emotion vocabulary (vocabulary.get(c) is not None), AND
  3. has |delta| >= _GAP_NOISE_FLOOR.

The delta is BIDIRECTIONAL: >0 running above baseline lately (golden cross), <0
below baseline lately (death cross).

Below-floor deltas are dropped: a channel whose recent read is within
_GAP_NOISE_FLOOR of its baseline is at-baseline, and near-zero noise summed over
~20 channels would otherwise inflate magnitude to a few points on a persona that
is not actually diverging, so the gap would surface every reflection tick. The
floor makes magnitude reflect only genuine crosses; magnitude == 0 on a steady
persona (nothing surfaces), and a genuine cross survives at its full delta. The
floor equals ambient._CHANNEL_FLOOR (0.5) by design — a channel is surfaced only
when it diverges enough for the ambient block to name it (kept a local constant
here, not imported, to avoid a self_model→self_model import cycle).

magnitude = sum(abs(v) for v in per_channel.values())  # over survivors

unnamed_pressure is carried through from the short (felt) read unchanged.

Gap dataclass fields per spec §4:
  per_channel        — {channel: delta}  registered-only, zero-deltas dropped
  magnitude          — sum of absolute deltas
  unnamed_pressure   — passed through from DerivedRead
  note               — optional Haiku-articulated note (None until Task 4)
  status             — "open" | "acknowledged" | "dismissed" | "resolved"
  first_seen_ts      — ISO-8601 UTC string, set by the caller / cadence layer
  last_seen_ts       — ISO-8601 UTC string, set by the caller / cadence layer
  sustained_ticks    — integer count, incremented by the cadence layer
  channel_cooldowns  — {channel: ISO-8601 UTC string} — set by reconcile layer
  directions         — {channel: +1 | -1} the engine's shift direction per channel
                       (rising / fading); empty on a gap built by `compute_gap`
  reset_seen         — the engine's `status().reset_at` as the tick saw it second,
                       as an ISO-8601 UTC string; `UNSET` for a gap saved before
                       the field existed, None when the engine had never reset
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from brain.emotion.vocabulary import get as _get_emotion
from brain.self_model.derived import DerivedRead

if TYPE_CHECKING:
    from brain.emotion.engine import Reading

# A channel is surfaced only when its recent read diverges from baseline by at
# least this much (on the 0–10 intensity scale). Below-floor deltas are treated
# as at-baseline and dropped, so magnitude reflects only genuine crosses. Set
# equal to ambient._CHANNEL_FLOOR (0.5) by design — the same bar at which the
# ambient block would name the channel. Kept local (not imported) to avoid a
# self_model→self_model import cycle; the equality is intentional.
_GAP_NOISE_FLOOR: float = 0.5


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
        per_channel: {channel_name: delta} where delta = short − long.
            Registered channels only. Below-noise-floor channels are omitted.
            (A gap built by `gap_from_read` holds the engine's signed difference for
            each shifted channel instead, with no noise floor.)
        magnitude: sum(abs(delta) for delta in per_channel.values()).
            0.0 when short and long are within the noise floor on every channel.
        unnamed_pressure: residual body signal from the short read that maps to no
            known channel. Carried through unchanged from the short (felt) read.
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
            of `per_channel` (+1 rising, -1 fading). Empty when the gap did not come
            from the engine.
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


def compute_gap(short: DerivedRead, long: DerivedRead) -> Gap:
    """Compute the gap between the short (recent) and long (baseline) reads.

    Args:
        short: The felt-lately read (short-half-life normalized mean + body).
        long:  The baseline read (long-half-life normalized mean).

    Returns:
        Gap with registered-only per_channel deltas above the noise floor and
        passed-through unnamed_pressure. All persistence fields (ts, ticks,
        cooldowns) are at their zero/None defaults — the cadence layer sets them.
    """
    # Collect candidate channels: union of both reads, registered-only.
    candidate_channels: set[str] = set()
    for c in short.channels:
        if _get_emotion(c) is not None:
            candidate_channels.add(c)
    for c in long.channels:
        if _get_emotion(c) is not None:
            candidate_channels.add(c)

    per_channel: dict[str, float] = {}
    for c in candidate_channels:
        delta = short.channels.get(c, 0.0) - long.channels.get(c, 0.0)
        if abs(delta) >= _GAP_NOISE_FLOOR:
            per_channel[c] = delta

    magnitude = sum(abs(v) for v in per_channel.values())

    return Gap(
        per_channel=per_channel,
        magnitude=magnitude,
        unnamed_pressure=short.unnamed_pressure,
    )


def gap_from_read(read: Mapping[str, Reading], *, unnamed_pressure: float) -> Gap:
    """The gap from the emotion engine's `read()` (used when the engine is live).

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
