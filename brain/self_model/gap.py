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
"""

from __future__ import annotations

from dataclasses import dataclass, field

from brain.emotion.vocabulary import get as _get_emotion
from brain.self_model.derived import DerivedRead

# A channel is surfaced only when its recent read diverges from baseline by at
# least this much (on the 0–10 intensity scale). Below-floor deltas are treated
# as at-baseline and dropped, so magnitude reflects only genuine crosses. Set
# equal to ambient._CHANNEL_FLOOR (0.5) by design — the same bar at which the
# ambient block would name the channel. Kept local (not imported) to avoid a
# self_model→self_model import cycle; the equality is intentional.
_GAP_NOISE_FLOOR: float = 0.5


@dataclass
class Gap:
    """The divergence between a companion's declared and derived emotional reads.

    See module docstring for invariants.

    Attributes:
        per_channel: {channel_name: delta} where delta = short − long.
            Registered channels only. Below-noise-floor channels are omitted.
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
