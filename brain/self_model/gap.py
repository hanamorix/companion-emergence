"""Gap — derived-minus-declared emotional divergence.

`compute_gap(declared, derived) -> Gap`

Pure compute, no I/O, no LLM.

The gap is the vector difference between the derived (trend/body) and the
declared (max-pool peak) emotional reads, restricted to REGISTERED channels.
Unregistered channel names — whether from stale vocabulary or synthetic tests —
are silently dropped (vocab-flood guard, R-A adjacent).

per_channel[c] = derived.channels.get(c, 0) − declared.emotions.get(c, 0)

for every channel c that:
  1. appears in EITHER derived.channels or declared.emotions, AND
  2. is registered in the emotion vocabulary (vocabulary.get(c) is not None).

Exact-zero deltas (declared == derived for that channel) are dropped from
per_channel to keep the dict sparse. This is a deliberate design choice:
a zero delta carries no information and would inflate magnitude calculations.

magnitude = sum(abs(v) for v in per_channel.values())

Orthogonality assertion (R-C1):
  - Divergent declared/derived   → magnitude > 0  (per_channel is non-empty)
  - Identical declared/derived   → magnitude == 0.0

unnamed_pressure is carried through from the derived read unchanged.

Gap dataclass fields per spec §4:
  per_channel        — {channel: delta}  registered-only, zero-deltas dropped
  magnitude          — sum of absolute deltas
  unnamed_pressure   — passed through from DerivedRead
  note               — optional Haiku-articulated note (None until Task 4)
  status             — "open" | "resolved" | "dismissed"
  first_seen_ts      — ISO-8601 UTC string, set by the caller / cadence layer
  last_seen_ts       — ISO-8601 UTC string, set by the caller / cadence layer
  sustained_ticks    — integer count, incremented by the cadence layer
  channel_cooldowns  — {channel: ISO-8601 UTC string} — set by reconcile layer
"""

from __future__ import annotations

from dataclasses import dataclass, field

from brain.emotion.state import EmotionalState
from brain.emotion.vocabulary import get as _get_emotion
from brain.self_model.derived import DerivedRead


@dataclass
class Gap:
    """The divergence between a companion's declared and derived emotional reads.

    See module docstring for invariants.

    Attributes:
        per_channel: {channel_name: delta} where delta = derived − declared.
            Registered channels only. Zero-delta channels are omitted.
        magnitude: sum(abs(delta) for delta in per_channel.values()).
            0.0 when declared and derived are identical on all registered channels.
        unnamed_pressure: residual body signal from DerivedRead that maps to no
            known channel. Carried through unchanged from the derived read.
        note: optional human-readable articulation from the Haiku articulate layer
            (Task 4). None until articulated.
        status: lifecycle marker — "open" | "resolved" | "dismissed".
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


def compute_gap(declared: EmotionalState, derived: DerivedRead) -> Gap:
    """Compute the gap between declared and derived emotional reads.

    Args:
        declared: The existing aggregate_state (max-pool peak over memories).
        derived:  The new orthogonal read (recency-weighted mean + body).

    Returns:
        Gap with registered-only per_channel deltas and passed-through
        unnamed_pressure. All persistence fields (ts, ticks, cooldowns) are
        at their zero/None defaults — the cadence layer sets them.
    """
    # Collect candidate channels: union of both reads, registered-only.
    candidate_channels: set[str] = set()
    for c in derived.channels:
        if _get_emotion(c) is not None:
            candidate_channels.add(c)
    for c in declared.emotions:
        if _get_emotion(c) is not None:
            candidate_channels.add(c)

    per_channel: dict[str, float] = {}
    for c in candidate_channels:
        delta = derived.channels.get(c, 0.0) - declared.emotions.get(c, 0.0)
        if delta != 0.0:
            per_channel[c] = delta

    magnitude = sum(abs(v) for v in per_channel.values())

    return Gap(
        per_channel=per_channel,
        magnitude=magnitude,
        unnamed_pressure=derived.unnamed_pressure,
    )
