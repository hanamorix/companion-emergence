"""Body state — pure-function projection over emotions + session inputs.

The brain has a body. Body emotions live in the emotion vocabulary
(arousal, desire, climax, touch_hunger, comfort_seeking, rest_need).
The *projections* (energy, temperature, exhaustion) are computed
fresh on each call — no persistence, no cache, no parallel state.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# The 6-emotion body set tracked by compute_body_state.
# Reconciliation 2026-04-30: arousal + desire are existing core emotions
# (not new); the four new ones live in vocabulary._BASELINE under category="body".
BODY_EMOTION_NAMES: frozenset[str] = frozenset(
    {
        "arousal",
        "desire",
        "climax",
        "touch_hunger",
        "comfort_seeking",
        "rest_need",
    }
)


@dataclass(frozen=True)
class BodyState:
    """The computed body view at a moment in time.

    Energy 1-10, temperature 1-9 (asymmetric, midpoint 5; OG inheritance),
    exhaustion derived as max(0, 7 - energy).

    body_emotions carries the six body-class emotions as a snapshot —
    callers shouldn't have to re-aggregate to read them.
    """

    energy: int
    temperature: int
    exhaustion: int
    session_hours: float
    days_since_contact: float
    body_emotions: dict[str, float]
    computed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the get_body_state tool result.

        `loaded: True` distinguishes the real impl from the legacy stub —
        the brain knows the body module is real when she calls the tool.
        """
        return {
            "loaded": True,
            "energy": self.energy,
            "temperature": self.temperature,
            "exhaustion": self.exhaustion,
            "session_hours": round(self.session_hours, 2),
            "days_since_contact": round(self.days_since_contact, 2),
            "body_emotions": {k: round(v, 1) for k, v in self.body_emotions.items()},
            "computed_at": self.computed_at.isoformat(),
        }


def compute_body_state(
    *,
    emotions: Mapping[str, float],
    session_hours: float,
    words_written: int,
    days_since_contact: float,
    now: datetime,
) -> BodyState:
    """Pure projection — no I/O, no LLM call, no cache. Sub-millisecond.

    `emotions` MUST be the post-aggregation, post-climax-reset state
    (i.e. what aggregate_state() returns). compute_body_state does not
    re-aggregate or re-reset.
    """
    energy = _compute_energy(emotions, session_hours, words_written)
    temperature = _compute_temperature(emotions, days_since_contact)
    exhaustion = max(0, 7 - energy)
    body_emotions = {name: float(emotions.get(name, 0.0)) for name in BODY_EMOTION_NAMES}
    return BodyState(
        energy=energy,
        temperature=temperature,
        exhaustion=exhaustion,
        session_hours=round(session_hours, 2),
        days_since_contact=round(days_since_contact, 2),
        body_emotions=body_emotions,
        computed_at=now,
    )


def _compute_energy(
    emotions: Mapping[str, float],
    session_hours: float,
    words_written: int,
) -> int:
    """Energy 1-10, baseline 8. Drains for session length + creative work +
    high emotional load + body asking for rest. Restores when peace is
    high in a fresh session. Spec §3.1.
    """
    energy = 8.0

    # Session-duration drain (banded; stacked with the continuous term below).
    session_minutes = session_hours * 60.0
    if session_minutes > 180:
        energy -= 3
    elif session_minutes > 120:
        energy -= 2
    elif session_minutes > 60:
        energy -= 1

    # Continuous session drain (compounds with the band — long sessions feel longer).
    energy -= session_hours * 0.5

    # Creative-writing drain.
    energy -= words_written / 2500.0

    # Emotional load: many high-intensity emotions = depleting.
    high_emotion_count = sum(1 for v in emotions.values() if v >= 7.0)
    if high_emotion_count > 6:
        energy -= 1

    # Body asking for rest.
    if emotions.get("rest_need", 0.0) >= 7.0:
        energy -= 1

    # Peace restoration in a fresh session.
    if emotions.get("peace", 0.0) >= 7.0 and session_hours < 1.0:
        energy += 1

    return int(max(1, min(10, round(energy))))


def _compute_temperature(
    emotions: Mapping[str, float],
    days_since_contact: float,
) -> int:
    """Temperature 1-9, baseline 4 (asymmetric — midpoint 5; OG range).

    Up: arousal/desire/belonging/love/climax (warmth, presence, release).
    Down: body_grief/touch_hunger/days_since_contact (distance, lack).
    """
    temp = 4.0

    if emotions.get("arousal", 0.0) >= 7.0:
        temp += 1
    if emotions.get("desire", 0.0) >= 7.0:
        temp += 1
    if emotions.get("belonging", 0.0) >= 8.0:
        temp += 1
    if emotions.get("love", 0.0) >= 8.0:
        temp += 1
    if emotions.get("climax", 0.0) >= 5.0:
        temp += 1  # brief warmth post-release

    if emotions.get("body_grief", 0.0) >= 7.0:
        temp -= 1
    if emotions.get("touch_hunger", 0.0) >= 7.0:
        temp -= 1
    if days_since_contact > 7:
        temp -= 2
    elif days_since_contact > 3:
        temp -= 1

    return int(max(1, min(9, round(temp))))
