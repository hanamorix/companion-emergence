"""Body state — pure-function projection over emotions + session inputs.

The brain has a body. Body emotions live in the emotion vocabulary
(arousal, desire, climax, touch_hunger, comfort_seeking, rest_need).
The *projections* (energy, temperature, exhaustion) are computed
fresh on each call — no persistence, no cache, no parallel state.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.1.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import brain.dev_constants as dev_constants  # pure data, listed in the purity guard's _PURE_BRAIN_MODULES

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

    `emotions` MUST be the already pooled, post-climax-reset state the old
    emotion path hands it. compute_body_state does not re-pool or re-reset.

    This is the projection for the old emotion path: it fires on fixed levels (the only place that holds them).
    `body_state_from_short` is the one for the emotion engine.
    """
    energy = _energy_from_flags(
        session_hours,
        words_written,
        many_high=sum(1 for v in emotions.values() if v >= 7.0) > dev_constants.BODY_ENERGY_HIGH_COUNT,
        rest_need_high=emotions.get("rest_need", 0.0) >= 7.0,
        peace_high=emotions.get("peace", 0.0) >= 7.0,
    )
    ups = (
        (emotions.get("arousal", 0.0) >= 7.0)
        + (emotions.get("desire", 0.0) >= 7.0)
        + (emotions.get("belonging", 0.0) >= 8.0)
        + (emotions.get("love", 0.0) >= 8.0)
        + (emotions.get("climax", 0.0) >= 5.0)
    )
    downs = (emotions.get("body_grief", 0.0) >= 7.0) + (emotions.get("touch_hunger", 0.0) >= 7.0)
    temperature = _temperature_from_flags(days_since_contact, ups=ups, downs=downs)
    return _assemble(
        energy=energy,
        temperature=temperature,
        session_hours=session_hours,
        days_since_contact=days_since_contact,
        body_emotions={name: float(emotions.get(name, 0.0)) for name in BODY_EMOTION_NAMES},
        now=now,
    )


def body_state_from_short(
    short: Mapping[str, float],
    *,
    feels_high: Callable[[str], bool],
    session_hours: float,
    words_written: int,
    days_since_contact: float,
    now: datetime,
) -> BodyState:
    """The body state from the engine's short averages (`{name: short}` taken from `engine.read()`).

    Same projection as `compute_body_state` (the session, words and contact-gap rules are shared code), but the emotion
    conditions ask `feels_high(name)` (`engine.feels_high`) instead of comparing to fixed levels, and a body emotion the
    engine did not return is absent from `body_emotions` instead of filled with 0.0. A present 0.0 is kept. The caller
    passes the one `short` snapshot it read; the predicate asks the engine again for each name it is given, so it can
    disagree with that snapshot if the engine changes between the two reads.
    """
    energy = _energy_from_flags(
        session_hours,
        words_written,
        many_high=sum(1 for name in short if feels_high(name)) > dev_constants.BODY_ENERGY_HIGH_COUNT,
        rest_need_high=feels_high("rest_need"),
        peace_high=feels_high("peace"),
    )
    ups = sum(feels_high(name) for name in ("arousal", "desire", "belonging", "love", "climax"))
    downs = sum(feels_high(name) for name in ("body_grief", "touch_hunger"))
    temperature = _temperature_from_flags(days_since_contact, ups=ups, downs=downs)
    return _assemble(
        energy=energy,
        temperature=temperature,
        session_hours=session_hours,
        days_since_contact=days_since_contact,
        body_emotions={name: float(short[name]) for name in BODY_EMOTION_NAMES if name in short},
        now=now,
    )


def _assemble(
    *,
    energy: int,
    temperature: int,
    session_hours: float,
    days_since_contact: float,
    body_emotions: dict[str, float],
    now: datetime,
) -> BodyState:
    return BodyState(
        energy=energy,
        temperature=temperature,
        exhaustion=max(0, 7 - energy),
        session_hours=round(session_hours, 2),
        days_since_contact=round(days_since_contact, 2),
        body_emotions=body_emotions,
        computed_at=now,
    )


def _energy_from_flags(
    session_hours: float,
    words_written: int,
    *,
    many_high: bool,
    rest_need_high: bool,
    peace_high: bool,
) -> int:
    """Energy 1-10, baseline 8. Drains for session length + creative work +
    high emotional load + body asking for rest. Restores when peace is
    high in a fresh session. Spec §3.1.

    The three emotion conditions arrive as flags: the caller decides what "high" means.
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
    if many_high:
        energy -= 1

    # Body asking for rest.
    if rest_need_high:
        energy -= 1

    # Peace restoration in a fresh session.
    if peace_high and session_hours < 1.0:
        energy += 1

    return int(max(1, min(10, round(energy))))


def _temperature_from_flags(days_since_contact: float, *, ups: int, downs: int) -> int:
    """Temperature 1-9, baseline 4 (asymmetric — midpoint 5; OG range).

    Up: one for each of arousal/desire/belonging/love/climax that is high (warmth, presence, release): `ups`.
    Down: one for each of body_grief/touch_hunger that is high (distance, lack): `downs`; and the days since contact.
    """
    temp = 4.0 + ups - downs

    if days_since_contact > 7:
        temp -= 2
    elif days_since_contact > 3:
        temp -= 1

    return int(max(1, min(9, round(temp))))
