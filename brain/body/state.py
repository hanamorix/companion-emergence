"""Body state — pure-function projection over emotions + session inputs.

The brain has a body. Body emotions live in the emotion vocabulary
(arousal, desire, climax, touch_hunger, comfort_seeking, rest_need).
The temperature is computed fresh on each call, with no persistence, no cache and no parallel
state. The energy is not computed here: it is the level felt time carries from one heartbeat to the next
(`brain.felt_time.state.read_energy_level`), and each caller passes it in.

Spec: docs/source-spec/2026-04-29-body-state-design.md §3.1 (temperature); the energy is #650.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# The 6-emotion body set tracked by body_state_from_short.
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

    Energy 1-10 (the carried level, a float, never rounded here), temperature 1-9 (asymmetric, midpoint 5; OG
    inheritance). Energy is the only level shown.

    body_emotions carries the six body-class emotions as a snapshot —
    callers shouldn't have to read them again from the engine.
    """

    energy: float
    temperature: int
    session_hours: float
    days_since_contact: float
    body_emotions: dict[str, float]
    computed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the get_body_state tool result.

        `loaded: True` distinguishes the real impl from the legacy stub —
        the brain knows the body module is real when she calls the tool. The energy is rounded to one decimal
        here, for display only (as `body_emotions` are below); the level itself stays unrounded.
        """
        return {
            "loaded": True,
            "energy": round(self.energy, 1),
            "temperature": self.temperature,
            "session_hours": round(self.session_hours, 2),
            "days_since_contact": round(self.days_since_contact, 2),
            "body_emotions": {k: round(v, 1) for k, v in self.body_emotions.items()},
            "computed_at": self.computed_at.isoformat(),
        }


def body_state_from_short(
    short: Mapping[str, float],
    *,
    feels_high: Callable[[str], bool],
    session_hours: float,
    energy: float,
    days_since_contact: float,
    now: datetime,
) -> BodyState:
    """The body state from the engine's short averages (`{name: short}` taken from `engine.read()`).

    Pure projection: no I/O, no LLM call, no cache. `energy` is the saved level the caller read first (felt time's
    `read_energy_level`); it is kept as given, never rounded or recomputed here. The temperature conditions ask
    `feels_high(name)` (`engine.feels_high`), so no emotion level is compared here. A body emotion the engine did not
    return is absent from `body_emotions` (not filled with 0.0); a present 0.0 is kept. The caller passes the one
    `short` snapshot it read; the predicate asks the engine again for each name it is given, so it can disagree with
    that snapshot if the engine changes between the two reads.
    """
    ups = sum(feels_high(name) for name in ("arousal", "desire", "belonging", "love", "climax"))
    downs = int(feels_high("touch_hunger"))
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
    energy: float,
    temperature: int,
    session_hours: float,
    days_since_contact: float,
    body_emotions: dict[str, float],
    now: datetime,
) -> BodyState:
    return BodyState(
        energy=energy,
        temperature=temperature,
        session_hours=round(session_hours, 2),
        days_since_contact=round(days_since_contact, 2),
        body_emotions=body_emotions,
        computed_at=now,
    )


def _temperature_from_flags(days_since_contact: float, *, ups: int, downs: int) -> int:
    """Temperature 1-9, baseline 4 (asymmetric — midpoint 5; OG range).

    Up: one for each of arousal/desire/belonging/love/climax that is high (warmth, presence, release): `ups`.
    Down: one when touch_hunger is high (distance, lack): `downs`; and the days since contact.
    """
    temp = 4.0 + ups - downs

    if days_since_contact > 7:
        temp -= 2
    elif days_since_contact > 3:
        temp -= 1

    return int(max(1, min(9, round(temp))))
