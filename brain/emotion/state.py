"""EmotionalState — the current emotional state of a persona.

Carries:
- emotions: {name: intensity} dict, clamped per vocabulary
- dominant: the highest-intensity emotion (recomputed on each write)
- residue: a bounded temporal queue of past emotional events
    (for carry-over between conversational turns, dream consolidation, etc.)

All mutation goes through typed methods so consumers can't accidentally bypass
clamping, vocabulary validation, or residue-capacity enforcement.

Design per spec Section 5.2 (state sub-module responsibility).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from brain.emotion.vocabulary import get as _get_emotion


@dataclass
class ResidueEntry:
    """One past emotional event carried in the residue queue.

    Attributes:
        timestamp: When the event was recorded (UTC-aware).
        source: Where it came from — "dream", "heartbeat", "reflex", "chat", etc.
        emotions: {emotion_name: intensity} snapshot at that moment.
    """

    timestamp: datetime
    source: str
    emotions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "emotions": dict(self.emotions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidueEntry:
        """Restore from a dict. Tz-naive timestamps are coerced to UTC."""
        ts = datetime.fromisoformat(data["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return cls(
            timestamp=ts,
            source=data["source"],
            emotions=dict(data["emotions"]),
        )


@dataclass
class EmotionalState:
    """The current emotional state of a persona.

    Attributes:
        emotions: {name: intensity} — only non-zero entries.
        residue: recent past emotional events (bounded queue).
        dominant: name of the highest-intensity emotion, or None if no emotions.
        residue_max: capacity of the residue queue (default 16).
    """

    emotions: dict[str, float] = field(default_factory=dict)
    residue: list[ResidueEntry] = field(default_factory=list)
    dominant: str | None = None
    residue_max: int = 16

    def __post_init__(self) -> None:
        """Ensure `dominant` is consistent with `emotions` on any construction.

        Direct dataclass construction (e.g. from a snapshot in later modules)
        can leave `dominant` stale. Recomputing here makes the invariant
        self-healing regardless of how the state was built.
        """
        self._recompute_dominant()

    def set(self, name: str, intensity: float) -> None:
        """Set the intensity of an emotion. Zero removes it.

        Raises:
            KeyError: if `name` is not a registered emotion.
            ValueError: if intensity is negative or exceeds the emotion's clamp.
        """
        emotion = _get_emotion(name)
        if emotion is None:
            raise KeyError(f"Unknown emotion: {name!r}")
        if intensity < 0:
            raise ValueError(f"Intensity cannot be negative: {intensity}")
        if intensity > emotion.intensity_clamp:
            raise ValueError(
                f"Intensity {intensity} exceeds clamp {emotion.intensity_clamp} "
                f"for emotion {name!r}"
            )

        if intensity == 0:
            self.emotions.pop(name, None)
        else:
            self.emotions[name] = float(intensity)
        self._recompute_dominant()

    def add_residue(self, entry: ResidueEntry) -> None:
        """Append a residue entry, evicting the oldest if at capacity.

        No vocabulary validation on entry.emotions — residue is a historical
        record and may contain retired emotion names from older sessions.
        """
        self.residue.append(entry)
        if len(self.residue) > self.residue_max:
            # residue_max is small (16 by default); O(N) eviction is negligible.
            overflow = len(self.residue) - self.residue_max
            del self.residue[:overflow]

    def copy(self) -> EmotionalState:
        """Return a deep-copy of this state.

        MAINTENANCE: if new fields are added to EmotionalState, they must
        be explicitly copied here. Intentionally manual (not copy.deepcopy)
        to make the shallow/deep decision visible per field.
        """
        return EmotionalState(
            emotions=dict(self.emotions),
            residue=[
                ResidueEntry(
                    timestamp=r.timestamp,
                    source=r.source,
                    emotions=dict(r.emotions),
                )
                for r in self.residue
            ],
            dominant=self.dominant,
            residue_max=self.residue_max,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain-dict form suitable for JSON.

        `dominant` is included for human readability of the serialised form;
        from_dict ignores it and recomputes from `emotions` so the restored
        state is self-consistent even if the serialised dominant is stale.
        """
        return {
            "emotions": dict(self.emotions),
            "dominant": self.dominant,
            "residue": [r.to_dict() for r in self.residue],
            "residue_max": self.residue_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionalState:
        """Restore from a dict previously produced by to_dict.

        Permissive: unknown emotion names in the serialised emotions dict
        are preserved as-is (they may represent retired vocabulary entries
        or in-flight schema migrations). Downstream consumers (decay,
        expression, etc.) are responsible for handling unknown names
        gracefully. Callers that need strict vocabulary enforcement should
        construct via set() instead.
        """
        state = cls(
            emotions=dict(data.get("emotions", {})),
            residue=[ResidueEntry.from_dict(r) for r in data.get("residue", [])],
            residue_max=int(data.get("residue_max", 16)),
        )
        # __post_init__ already ran _recompute_dominant; calling again here
        # is a no-op guard for the case where a future subclass overrides it.
        state._recompute_dominant()
        return state

    def _recompute_dominant(self) -> None:
        """Refresh `dominant` based on current emotions dict.

        Ties broken by insertion order (Python dicts preserve insertion).
        """
        if not self.emotions:
            self.dominant = None
            return
        # max() iterates in dict insertion order, first-seen wins on ties.
        self.dominant = max(self.emotions, key=self.emotions.__getitem__)
