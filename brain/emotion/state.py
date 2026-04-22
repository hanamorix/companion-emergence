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
from datetime import datetime
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
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
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
        """Append a residue entry, evicting the oldest if at capacity."""
        self.residue.append(entry)
        if len(self.residue) > self.residue_max:
            # residue_max is small (16 by default); O(N) eviction is negligible.
            overflow = len(self.residue) - self.residue_max
            del self.residue[:overflow]

    def copy(self) -> EmotionalState:
        """Return a deep-copy of this state."""
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
        """Serialise to a plain-dict form suitable for JSON."""
        return {
            "emotions": dict(self.emotions),
            "dominant": self.dominant,
            "residue": [r.to_dict() for r in self.residue],
            "residue_max": self.residue_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionalState:
        """Restore from a dict previously produced by to_dict."""
        state = cls(
            emotions=dict(data.get("emotions", {})),
            residue=[ResidueEntry.from_dict(r) for r in data.get("residue", [])],
            residue_max=int(data.get("residue_max", 16)),
        )
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
