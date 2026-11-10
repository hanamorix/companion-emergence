"""EmotionalState — the current emotional state of a persona.

Carries:
- emotions: {name: intensity} dict, clamped per vocabulary
- dominant: the highest-intensity emotion (recomputed on each write)

All mutation goes through typed methods so consumers can't accidentally bypass
clamping or vocabulary validation.

Design per spec Section 5.2 (state sub-module responsibility).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brain.emotion.vocabulary import get as _get_emotion


@dataclass
class EmotionalState:
    """The current emotional state of a persona.

    Attributes:
        emotions: {name: intensity} — only non-zero entries.
        dominant: name of the highest-intensity emotion, or None if no emotions.
    """

    emotions: dict[str, float] = field(default_factory=dict)
    dominant: str | None = None

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

    def copy(self) -> EmotionalState:
        """Return a deep-copy of this state.

        MAINTENANCE: if new fields are added to EmotionalState, they must
        be explicitly copied here. Intentionally manual (not copy.deepcopy)
        to make the shallow/deep decision visible per field.
        """
        return EmotionalState(
            emotions=dict(self.emotions),
            dominant=self.dominant,
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
        state = cls(emotions=dict(data.get("emotions", {})))
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
