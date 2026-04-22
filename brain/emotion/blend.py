"""Emergent blend detection.

Observes emotional states over time. When two or more emotions co-occur at
high intensity repeatedly, the detector records them as a named blend.
Names can be assigned later (once the shape is recognised).

Design per spec Section 5.2 (blend sub-module). Threshold tunable — current
defaults: intensity ≥5 each, ≥5 co-occurrences to register.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from brain.emotion.state import EmotionalState


@dataclass
class DetectedBlend:
    """A repeatedly-observed co-occurrence of high-intensity emotions.

    Attributes:
        components: Tuple of emotion names (sorted alphabetically for stable hashing).
        count: How many times this combination has been observed above threshold.
        name: Optional human-readable label, assigned via BlendDetector.name_blend().
    """

    components: tuple[str, ...]
    count: int
    name: str | None = None


@dataclass
class BlendDetector:
    """Tracks emotional co-occurrences to surface emergent patterns.

    Attributes:
        intensity_threshold: Minimum per-emotion intensity to count an observation.
        detection_threshold: Minimum observations to register the pattern.
        _observations: Internal count map (components tuple → count).
        _names: Internal name map (components tuple → name).
    """

    intensity_threshold: float = 5.0
    detection_threshold: int = 5
    _observations: dict[tuple[str, ...], int] = field(default_factory=dict)
    _names: dict[tuple[str, ...], str] = field(default_factory=dict)

    def observe(self, state: EmotionalState) -> None:
        """Record the high-intensity emotion combinations from the given state."""
        high = tuple(
            sorted(
                name
                for name, intensity in state.emotions.items()
                if intensity >= self.intensity_threshold
            )
        )
        if len(high) < 2:
            return

        # Track every pair and every triple. We bound subset size at 3 so the
        # combinatorics stay manageable even for rich states.
        for size in (2, 3):
            if size > len(high):
                break
            for combo in combinations(high, size):
                self._observations[combo] = self._observations.get(combo, 0) + 1

    def detected(self) -> list[DetectedBlend]:
        """Return every combination that has crossed the detection threshold."""
        result = []
        for components, count in self._observations.items():
            if count >= self.detection_threshold:
                result.append(
                    DetectedBlend(
                        components=components,
                        count=count,
                        name=self._names.get(components),
                    )
                )
        return result

    def name_blend(self, components: Iterable[str], name: str) -> None:
        """Assign a human-readable name to a previously-detected blend.

        Raises:
            KeyError: if the given components haven't been detected yet.
        """
        key = tuple(sorted(components))
        if key not in self._observations or self._observations[key] < self.detection_threshold:
            raise KeyError(f"Blend {key!r} has not been detected yet")
        self._names[key] = name

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain-dict form suitable for JSON."""
        return {
            "intensity_threshold": self.intensity_threshold,
            "detection_threshold": self.detection_threshold,
            "observations": [
                {"components": list(k), "count": v} for k, v in self._observations.items()
            ],
            "names": [{"components": list(k), "name": v} for k, v in self._names.items()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlendDetector:
        """Restore from a dict produced by to_dict."""
        detector = cls(
            intensity_threshold=float(data.get("intensity_threshold", 5.0)),
            detection_threshold=int(data.get("detection_threshold", 5)),
        )
        for entry in data.get("observations", []):
            key = tuple(entry["components"])
            detector._observations[key] = int(entry["count"])
        for entry in data.get("names", []):
            key = tuple(entry["components"])
            detector._names[key] = entry["name"]
        return detector
