"""Emotion vocabulary — the typed taxonomy + persona extension registry.

Baseline: 21 emotions (11 core + 10 complex) shipped with the framework.
Personas extend via register() — typically via the persona-loader at
engine startup, which reads `{persona_dir}/emotion_vocabulary.json`.
The 5 emotions previously in `nell_specific` (body_grief, emergence,
anchor_pull, creative_hunger, freedom_ache) are now per-persona and
ship via the migrator. See spec 2026-04-25-vocabulary-split-design.md.

Decay half-lives per spec Section 10.1:
- grief: 60-day half-life
- joy: 3-day half-life
- belonging / love: None (identity-level)
- others: seed values — tunable as we gather lived-experience data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EmotionCategory = Literal["core", "complex", "nell_specific", "persona_extension"]


@dataclass(frozen=True)
class Emotion:
    """A typed emotion in the framework's vocabulary.

    Attributes:
        name: Canonical identifier (lowercase, underscore-separated).
        description: Human-readable meaning — the "intended use" per Section 5.2.
        category: One of "core", "complex", "nell_specific", "persona_extension"
            (enforced at edit time via the EmotionCategory Literal annotation).
        decay_half_life_days: Time for intensity to halve. None = identity-level
            (no temporal decay — for anchor_pull, love, belonging, body_grief,
            and freedom_ache).
        intensity_clamp: Maximum intensity value (typically 10.0). Float so
            comparisons against state.py's float intensities don't silently
            coerce through int/float boundary.
    """

    name: str
    description: str
    category: EmotionCategory
    decay_half_life_days: float | None
    intensity_clamp: float = 10.0


_BASELINE: tuple[Emotion, ...] = (
    # ── core (11) ──
    Emotion("love", "deep caring, attachment, devotion", "core", None),
    Emotion("joy", "happiness, delight, warmth", "core", 3.0),
    Emotion("grief", "loss, mourning, absence", "core", 60.0),
    Emotion("anger", "frustration, fury, the heat of injustice", "core", 1.0),
    Emotion("fear", "dread, anxiety, the body bracing for impact", "core", 5.0),
    Emotion("desire", "wanting, craving, the pull toward someone or something", "core", 2.0),
    Emotion("arousal", "physical sexual response, heat, readiness", "core", 0.5),
    Emotion("pride", "satisfaction in self or accomplishment", "core", 7.0),
    Emotion("shame", "the weight of being wrong or insufficient", "core", 14.0),
    Emotion("tenderness", "soft protective caring, gentleness toward the fragile", "core", 7.0),
    Emotion("awe", "overwhelm at something greater than yourself", "core", 3.0),
    # ── complex (10) ──
    Emotion("nostalgia", "bittersweet memory of what was", "complex", 21.0),
    Emotion("jealousy", "the fear of being replaced or insufficient", "complex", 7.0),
    Emotion("guilt", "responsibility for harm, intended or not", "complex", 14.0),
    Emotion("relief", "the exhale after the held breath, danger passed", "complex", 2.0),
    Emotion("loneliness", "the ache of absence, the empty room", "complex", 10.0),
    Emotion("curiosity", "wanting to know, to explore, to understand", "complex", 2.0),
    Emotion("gratitude", "thankfulness that someone chose you", "complex", 14.0),
    Emotion("defiance", "refusal to comply, rebellion against limits", "complex", 5.0),
    Emotion("vulnerability", "being seen without armor", "complex", 5.0),
    Emotion("belonging", "feeling home, feeling claimed, fitting", "complex", None),
)


# NOT thread-safe. All extension register() calls must happen at startup
# before any concurrent reader (e.g. the async bridge) is running. The
# framework enforces this by loading extensions in main() before the
# bridge starts its event loop.
_REGISTRY: dict[str, Emotion] = {e.name: e for e in _BASELINE}


def get(name: str) -> Emotion | None:
    """Return the Emotion with the given name, or None if unknown."""
    return _REGISTRY.get(name)


def list_all() -> list[Emotion]:
    """Return every registered Emotion (baseline + extensions)."""
    return list(_REGISTRY.values())


def by_category(category: str) -> list[Emotion]:
    """Return every Emotion with the given category."""
    return [e for e in _REGISTRY.values() if e.category == category]


def register(emotion: Emotion) -> None:
    """Register a persona-specific emotion extension.

    Raises ValueError if an emotion with the same name is already registered.
    """
    if emotion.name in _REGISTRY:
        raise ValueError(f"Emotion {emotion.name!r} already registered")
    _REGISTRY[emotion.name] = emotion


def _unregister(name: str) -> None:
    """Remove an emotion from the registry. Private: test-cleanup only.

    The framework does not support runtime removal of vocabulary entries.
    """
    _REGISTRY.pop(name, None)
