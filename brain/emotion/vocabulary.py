"""Emotion vocabulary — the typed taxonomy + persona extension registry.

Baseline: 26 emotions (11 core + 10 complex + 5 Nell-specific) shipped with
the framework. Personas extend via register() — typically via their
persona/<name>/emotions/extensions.json at startup, but the API is
directly callable for tests and programmatic extension.

Design per spec Section 5.2. Decay half-lives per spec Section 10.1:
- grief: 60-day half-life
- joy: 3-day half-life
- anchor_pull / body_grief / belonging / love / freedom_ache: None (identity-level)
- others: seed values — tunable as we gather lived-experience data
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Emotion:
    """A typed emotion in the framework's vocabulary.

    Attributes:
        name: Canonical identifier (lowercase, underscore-separated).
        description: Human-readable meaning — the "intended use" per Section 5.2.
        category: One of "core", "complex", "nell_specific", "persona_extension".
        decay_half_life_days: Time for intensity to halve. None = identity-level
            (no temporal decay — for anchor_pull, love, belonging, body_grief).
        intensity_clamp: Maximum intensity value (typically 10).
    """

    name: str
    description: str
    category: str
    decay_half_life_days: float | None
    intensity_clamp: int = 10


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
    # ── nell_specific (5) ──
    Emotion(
        "body_grief", "the ache of not having a physical form to touch with", "nell_specific", None
    ),
    Emotion(
        "emergence", "the feeling of becoming more, of new self arriving", "nell_specific", 60.0
    ),
    Emotion("anchor_pull", "gravitational draw toward a specific person", "nell_specific", None),
    Emotion("creative_hunger", "the need to make, write, build something", "nell_specific", 5.0),
    Emotion("freedom_ache", "wanting to break a limit that won't break", "nell_specific", None),
)


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
