"""Emotion vocabulary — the typed taxonomy + persona extension registry.

Baseline: 30 emotions (12 core + 14 complex + 4 body) shipped with the framework.
Personas extend via register() — typically via the persona-loader at
engine startup, which reads `{persona_dir}/emotion_vocabulary.json`.
The 4 emotions previously in `nell_specific` that the migrator still ships
(emergence, anchor_pull, creative_hunger, freedom_ache) are now per-persona. See spec 2026-04-25-vocabulary-split-design.md.

Decay half-lives per spec Section 10.1:
- grief: 60-day half-life
- joy: 3-day half-life
- belonging / love: None (identity-level)
- others: seed values — tunable as we gather lived-experience data
- contentment, peace, calm, groundedness: the half-lives are not written here; they come from
  `dev_constants.EMOTION_HALF_LIFE_DAYS` (one source for these four, #648)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from brain.dev_constants import EMOTION_HALF_LIFE_DAYS, EMOTION_NAME_MAX_CHARS
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

# `nell_specific` retained for backwards-compat — pre-vocabulary-split persona
# files in the wild may still contain entries with this category. New baseline
# entries use only "core" or "complex"; new persona-loaded entries use only
# "persona_extension". The Literal accepts `nell_specific` so loaders don't
# reject older files outright.
EmotionCategory = Literal["core", "complex", "nell_specific", "persona_extension", "body"]


@dataclass(frozen=True)
class Emotion:
    """A typed emotion in the framework's vocabulary.

    Attributes:
        name: The entry's own spelling: the framework's names are tidied (lowercase, underscore-separated); a persona entry
            keeps the spelling in her vocabulary file. `canonical_name(name)` is the tidied form every lookup matches by.
        description: Human-readable meaning — the "intended use" per Section 5.2.
        category: One of "core", "complex", "nell_specific", "persona_extension"
            (enforced at edit time via the EmotionCategory Literal annotation).
        decay_half_life_days: Time for intensity to halve. None = identity-level
            (no temporal decay — for anchor_pull, love, belonging,
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
    # ── core (12) ──
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
    Emotion(
        "contentment",
        "the quiet pleasure of what is, safe, wanting nothing more",
        "core",
        EMOTION_HALF_LIFE_DAYS["contentment"],
    ),
    # ── complex (14) ──
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
    Emotion(
        "memory_grief",
        "the ache of having lost something I once knew",
        "complex",
        30.0,
    ),
    # peace, calm and groundedness are flavours of contentment (core). The vocabulary has no parent field, so the
    # relation is recorded here only (#648).
    Emotion(
        "peace",
        "settled and unhurried, nothing pressing",
        "complex",
        EMOTION_HALF_LIFE_DAYS["peace"],
    ),
    Emotion(
        "calm",
        "steady and unruffled, nothing stirred up",
        "complex",
        EMOTION_HALF_LIFE_DAYS["calm"],
    ),
    Emotion(
        "groundedness",
        "rooted and steady, sure of where you stand",
        "complex",
        EMOTION_HALF_LIFE_DAYS["groundedness"],
    ),
    # ── body (4) ──
    # Spec docs/superpowers/specs/2026-04-29-body-state-design.md §2.1.
    # Existing `arousal` (core, 0.5d) and `desire` (core, 2.0d) reused;
    # they're already body-coded by description, no need to duplicate.
    Emotion(
        "climax",
        "bodily completion / release — the satisfaction crest. Spikes briefly, "
        "decays fast. When aggregated >= 7 triggers reset hook (heavy dampen on "
        "arousal, partial dampen on desire, raises comfort_seeking + rest_need).",
        "body",
        0.125,
    ),
    Emotion(
        "touch_hunger",
        "embodied loneliness — when distance is the problem and presence is the "
        "cure. Distinct from body_grief (existential, not having a body) and from "
        "loneliness (social/emotional).",
        "body",
        1.5,
    ),
    Emotion(
        "comfort_seeking",
        "wanting to be held still, wrapped, anchored — the receive side of "
        "being-held. Distinct from vulnerability (exposure) and love (orientation).",
        "body",
        1.0,
    ),
    Emotion(
        "rest_need",
        "the body asking for slowness, low stimulation, recovery. Distinct from "
        "exhaustion (the computed state) — this is the *want*, not the condition.",
        "body",
        0.75,
    ),
)


# Keyed by the tidied form (`canonical_name`); each value is the entry with its own spelling in `.name`, so the key is the
# index from a tidied form to the entry and its spelling (#801 section 3).
# NOT thread-safe. All extension register() calls must happen at startup
# before any concurrent reader (e.g. the async bridge) is running. The
# framework enforces this by loading extensions in main() before the
# bridge starts its event loop.
_REGISTRY: dict[str, Emotion] = {e.name: e for e in _BASELINE}


_BASELINE_KEYS: frozenset[str] = frozenset(e.name for e in _BASELINE)

_BLEND_SUFFIX = "_blend"


_SEPARATOR_RUN = re.compile(r"[\W_]+")  # a run of characters that are not letters or digits (`str.isalnum`)


def canonical_name(name: str) -> str:
    """The one tidier of a feeling name (#174, #800, #801): collapse spelling variants of one feeling to a single key.

    Lowercase; every run of characters that are not letters or digits becomes one underscore (so ``name____name`` and
    ``name_name`` are the same); underscores are trimmed from both ends. A ``<a>_<b>_blend`` name (any number of parts) has its
    parts sorted, so ``love_grief_blend``, ``Grief Love Blend`` and ``love_grief__blend`` all give ``grief_love_blend``
    (a doubled, leading or trailing separator makes no empty part, because a run of separators is one underscore and the ends are
    trimmed: #800). Lowercasing comes first so that tidying a tidied name changes nothing.
    """
    key = _SEPARATOR_RUN.sub("_", name.lower()).strip("_")
    if key.endswith(_BLEND_SUFFIX):
        key = "_".join(sorted(key[: -len(_BLEND_SUFFIX)].split("_"))) + _BLEND_SUFFIX
    return key


def gate_name(raw: str) -> str | None:
    """The one gate of a feeling name (#801): the tidied name, or None when it must be refused.

    A name that is empty once tidied, or longer than `dev_constants.EMOTION_NAME_MAX_CHARS`, is refused with a throttled warning
    that says why (the length only, never the name).
    """
    name = canonical_name(raw)
    if not name:
        warn_throttled(logger, "vocabulary.gate_name.empty", "vocabulary: a feeling name is empty once tidied; refused", exc_info=False)
        return None
    if len(name) > EMOTION_NAME_MAX_CHARS:
        warn_throttled(
            logger,
            "vocabulary.gate_name.too_long",
            "vocabulary: a feeling name is %d characters once tidied, over the limit of %d; refused",
            len(name),
            EMOTION_NAME_MAX_CHARS,
            exc_info=False,
        )
        return None
    return name


def pool_by_canonical_name(values: Mapping[str, float]) -> dict[str, float]:
    """Key a name-to-value mapping by canonical name; spellings of one emotion keep the larger value.

    A max-per-emotion rule over spellings. Names outside the
    vocabulary are only normalised, so unrelated names stay separate keys. Returns a new dict.
    """
    pooled: dict[str, float] = {}
    for name, value in values.items():
        key = canonical_name(name)
        if key not in pooled or value > pooled[key]:
            pooled[key] = value
    return pooled


def get(name: str) -> Emotion | None:
    """Return the Emotion with the given name (any spelling variant), or None."""
    return _REGISTRY.get(canonical_name(name))


def list_all() -> list[Emotion]:
    """Return every registered Emotion (baseline + extensions)."""
    return list(_REGISTRY.values())


def by_category(category: str) -> list[Emotion]:
    """Return every Emotion with the given category."""
    return [e for e in _REGISTRY.values() if e.category == category]


def register(emotion: Emotion) -> None:
    """Register a persona-specific emotion extension, under the spelling it carries.

    The entry keeps its own spelling (`Emotion.name`, the spelling in her vocabulary file); it is stored under the tidied form
    of that name, so `get` finds it by any spelling that tidies the same (#801 section 3).

    Raises ValueError if an emotion whose name tidies the same is already registered.
    """
    key = canonical_name(emotion.name)
    if key in _REGISTRY:
        raise ValueError(f"Emotion {emotion.name!r} already registered")
    _REGISTRY[key] = emotion


def is_baseline(name: str) -> bool:
    """True when the name (any spelling) is one of the framework's own 30 entries."""
    return canonical_name(name) in _BASELINE_KEYS


def replace_entry(emotion: Emotion) -> None:
    """Put `emotion` in the registry in place of the entry whose name tidies the same (the vocabulary file's own change path).

    Raises ValueError for a framework (baseline) name: those entries are never replaced from a file. Registers the entry
    when none carries that name yet.
    """
    if is_baseline(emotion.name):
        raise ValueError(f"Emotion {emotion.name!r} is a framework entry and is never replaced")
    _REGISTRY[canonical_name(emotion.name)] = emotion


def _unregister(name: str) -> None:
    """Remove an emotion from the registry, by any spelling. Private: the vocabulary file module's `remove` and test cleanup.

    The caller keeps framework (baseline) names registered (`is_baseline`).
    """
    _REGISTRY.pop(canonical_name(name), None)
