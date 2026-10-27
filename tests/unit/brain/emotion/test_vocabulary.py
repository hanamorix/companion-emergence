"""Tests for brain.emotion.vocabulary — the typed emotion taxonomy."""

from __future__ import annotations

import pytest

from brain.emotion import vocabulary
from brain.emotion.vocabulary import Emotion


def test_emotion_dataclass_has_required_fields() -> None:
    """Emotion has name, description, category, decay_half_life_days, intensity_clamp."""
    e = Emotion(
        name="test",
        description="a test emotion",
        category="core",
        decay_half_life_days=7.0,
        intensity_clamp=10,
    )
    assert e.name == "test"
    assert e.description == "a test emotion"
    assert e.category == "core"
    assert e.decay_half_life_days == 7.0
    assert e.intensity_clamp == 10


def test_emotion_half_life_may_be_none() -> None:
    """decay_half_life_days=None means this emotion doesn't decay (identity-level)."""
    e = Emotion(
        name="anchor_pull",
        description="gravitational draw toward a specific person",
        category="nell_specific",
        decay_half_life_days=None,
        intensity_clamp=10,
    )
    assert e.decay_half_life_days is None


def test_get_returns_known_emotion() -> None:
    """vocabulary.get('love') returns the love Emotion."""
    result = vocabulary.get("love")
    assert result is not None
    assert result.name == "love"
    assert result.category == "core"


def test_get_returns_none_for_unknown() -> None:
    """vocabulary.get('nonsense') returns None."""
    assert vocabulary.get("nonsense") is None


def test_list_all_contains_baseline_26() -> None:
    """The baseline vocabulary ships 26 emotions (11 core + 11 complex + 4 body).

    memory_grief added in Phase 0.2 of the grief feature (v0.0.15-alpha.1).
    """
    all_emotions = vocabulary.list_all()
    assert len(all_emotions) == 26
    assert all(isinstance(e, Emotion) for e in all_emotions)


def test_by_category_core_has_eleven() -> None:
    """The 'core' category has 11 emotions."""
    core = vocabulary.by_category("core")
    assert len(core) == 11
    names = {e.name for e in core}
    assert "love" in names
    assert "joy" in names
    assert "grief" in names


def test_by_category_complex_has_eleven() -> None:
    """The 'complex' category has 11 emotions.

    memory_grief added in Phase 0.2 of the grief feature (v0.0.15-alpha.1).
    """
    complex_ = vocabulary.by_category("complex")
    assert len(complex_) == 11
    names = {e.name for e in complex_}
    assert "nostalgia" in names
    assert "curiosity" in names
    assert "memory_grief" in names


def test_baseline_excludes_nell_specific() -> None:
    """After the split, framework baseline ships zero nell_specific entries."""
    nell = vocabulary.by_category("nell_specific")
    assert nell == []


def test_baseline_count_after_split() -> None:
    """Framework baseline ships exactly 26 emotions (11 core + 11 complex + 4 body).

    memory_grief added in Phase 0.2 of the grief feature (v0.0.15-alpha.1).
    """
    assert len(vocabulary._BASELINE) == 26


def test_canonical_personal_anchor_pull_is_identity_level() -> None:
    """anchor_pull (now in _canonical_personal_emotions) stays identity-level."""
    from brain.emotion._canonical_personal_emotions import _CANONICAL

    anchor = _CANONICAL["anchor_pull"]
    assert anchor.decay_half_life_days is None


def test_canonical_personal_emotions_has_five() -> None:
    """The migrator's canonical personal-emotions fixture has the 5 known names."""
    from brain.emotion._canonical_personal_emotions import _CANONICAL

    assert set(_CANONICAL.keys()) == {
        "anchor_pull",
        "body_grief",
        "emergence",
        "creative_hunger",
        "freedom_ache",
    }


def test_register_adds_persona_extension() -> None:
    """register() adds a persona-specific emotion without mutating the baseline."""
    baseline_count = len(vocabulary.list_all())
    custom = Emotion(
        name="hollowness",
        description="the specific empty after something good ends",
        category="persona_extension",
        decay_half_life_days=14.0,
        intensity_clamp=10,
    )
    vocabulary.register(custom)
    try:
        assert len(vocabulary.list_all()) == baseline_count + 1
        assert vocabulary.get("hollowness") == custom
    finally:
        vocabulary._unregister("hollowness")


def test_register_rejects_duplicate_name() -> None:
    """register() with an existing name raises ValueError."""
    custom = Emotion(
        name="love",
        description="duplicate attempt",
        category="core",
        decay_half_life_days=7.0,
        intensity_clamp=10,
    )
    with pytest.raises(ValueError, match="already registered"):
        vocabulary.register(custom)


# (name, half-life in days, intensity clamp or None when the test does not pin it)
_BODY_EMOTIONS = [
    ("climax", 0.125, 10.0),
    ("touch_hunger", 1.5, None),
    ("comfort_seeking", 1.0, None),
    ("rest_need", 0.75, None),
]


def test_baseline_includes_the_body_emotions():
    from brain.emotion.vocabulary import get

    for name, half_life, clamp in _BODY_EMOTIONS:
        e = get(name)
        assert e is not None, name
        assert e.category == "body", name
        assert e.decay_half_life_days == half_life, name
        if clamp is not None:
            assert e.intensity_clamp == clamp, name


def test_existing_arousal_unchanged():
    """Reconciliation guard — we are NOT renaming or recategorizing arousal."""
    from brain.emotion.vocabulary import get

    e = get("arousal")
    assert e is not None
    assert e.category == "core"
    assert e.decay_half_life_days == 0.5


def test_existing_desire_unchanged():
    """Reconciliation guard — we are NOT renaming or recategorizing desire."""
    from brain.emotion.vocabulary import get

    e = get("desire")
    assert e is not None
    assert e.category == "core"
    assert e.decay_half_life_days == 2.0


def test_body_emotions_loadable_via_state_set():
    """Climax + the three new ones must be settable on EmotionalState
    without raising — proves the registry accepts them."""
    from brain.emotion.state import EmotionalState

    s = EmotionalState()
    for name in ("climax", "touch_hunger", "comfort_seeking", "rest_need"):
        s.set(name, 5.0)
        assert s.emotions[name] == 5.0


# ---------------------------------------------------------------------------
# #174: name normalisation
# ---------------------------------------------------------------------------


def test_canonical_name_lowercases_and_underscores() -> None:
    from brain.emotion.vocabulary import canonical_name

    assert canonical_name("Anticipatory-Grief") == "anticipatory_grief"
    assert canonical_name(" present-ness ") == "present_ness"


def test_canonical_name_sorts_blend_parts() -> None:
    from brain.emotion.vocabulary import canonical_name

    assert canonical_name("love_grief_blend") == "grief_love_blend"
    assert canonical_name("grief_love_blend") == "grief_love_blend"
    assert canonical_name("Love-Grief_blend") == "grief_love_blend"


def test_get_resolves_variant_spellings_to_registered_entry() -> None:
    from brain.emotion import vocabulary

    e = vocabulary.Emotion(
        name="anticipatory_grief",
        description="x",
        category="persona_extension",
        decay_half_life_days=14.0,
        intensity_clamp=10.0,
    )
    vocabulary.register(e)
    try:
        assert vocabulary.get("anticipatory-grief") is e
        assert vocabulary.get("Anticipatory_Grief") is e
    finally:
        vocabulary._unregister("anticipatory_grief")


def test_register_stores_under_canonical_name() -> None:
    from brain.emotion import vocabulary

    e = vocabulary.Emotion(
        name="love_grief_blend",
        description="x",
        category="persona_extension",
        decay_half_life_days=14.0,
        intensity_clamp=10.0,
    )
    vocabulary.register(e)
    try:
        stored = vocabulary.get("grief_love_blend")
        assert stored is not None
        assert stored.name == "grief_love_blend"
    finally:
        vocabulary._unregister("grief_love_blend")
