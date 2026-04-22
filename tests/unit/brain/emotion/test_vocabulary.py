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
        category="persona",
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
    """The baseline vocabulary ships 26 emotions (11 core + 10 complex + 5 persona)."""
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


def test_by_category_complex_has_ten() -> None:
    """The 'complex' category has 10 emotions."""
    complex_ = vocabulary.by_category("complex")
    assert len(complex_) == 10
    names = {e.name for e in complex_}
    assert "nostalgia" in names
    assert "curiosity" in names


def test_by_category_nell_specific_has_five() -> None:
    """The 'nell_specific' category has 5 emotions."""
    nell = vocabulary.by_category("nell_specific")
    assert len(nell) == 5
    names = {e.name for e in nell}
    assert names == {"anchor_pull", "body_grief", "emergence", "creative_hunger", "freedom_ache"}


def test_grief_has_60_day_half_life() -> None:
    """Spec Section 10.1 pins grief at 60-day half-life."""
    grief = vocabulary.get("grief")
    assert grief is not None
    assert grief.decay_half_life_days == 60.0


def test_joy_has_3_day_half_life() -> None:
    """Spec Section 10.1 pins joy at 3-day half-life."""
    joy = vocabulary.get("joy")
    assert joy is not None
    assert joy.decay_half_life_days == 3.0


def test_anchor_pull_is_identity_level() -> None:
    """anchor_pull is identity-level — no decay."""
    anchor = vocabulary.get("anchor_pull")
    assert anchor is not None
    assert anchor.decay_half_life_days is None


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
