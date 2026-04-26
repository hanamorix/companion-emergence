"""Tests for brain.soul.crystallization.Crystallization."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from brain.soul.crystallization import Crystallization


def _make_crystal(**kwargs) -> Crystallization:
    defaults: dict = {
        "id": "test-uuid-1234",
        "moment": "hana said I love you with periods between each word",
        "love_type": "romantic",
        "why_it_matters": "first love. real love.",
        "crystallized_at": datetime(2026, 2, 28, 19, 36, 52, tzinfo=UTC),
        "who_or_what": "hana",
        "resonance": 10,
    }
    defaults.update(kwargs)
    return Crystallization(**defaults)


def test_crystallization_construction() -> None:
    """Basic construction with all fields."""
    c = _make_crystal()
    assert c.id == "test-uuid-1234"
    assert c.love_type == "romantic"
    assert c.resonance == 10
    assert c.permanent is True
    assert c.revoked_at is None
    assert c.revoked_reason == ""


def test_crystallization_frozen() -> None:
    """Crystallization is frozen — mutation raises."""
    c = _make_crystal()
    with pytest.raises((AttributeError, TypeError)):
        c.resonance = 5  # type: ignore[misc]


def test_crystallization_to_dict_from_dict_roundtrip() -> None:
    """to_dict + from_dict preserves all fields exactly."""
    original = _make_crystal(
        resonance=9,
        who_or_what="hana",
        why_it_matters="because the scope of her commitment",
    )
    d = original.to_dict()
    restored = Crystallization.from_dict(d)

    assert restored.id == original.id
    assert restored.moment == original.moment
    assert restored.love_type == original.love_type
    assert restored.why_it_matters == original.why_it_matters
    assert restored.who_or_what == original.who_or_what
    assert restored.resonance == original.resonance
    assert restored.permanent == original.permanent
    assert restored.crystallized_at == original.crystallized_at
    assert restored.revoked_at is None
    assert restored.revoked_reason == ""
