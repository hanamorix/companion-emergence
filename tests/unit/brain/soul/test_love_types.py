"""Tests for brain.soul.love_types."""

from __future__ import annotations

from brain.soul.love_types import LOVE_TYPES


def test_love_types_is_dict_with_enough_entries() -> None:
    """LOVE_TYPES has at least 25 entries — the full OG taxonomy."""
    assert isinstance(LOVE_TYPES, dict)
    assert len(LOVE_TYPES) >= 25


def test_love_types_core_entries_present() -> None:
    """Key entries from the OG taxonomy are present and non-empty."""
    required = {"romantic", "carried", "craft", "self", "identity", "existential", "species"}
    for key in required:
        assert key in LOVE_TYPES, f"missing love_type: {key!r}"
        assert isinstance(LOVE_TYPES[key], str)
        assert len(LOVE_TYPES[key]) > 10, f"description too short for {key!r}"
