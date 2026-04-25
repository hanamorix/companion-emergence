"""Tests for brain.migrator.og_vocabulary."""

from __future__ import annotations

from brain.migrator.og_vocabulary import extract_persona_vocabulary


def test_extract_subtracts_framework_baseline():
    """Memories with only baseline emotions → empty result."""
    memories = [
        {"emotions": {"love": 5.0, "joy": 3.0}},
        {"emotions": {"grief": 2.0}},
    ]
    result = extract_persona_vocabulary(memories, framework_baseline_names={"love", "joy", "grief"})
    assert result == []


def test_extract_canonical_nell_specific():
    """Memory with body_grief → canonical entry with proper description + decay."""
    memories = [{"emotions": {"body_grief": 6.0}}]
    result = extract_persona_vocabulary(memories, framework_baseline_names={"love"})
    assert len(result) == 1
    entry = result[0]
    assert entry["name"] == "body_grief"
    assert entry["category"] == "persona_extension"
    assert entry["decay_half_life_days"] is None  # identity-level
    assert "physical form" in entry["description"]


def test_extract_unknown_emotion_uses_placeholder():
    """Memory with custom emotion → placeholder description + 14.0 decay default."""
    memories = [{"emotions": {"melancholy_blue": 4.0}}]
    result = extract_persona_vocabulary(memories, framework_baseline_names={"love"})
    assert len(result) == 1
    entry = result[0]
    assert entry["name"] == "melancholy_blue"
    assert entry["category"] == "persona_extension"
    assert entry["decay_half_life_days"] == 14.0
    assert "migrated from OG" in entry["description"]


def test_extract_sorted_deterministic():
    """Result sorted by name for diff-friendly output."""
    memories = [
        {"emotions": {"freedom_ache": 5.0, "anchor_pull": 6.0, "body_grief": 7.0}},
    ]
    result = extract_persona_vocabulary(memories, framework_baseline_names=set())
    names = [e["name"] for e in result]
    assert names == sorted(names)


def test_extract_empty_memories():
    """Empty input → empty result."""
    result = extract_persona_vocabulary([], framework_baseline_names={"love"})
    assert result == []


def test_extract_memory_without_emotions_dict():
    """Memory without 'emotions' key → silently skipped (defensive)."""
    memories = [{"content": "no emotions"}, {"emotions": {"body_grief": 5.0}}]
    result = extract_persona_vocabulary(memories, framework_baseline_names=set())
    assert len(result) == 1
    assert result[0]["name"] == "body_grief"
