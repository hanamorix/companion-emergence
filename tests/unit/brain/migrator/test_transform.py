"""Tests for brain.migrator.transform — OG memory dict → Memory dataclass."""

from __future__ import annotations

from typing import Any

from brain.migrator.transform import (
    SkippedMemory,
    transform_memory,
)


def _og(
    *,
    mem_id: str = "m1",
    content: str = "hello",
    **overrides: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": mem_id,
        "content": content,
        "memory_type": "conversation",
        "domain": "us",
        "created_at": "2024-01-01T12:00:00+00:00",
        "emotions": {"love": 9.0},
        "tags": ["a"],
        "importance": 5.0,
        "emotion_score": 9.0,
        "active": True,
    }
    base.update(overrides)
    return base


def test_transform_happy_path_produces_memory() -> None:
    """A well-formed OG memory transforms cleanly."""
    mem, skipped = transform_memory(_og())
    assert skipped is None
    assert mem is not None
    assert mem.id == "m1"
    assert mem.content == "hello"
    assert mem.emotions == {"love": 9.0}
    assert mem.score == 9.0


def test_transform_preserves_tz_aware_created_at() -> None:
    """created_at with tz-offset is preserved as UTC-aware."""
    mem, _ = transform_memory(_og(created_at="2024-03-01T15:00:00+00:00"))
    assert mem is not None
    assert mem.created_at.tzinfo is not None


def test_transform_coerces_tz_naive_created_at() -> None:
    """Naive created_at is coerced to UTC."""
    mem, _ = transform_memory(_og(created_at="2024-03-01T15:00:00"))
    assert mem is not None
    assert mem.created_at.tzinfo is not None


def test_transform_renames_last_accessed_to_last_accessed_at() -> None:
    """OG last_accessed → new last_accessed_at."""
    mem, _ = transform_memory(_og(last_accessed="2024-05-01T10:00:00+00:00"))
    assert mem is not None
    assert mem.last_accessed_at is not None


def test_transform_uses_emotion_score_as_score() -> None:
    """OG emotion_score → new score (verbatim)."""
    mem, _ = transform_memory(_og(emotions={"love": 8.0, "tenderness": 6.0}, emotion_score=14.0))
    assert mem is not None
    assert mem.score == 14.0


def test_transform_defaults_missing_optional_fields() -> None:
    """Missing tags / importance / active fall back to sensible defaults."""
    minimal = {
        "id": "m2",
        "content": "x",
        "memory_type": "meta",
        "domain": "work",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    mem, skipped = transform_memory(minimal)
    assert skipped is None
    assert mem is not None
    assert mem.tags == []
    assert mem.importance == 0.0
    assert mem.active is True
    assert mem.emotions == {}
    assert mem.score == 0.0


def test_transform_absorbs_og_only_fields_into_metadata() -> None:
    """source_date, source_summary, supersedes, etc. land in metadata verbatim."""
    mem, _ = transform_memory(
        _og(
            source_date="2024-01-01",
            source_summary="first contact",
            emotional_tone="tender",
            supersedes="abc",
            access_count=3,
            emotion_count=1,
            intensity=7.0,
            schema_version=2,
            connections=["xyz"],
        )
    )
    assert mem is not None
    md = mem.metadata
    assert md["source_date"] == "2024-01-01"
    assert md["source_summary"] == "first contact"
    assert md["emotional_tone"] == "tender"
    assert md["supersedes"] == "abc"
    assert md["access_count"] == 3
    assert md["emotion_count"] == 1
    assert md["intensity"] == 7.0
    assert md["schema_version"] == 2
    assert md["connections"] == ["xyz"]


def test_transform_absorbs_unknown_fields_into_metadata() -> None:
    """Unknown OG keys (forward-proof) also land in metadata."""
    mem, _ = transform_memory(_og(some_future_field="hello", another={"nested": 1}))
    assert mem is not None
    assert mem.metadata["some_future_field"] == "hello"
    assert mem.metadata["another"] == {"nested": 1}


def test_transform_skips_missing_content() -> None:
    """Memory with missing / empty content is skipped."""
    mem, skipped = transform_memory(_og(content=""))
    assert mem is None
    assert skipped is not None
    assert skipped.reason == "missing_content"

    og = _og()
    del og["content"]
    mem2, skipped2 = transform_memory(og)
    assert mem2 is None
    assert skipped2 is not None
    assert skipped2.reason == "missing_content"


def test_transform_skips_non_numeric_emotion_value() -> None:
    """Emotions dict with a non-numeric value → skip."""
    mem, skipped = transform_memory(_og(emotions={"love": "high"}))
    assert mem is None
    assert skipped is not None
    assert skipped.reason == "non_numeric_emotion"
    assert skipped.field == "emotions"


def test_transform_skips_unparseable_created_at() -> None:
    """created_at that isoformat can't parse → skip."""
    mem, skipped = transform_memory(_og(created_at="not-a-date"))
    assert mem is None
    assert skipped is not None
    assert skipped.reason == "unparseable_created_at"


def test_transform_skips_missing_id() -> None:
    """A memory without an id cannot be referenced by the Hebbian matrix; skip."""
    og = _og()
    del og["id"]
    mem, skipped = transform_memory(og)
    assert mem is None
    assert skipped is not None
    assert skipped.reason == "missing_id"


def test_transform_score_mismatch_prefers_og_value() -> None:
    """If emotion_score disagrees with sum(emotions.values()), use OG's value."""
    mem, _ = transform_memory(_og(emotions={"love": 8.0, "tenderness": 6.0}, emotion_score=99.0))
    assert mem is not None
    assert mem.score == 99.0


def test_skipped_memory_dataclass_shape() -> None:
    """SkippedMemory has id, reason, field, raw_snippet fields."""
    s = SkippedMemory(id="m1", reason="missing_content", field="content", raw_snippet="...")
    assert s.id == "m1"
    assert s.reason == "missing_content"
    assert s.field == "content"
    assert s.raw_snippet == "..."


def test_transform_non_numeric_importance_degrades_to_zero() -> None:
    """Non-numeric importance (e.g. 'high') must not crash float() — degrade to 0.0."""
    mem, skipped = transform_memory(_og(importance="high"))
    assert skipped is None
    assert mem is not None
    assert mem.importance == 0.0


def test_transform_list_importance_degrades_to_zero() -> None:
    """A list-valued importance field degrades to 0.0 rather than crashing."""
    mem, skipped = transform_memory(_og(importance=[1, 2, 3]))
    assert skipped is None
    assert mem is not None
    assert mem.importance == 0.0


def test_transform_string_tags_degrades_to_empty_list() -> None:
    """A string-valued tags field must NOT character-explode via list('mytag').

    list('mytag') returns ['m','y','t','a','g'] — silent data corruption.
    Tags is optional metadata, so degrade to [] rather than skip the memory.
    """
    mem, skipped = transform_memory(_og(tags="mytag"))
    assert skipped is None
    assert mem is not None
    assert mem.tags == []


def test_transform_dict_tags_degrades_to_empty_list() -> None:
    """A dict-valued tags field degrades to [] rather than pulling dict keys."""
    mem, skipped = transform_memory(_og(tags={"a": 1, "b": 2}))
    assert skipped is None
    assert mem is not None
    assert mem.tags == []


def test_transform_bool_emotion_score_falls_back_to_sum() -> None:
    """bool emotion_score is rejected by the isinstance(v, bool) guard; falls back to sum."""
    mem, _ = transform_memory(_og(emotions={"love": 5.0}, emotion_score=True))
    assert mem is not None
    assert mem.score == 5.0  # sum of emotions, not float(True)=1.0
