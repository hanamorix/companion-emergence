"""Tests for brain.memory.store — Memory dataclass + MemoryStore."""

from __future__ import annotations

from datetime import UTC, datetime

from brain.memory.store import Memory


def test_memory_create_new_generates_uuid() -> None:
    """Memory.create_new generates a UUID id."""
    m = Memory.create_new(
        content="first meeting",
        memory_type="conversation",
        domain="us",
    )
    assert isinstance(m.id, str)
    assert len(m.id) == 36  # canonical UUID string form
    assert m.id.count("-") == 4


def test_memory_create_new_sets_created_at_utc() -> None:
    """create_new sets created_at to a tz-aware UTC datetime."""
    before = datetime.now(UTC)
    m = Memory.create_new(content="x", memory_type="meta", domain="work")
    after = datetime.now(UTC)
    assert m.created_at.tzinfo is not None
    assert before <= m.created_at <= after


def test_memory_create_new_computes_score_from_emotions() -> None:
    """score = sum of emotion intensities at create time."""
    m = Memory.create_new(
        content="held",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0, "tenderness": 6.0},
    )
    assert m.score == 15.0


def test_memory_create_new_score_zero_when_no_emotions() -> None:
    """Empty emotions dict → score 0."""
    m = Memory.create_new(content="note", memory_type="meta", domain="work")
    assert m.score == 0.0


def test_memory_create_new_importance_defaults_to_score_over_ten() -> None:
    """If importance unspecified, default = score / 10.0 (normalised)."""
    m = Memory.create_new(
        content="held",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0, "tenderness": 6.0},
    )
    assert m.importance == 1.5  # 15.0 / 10.0


def test_memory_create_new_importance_manual_override() -> None:
    """Explicit importance overrides the score-based default."""
    m = Memory.create_new(
        content="held",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0},
        importance=7.0,
    )
    assert m.importance == 7.0


def test_memory_defaults_active_and_unprotected() -> None:
    """New memories are active and unprotected by default."""
    m = Memory.create_new(content="x", memory_type="meta", domain="work")
    assert m.active is True
    assert m.protected is False


def test_memory_to_dict_round_trips() -> None:
    """to_dict / from_dict round-trips cleanly."""
    original = Memory.create_new(
        content="the moment",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0, "anchor_pull": 8.0},
        tags=["first", "important"],
    )
    data = original.to_dict()
    restored = Memory.from_dict(data)

    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.memory_type == original.memory_type
    assert restored.domain == original.domain
    assert restored.emotions == original.emotions
    assert restored.tags == original.tags
    assert restored.score == original.score
    assert restored.importance == original.importance
    assert restored.created_at == original.created_at
    assert restored.active == original.active


def test_memory_from_dict_coerces_naive_timestamps_to_utc() -> None:
    """Naive timestamps in JSON restore as UTC-aware."""
    data = {
        "id": "00000000-0000-0000-0000-000000000001",
        "content": "legacy",
        "memory_type": "meta",
        "domain": "work",
        "emotions": {},
        "tags": [],
        "importance": 0.0,
        "score": 0.0,
        "created_at": "2024-01-01T12:00:00",  # no tz
        "last_accessed_at": None,
        "active": True,
        "protected": False,
    }
    m = Memory.from_dict(data)
    assert m.created_at.tzinfo is not None


def test_memory_dataclass_preserves_explicit_id_for_migration() -> None:
    """Memory() direct construction accepts an explicit id (for migrator use)."""
    m = Memory(
        id="abc-123",
        content="migrated",
        memory_type="conversation",
        domain="us",
        created_at=datetime.now(UTC),
    )
    assert m.id == "abc-123"
