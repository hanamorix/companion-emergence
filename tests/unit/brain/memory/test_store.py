"""Tests for brain.memory.store — Memory dataclass + MemoryStore."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from brain.memory.store import Memory, MemoryStore


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
    assert restored.last_accessed_at is None  # None survives the round-trip


def test_memory_to_dict_round_trips_last_accessed_and_protected() -> None:
    """Set last_accessed_at + protected=True and confirm they round-trip."""
    original = Memory.create_new(
        content="important memory",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0},
    )
    original.last_accessed_at = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
    original.protected = True

    restored = Memory.from_dict(original.to_dict())
    assert restored.last_accessed_at == original.last_accessed_at
    assert restored.protected is True


def test_memory_from_dict_coerces_naive_timestamps_to_utc() -> None:
    """Naive created_at AND last_accessed_at in JSON both restore as UTC-aware."""
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
        "last_accessed_at": "2024-01-02T08:00:00",  # no tz
        "active": True,
        "protected": False,
    }
    m = Memory.from_dict(data)
    assert m.created_at.tzinfo is not None
    assert m.last_accessed_at is not None
    assert m.last_accessed_at.tzinfo is not None


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


@pytest.fixture
def store() -> MemoryStore:
    """In-memory MemoryStore, fresh per test."""
    return MemoryStore(db_path=":memory:")


def _mem(content: str = "x", **kw: object) -> Memory:
    defaults = {"memory_type": "conversation", "domain": "us"}
    defaults.update(kw)
    return Memory.create_new(content=content, **defaults)  # type: ignore[arg-type]


def test_store_init_creates_schema(store: MemoryStore) -> None:
    """Fresh store has a memories table."""
    cursor = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    )
    assert cursor.fetchone() is not None


def test_store_create_and_get_round_trips(store: MemoryStore) -> None:
    """create() then get() returns the same Memory."""
    original = _mem("first meeting", emotions={"love": 9.0})
    store.create(original)

    restored = store.get(original.id)
    assert restored is not None
    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.emotions == original.emotions
    assert restored.score == original.score


def test_store_get_unknown_returns_none(store: MemoryStore) -> None:
    """get() on a nonexistent id returns None."""
    assert store.get("nonexistent-id") is None


def test_store_create_returns_the_memory_id(store: MemoryStore) -> None:
    """create() returns the id it stored."""
    m = _mem("x")
    returned = store.create(m)
    assert returned == m.id


def test_store_create_duplicate_id_raises(store: MemoryStore) -> None:
    """Creating two memories with the same id raises."""
    m = _mem("x")
    store.create(m)
    with pytest.raises(sqlite3.IntegrityError):
        store.create(m)


def test_store_list_by_domain(store: MemoryStore) -> None:
    """list_by_domain filters correctly."""
    store.create(_mem("a", domain="us"))
    store.create(_mem("b", domain="us"))
    store.create(_mem("c", domain="work"))

    us = store.list_by_domain("us")
    work = store.list_by_domain("work")
    assert len(us) == 2
    assert len(work) == 1
    assert all(m.domain == "us" for m in us)


def test_store_list_by_type(store: MemoryStore) -> None:
    """list_by_type filters by memory_type."""
    store.create(_mem("a", memory_type="conversation"))
    store.create(_mem("b", memory_type="meta"))
    store.create(_mem("c", memory_type="conversation"))

    convs = store.list_by_type("conversation")
    assert len(convs) == 2
    assert all(m.memory_type == "conversation" for m in convs)


def test_store_list_by_emotion_filters_by_intensity(store: MemoryStore) -> None:
    """list_by_emotion returns memories where that emotion >= min_intensity."""
    store.create(_mem("a", emotions={"love": 9.0}))
    store.create(_mem("b", emotions={"love": 3.0}))
    store.create(_mem("c", emotions={"grief": 8.0}))

    strong_love = store.list_by_emotion("love", min_intensity=5.0)
    assert len(strong_love) == 1
    assert strong_love[0].content == "a"


def test_store_list_excludes_inactive_by_default(store: MemoryStore) -> None:
    """list_by_domain excludes deactivated memories by default."""
    m1 = _mem("active", domain="us")
    m2 = _mem("inactive", domain="us")
    store.create(m1)
    store.create(m2)
    store.deactivate(m2.id)

    active = store.list_by_domain("us")
    assert len(active) == 1
    assert active[0].content == "active"


def test_store_list_includes_inactive_when_requested(store: MemoryStore) -> None:
    """Passing active_only=False includes deactivated memories."""
    m1 = _mem("active", domain="us")
    m2 = _mem("inactive", domain="us")
    store.create(m1)
    store.create(m2)
    store.deactivate(m2.id)

    all_ = store.list_by_domain("us", active_only=False)
    assert len(all_) == 2


def test_store_list_respects_limit(store: MemoryStore) -> None:
    """limit caps the result count."""
    for i in range(5):
        store.create(_mem(f"m{i}", domain="us"))
    assert len(store.list_by_domain("us", limit=3)) == 3


def test_store_update_mutates_specified_fields(store: MemoryStore) -> None:
    """update() mutates only the given fields."""
    m = _mem("original")
    store.create(m)
    store.update(m.id, content="modified", importance=9.0)

    restored = store.get(m.id)
    assert restored is not None
    assert restored.content == "modified"
    assert restored.importance == 9.0
    assert restored.domain == m.domain  # unchanged


def test_store_update_unknown_raises(store: MemoryStore) -> None:
    """update() on a nonexistent id raises KeyError."""
    with pytest.raises(KeyError):
        store.update("nonexistent", content="x")


def test_store_deactivate_flips_active_flag(store: MemoryStore) -> None:
    """deactivate() sets active=False without deleting the row."""
    m = _mem("x")
    store.create(m)
    store.deactivate(m.id)

    restored = store.get(m.id)
    assert restored is not None
    assert restored.active is False


def test_store_deactivate_unknown_raises(store: MemoryStore) -> None:
    """deactivate() on a nonexistent id raises KeyError."""
    with pytest.raises(KeyError):
        store.deactivate("nonexistent")


def test_store_count_active_only_default(store: MemoryStore) -> None:
    """count() excludes inactive by default."""
    m1 = _mem("a")
    m2 = _mem("b")
    store.create(m1)
    store.create(m2)
    store.deactivate(m2.id)
    assert store.count() == 1


def test_store_count_including_inactive(store: MemoryStore) -> None:
    """count(active_only=False) includes inactive memories."""
    m1 = _mem("a")
    m2 = _mem("b")
    store.create(m1)
    store.create(m2)
    store.deactivate(m2.id)
    assert store.count(active_only=False) == 2


def test_store_search_text_returns_substring_matches(store: MemoryStore) -> None:
    """search_text finds memories whose content contains the query."""
    store.create(_mem("cold coffee, warm hana"))
    store.create(_mem("the evening has a shape to it now"))
    store.create(_mem("creative hunger strikes"))

    results = store.search_text("evening")
    assert len(results) == 1
    assert "evening" in results[0].content


def test_store_search_text_is_case_insensitive(store: MemoryStore) -> None:
    """Substring matching ignores case."""
    store.create(_mem("The Moment"))
    results = store.search_text("moment")
    assert len(results) == 1
