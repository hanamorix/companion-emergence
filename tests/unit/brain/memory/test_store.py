"""Tests for brain.memory.store — Memory dataclass + MemoryStore."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from brain.memory.store import _SCHEMA, Memory, MemoryStore


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


def test_memory_metadata_defaults_to_empty_dict() -> None:
    """metadata defaults to {} if not specified."""
    m = Memory.create_new(content="x", memory_type="meta", domain="work")
    assert m.metadata == {}


def test_memory_create_new_accepts_metadata_kwarg() -> None:
    """Memory.create_new accepts a metadata dict and preserves it verbatim."""
    m = Memory.create_new(
        content="x",
        memory_type="meta",
        domain="work",
        metadata={"source_date": "2024-01-01", "supersedes": "abc-123"},
    )
    assert m.metadata == {"source_date": "2024-01-01", "supersedes": "abc-123"}


def test_memory_metadata_round_trips_through_dict() -> None:
    """metadata round-trips through to_dict / from_dict."""
    original = Memory.create_new(
        content="x",
        memory_type="meta",
        domain="work",
        metadata={"emotional_tone": "tender", "access_count": 3, "tags_sig": None},
    )
    data = original.to_dict()
    assert data["metadata"] == original.metadata
    restored = Memory.from_dict(data)
    assert restored.metadata == original.metadata


def test_memory_from_dict_missing_metadata_defaults_empty() -> None:
    """Legacy dicts without a 'metadata' key restore cleanly with metadata={}."""
    data = {
        "id": "legacy-001",
        "content": "legacy",
        "memory_type": "meta",
        "domain": "work",
        "emotions": {},
        "tags": [],
        "importance": 0.0,
        "score": 0.0,
        "created_at": datetime.now(UTC).isoformat(),
        "last_accessed_at": None,
        "active": True,
        "protected": False,
        # no 'metadata' key
    }
    restored = Memory.from_dict(data)
    assert restored.metadata == {}


def test_memory_metadata_defensive_copy_on_create_new() -> None:
    """Mutating the caller's dict after create_new does not affect the memory."""
    caller_dict = {"source_date": "2024-01-01"}
    m = Memory.create_new(content="x", memory_type="meta", domain="work", metadata=caller_dict)
    caller_dict["source_date"] = "mutated"
    assert m.metadata == {"source_date": "2024-01-01"}


def test_memory_from_dict_metadata_null_defaults_empty() -> None:
    """Explicit JSON null for 'metadata' restores as {}, not a crash.

    OG JSON can legally contain "metadata": null; the migrator will feed
    those dicts straight into from_dict. `dict(None)` would TypeError, so
    from_dict uses `data.get("metadata") or {}` to absorb both the
    absent-key and present-but-null cases.
    """
    data = {
        "id": "null-md-001",
        "content": "y",
        "memory_type": "meta",
        "domain": "work",
        "emotions": {},
        "tags": [],
        "importance": 0.0,
        "score": 0.0,
        "created_at": datetime.now(UTC).isoformat(),
        "last_accessed_at": None,
        "active": True,
        "protected": False,
        "metadata": None,  # explicit null
    }
    restored = Memory.from_dict(data)
    assert restored.metadata == {}


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


def test_store_search_text_rejects_empty_query(store: MemoryStore) -> None:
    """Callers must choose list_active() explicitly instead of LIKE '%%'."""
    store.create(_mem("cold coffee, warm hana"))

    with pytest.raises(ValueError, match="use list_active"):
        store.search_text("")


def test_store_list_active_is_explicit_list_all_path(store: MemoryStore) -> None:
    """list_active returns active memories without relying on empty search."""
    active = _mem("active")
    inactive = _mem("inactive")
    store.create(active)
    store.create(inactive)
    store.deactivate(inactive.id)

    results = store.list_active()

    assert [m.id for m in results] == [active.id]


def test_store_list_active_since_none_cursor_returns_from_beginning(
    store: MemoryStore,
) -> None:
    """cursor_iso=None starts from the oldest active memory, ascending."""
    older = _mem("older")
    older.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    newer = _mem("newer")
    newer.created_at = datetime(2020, 1, 2, tzinfo=UTC)
    store.create(older)
    store.create(newer)

    results = store.list_active_since(None, limit=10)

    assert [m.id for m in results] == [older.id, newer.id]


def test_store_list_active_since_excludes_at_or_before_cursor(
    store: MemoryStore,
) -> None:
    """Only rows strictly AFTER the (created_at, id) cursor are returned."""
    a = _mem("a")
    a.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    b = _mem("b")
    b.created_at = datetime(2020, 1, 2, tzinfo=UTC)
    c = _mem("c")
    c.created_at = datetime(2020, 1, 3, tzinfo=UTC)
    store.create(a)
    store.create(b)
    store.create(c)

    results = store.list_active_since((b.created_at.isoformat(), b.id), limit=10)

    assert [m.id for m in results] == [c.id]


def test_store_list_active_since_tie_inclusive_on_shared_created_at(
    store: MemoryStore,
) -> None:
    """A cursor pinned at (ts, id) of one row must still return ANOTHER row
    sharing that EXACT created_at, provided its id sorts after the cursor's.

    Regression for the keyset-pagination fix: the old cursor was a bare
    `created_at > ?` filter, so pinning it at the created_at of a row that
    shared its timestamp with siblings made every sibling but the pinned one
    permanently invisible to every future call — exactly the shape produced
    by a bulk migrator import (brain/migrator/emergence_kit.py) where many
    rows commonly share a second-granularity created_at.
    """
    shared_ts = datetime(2020, 1, 1, tzinfo=UTC)
    first = _mem("first")
    first.created_at = shared_ts
    second = _mem("second")
    second.created_at = shared_ts
    # Force a deterministic id ordering regardless of uuid4 randomness, so
    # the test asserts the intended tie-break rather than depending on luck.
    first.id = "aaaa0000-0000-0000-0000-000000000000"
    second.id = "bbbb0000-0000-0000-0000-000000000000"
    store.create(first)
    store.create(second)

    # Pin the cursor exactly at `first`'s (created_at, id) — as
    # embedding_backfill does after processing `first`.
    results = store.list_active_since((shared_ts.isoformat(), first.id), limit=10)

    assert [m.id for m in results] == [second.id]


def test_store_list_active_since_respects_limit(store: MemoryStore) -> None:
    """Bounded per call — never returns more than `limit` rows."""
    for i in range(5):
        m = _mem(f"item-{i}")
        m.created_at = datetime(2020, 1, i + 1, tzinfo=UTC)
        store.create(m)

    results = store.list_active_since(None, limit=2)

    assert len(results) == 2
    assert [m.content for m in results] == ["item-0", "item-1"]


def test_store_list_active_since_excludes_inactive(store: MemoryStore) -> None:
    """Same active=1 filter as list_active() — deactivated rows never appear."""
    active = _mem("active")
    inactive = _mem("inactive")
    store.create(active)
    store.create(inactive)
    store.deactivate(inactive.id)

    results = store.list_active_since(None, limit=10)

    assert [m.id for m in results] == [active.id]


# ---------------------------------------------------------------------------
# list_unembedded_since — the embedding backfill's own backlog query (F1
# #259 increment 3). Same keyset-cursor shape as list_active_since above,
# scoped to `embedding IS NULL`.
# ---------------------------------------------------------------------------


def test_store_list_unembedded_since_excludes_already_embedded_rows(
    store: MemoryStore,
) -> None:
    """A row embedded under the CURRENT model never appears in the backlog
    query — backlog membership is the row's own columns, not a side-table
    lookup. (The row is embedded under "some-model", and the query is asked
    about "some-model" too, so this is the steady-state / no-swap case —
    see the model-mismatch tests below for the swap case.)"""
    embedded = _mem("already embedded")
    embedded.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    unembedded = _mem("still needs an embedding")
    unembedded.created_at = datetime(2020, 1, 2, tzinfo=UTC)
    store.create(embedded)
    store.create(unembedded)
    store._conn.execute(
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (b"\x00" * 4, "some-model", embedded.id),
    )
    store._conn.commit()

    results = store.list_unembedded_since(
        None, limit=10, current_model_id="some-model", min_chars=0
    )

    assert [m.id for m in results] == [unembedded.id]


def test_store_list_unembedded_since_includes_stale_model_rows(
    store: MemoryStore,
) -> None:
    """FIX A (Planning-ruled 2026-09-16, F1 #259 increment-3 spec-gap): a row
    embedded under a model_id that no longer matches `current_model_id` (a
    model swap happened) IS backlog — `embedding_model_id != current` — even
    though its `embedding` column is non-NULL. Restores the model-scoped
    self-healing the old content-hash cache had; without this, a post-swap
    row would fall to lexical recall forever with no re-embed path."""
    stale = _mem("embedded under the old model")
    stale.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    current = _mem("embedded under the current model")
    current.created_at = datetime(2020, 1, 2, tzinfo=UTC)
    store.create(stale)
    store.create(current)
    store._conn.execute(
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (b"\x00" * 4, "old-model", stale.id),
    )
    store._conn.execute(
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (b"\x00" * 4, "new-model", current.id),
    )
    store._conn.commit()

    results = store.list_unembedded_since(
        None, limit=10, current_model_id="new-model", min_chars=0
    )

    # The stale-model row IS in the backlog; the current-model row is not.
    assert [m.id for m in results] == [stale.id]


def test_store_list_unembedded_since_excludes_short_rows(
    store: MemoryStore,
) -> None:
    """FIX B (F1 #259 increment-3 red-team): a row under `min_chars` is
    excluded at the SQL level, not merely skipped Python-side — it never
    even appears in the returned candidates."""
    short = _mem("x" * 5)
    short.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    long_enough = _mem("y" * 25)
    long_enough.created_at = datetime(2020, 1, 2, tzinfo=UTC)
    store.create(short)
    store.create(long_enough)

    results = store.list_unembedded_since(
        None, limit=10, current_model_id="some-model", min_chars=20
    )

    assert [m.id for m in results] == [long_enough.id]


def test_store_list_unembedded_since_none_cursor_returns_from_beginning(
    store: MemoryStore,
) -> None:
    """cursor=None starts from the oldest still-unembedded memory, ascending."""
    older = _mem("older")
    older.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    newer = _mem("newer")
    newer.created_at = datetime(2020, 1, 2, tzinfo=UTC)
    store.create(older)
    store.create(newer)

    results = store.list_unembedded_since(
        None, limit=10, current_model_id="some-model", min_chars=0
    )

    assert [m.id for m in results] == [older.id, newer.id]


def test_store_list_unembedded_since_excludes_at_or_before_cursor(
    store: MemoryStore,
) -> None:
    """Only rows strictly AFTER the (created_at, id) cursor are returned."""
    a = _mem("a")
    a.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    b = _mem("b")
    b.created_at = datetime(2020, 1, 2, tzinfo=UTC)
    c = _mem("c")
    c.created_at = datetime(2020, 1, 3, tzinfo=UTC)
    store.create(a)
    store.create(b)
    store.create(c)

    results = store.list_unembedded_since(
        (b.created_at.isoformat(), b.id), limit=10, current_model_id="some-model", min_chars=0
    )

    assert [m.id for m in results] == [c.id]


def test_store_list_unembedded_since_tie_inclusive_on_shared_created_at(
    store: MemoryStore,
) -> None:
    """A cursor pinned at (ts, id) of one row must still return ANOTHER row
    sharing that EXACT created_at, provided its id sorts after the cursor's
    — same tied-timestamp regression coverage as list_active_since."""
    shared_ts = datetime(2020, 1, 1, tzinfo=UTC)
    first = _mem("first")
    first.created_at = shared_ts
    second = _mem("second")
    second.created_at = shared_ts
    first.id = "aaaa0000-0000-0000-0000-000000000000"
    second.id = "bbbb0000-0000-0000-0000-000000000000"
    store.create(first)
    store.create(second)

    results = store.list_unembedded_since(
        (shared_ts.isoformat(), first.id), limit=10, current_model_id="some-model", min_chars=0
    )

    assert [m.id for m in results] == [second.id]


def test_store_list_unembedded_since_respects_limit(store: MemoryStore) -> None:
    """Bounded per call — never returns more than `limit` rows."""
    for i in range(5):
        m = _mem(f"item-{i}")
        m.created_at = datetime(2020, 1, i + 1, tzinfo=UTC)
        store.create(m)

    results = store.list_unembedded_since(
        None, limit=2, current_model_id="some-model", min_chars=0
    )

    assert len(results) == 2
    assert [m.content for m in results] == ["item-0", "item-1"]


def test_store_list_unembedded_since_excludes_inactive(store: MemoryStore) -> None:
    """Same active=1 filter as list_active()/list_active_since() —
    deactivated rows never appear even if unembedded."""
    active = _mem("active")
    inactive = _mem("inactive")
    store.create(active)
    store.create(inactive)
    store.deactivate(inactive.id)

    results = store.list_unembedded_since(
        None, limit=10, current_model_id="some-model", min_chars=0
    )

    assert [m.id for m in results] == [active.id]


def test_store_list_unembedded_since_row_reappears_after_going_null_again(
    store: MemoryStore,
) -> None:
    """A row that was embedded and then goes NULL again (e.g. a content
    edit whose synchronous re-embed failed — see
    `MemoryStore._reembed_or_clear`) must reappear in the backlog query
    regardless of its `created_at` position relative to any OTHER row —
    membership is purely `embedding IS NULL`, not a forward-only cursor
    position."""
    m = _mem("will be embedded, then re-nulled")
    m.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    store.create(m)
    store._conn.execute(
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (b"\x00" * 4, "some-model", m.id),
    )
    store._conn.commit()
    assert (
        store.list_unembedded_since(
            None, limit=10, current_model_id="some-model", min_chars=0
        )
        == []
    )

    store._conn.execute(
        "UPDATE memories SET embedding = NULL, embedding_model_id = NULL WHERE id = ?",
        (m.id,),
    )
    store._conn.commit()

    results = store.list_unembedded_since(
        None, limit=10, current_model_id="some-model", min_chars=0
    )
    assert [row.id for row in results] == [m.id]


# ---------------------------------------------------------------------------
# count_unembedded — same backlog predicate as list_unembedded_since, but a
# cursor-free full-table COUNT (F1 #259 increment 6, for the `nell embed
# backfill` CLI's progress-ticker denominator + the drain-to-completion
# helper's final failure count).
# ---------------------------------------------------------------------------


def test_count_unembedded_matches_list_unembedded_since_predicate(
    store: MemoryStore,
) -> None:
    """Same backlog membership as list_unembedded_since: NULL embedding OR
    stale model, active, long enough."""
    embedded_current = _mem("embedded under the current model" + "x" * 20)
    embedded_stale = _mem("embedded under a stale model" + "x" * 20)
    unembedded = _mem("never embedded" + "x" * 20)
    too_short = _mem("short")
    inactive = _mem("inactive but otherwise eligible" + "x" * 20)
    store.create(embedded_current)
    store.create(embedded_stale)
    store.create(unembedded)
    store.create(too_short)
    store.create(inactive)
    store.deactivate(inactive.id)
    store._conn.execute(
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (b"\x00" * 4, "current-model", embedded_current.id),
    )
    store._conn.execute(
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (b"\x00" * 4, "old-model", embedded_stale.id),
    )
    store._conn.commit()

    count = store.count_unembedded(current_model_id="current-model", min_chars=20)

    # Backlog: `unembedded` (NULL) + `embedded_stale` (model mismatch).
    # NOT `embedded_current` (matches), `too_short` (< min_chars), or
    # `inactive` (deactivated).
    assert count == 2
    matching_ids = {
        m.id
        for m in store.list_unembedded_since(
            None, limit=10, current_model_id="current-model", min_chars=20
        )
    }
    assert matching_ids == {unembedded.id, embedded_stale.id}


def test_count_unembedded_zero_on_empty_store(store: MemoryStore) -> None:
    """No rows at all -> 0, not an error."""
    assert store.count_unembedded(current_model_id="any-model", min_chars=0) == 0


def test_count_unembedded_zero_once_everything_matches_current_model(
    store: MemoryStore,
) -> None:
    """Steady state: every row already embedded under the queried model ->
    0, matching list_unembedded_since's own no-op steady state."""
    m = _mem("embedded and current" + "x" * 20)
    store.create(m)
    store._conn.execute(
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (b"\x00" * 4, "current-model", m.id),
    )
    store._conn.commit()

    assert store.count_unembedded(current_model_id="current-model", min_chars=0) == 0


def test_count_unembedded_ignores_any_persisted_backfill_cursor(
    store: MemoryStore,
) -> None:
    """count_unembedded takes NO cursor argument at all — it always counts
    from the top of the table, unlike list_unembedded_since which a caller
    can (and the backfill does) page through with a persisted cursor. This
    is the property `run_embedding_backfill_to_completion` relies on to
    report the TRUE remaining backlog even when a scattered permanently-
    failing row sits behind an already-advanced forward cursor position."""
    older = _mem("older, still unembedded" + "x" * 20)
    older.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    newer = _mem("newer, still unembedded" + "x" * 20)
    newer.created_at = datetime(2020, 1, 2, tzinfo=UTC)
    store.create(older)
    store.create(newer)

    # A caller that had already paged PAST `older` via list_unembedded_since
    # would no longer see it — count_unembedded is unaffected by any such
    # cursor because it never takes one.
    paged = store.list_unembedded_since(
        (older.created_at.isoformat(), older.id),
        limit=10,
        current_model_id="some-model",
        min_chars=0,
    )
    assert [row.id for row in paged] == [newer.id]

    assert store.count_unembedded(current_model_id="some-model", min_chars=0) == 2


def test_store_search_text_is_case_insensitive(store: MemoryStore) -> None:
    """Substring matching ignores case."""
    store.create(_mem("The Moment"))
    results = store.search_text("moment")
    assert len(results) == 1


def test_store_search_text_escapes_like_wildcards(store: MemoryStore) -> None:
    """`%` and `_` in the query match literally, not as LIKE wildcards.

    Without escaping, search_text("%") would wrap to "%%%" and match every
    row — a footgun the moment Task 5 composes this with user-supplied
    queries.
    """
    store.create(_mem("cold coffee"))
    store.create(_mem("50% cream"))
    store.create(_mem("under_score"))

    assert len(store.search_text("%")) == 1  # only the one with literal %
    assert len(store.search_text("_")) == 1  # only the one with literal _
    assert store.search_text("%")[0].content == "50% cream"
    assert store.search_text("_")[0].content == "under_score"


def test_store_list_by_emotion_skips_non_numeric_values(store: MemoryStore) -> None:
    """If a memory's emotions_json has a non-numeric value for the queried
    emotion (e.g. from a malformed migration), skip it without raising.
    """
    m = _mem("corrupt", emotions={"love": 9.0})
    store.create(m)
    # Manually corrupt the stored JSON to simulate bad migrator output.
    import json as _json

    store._conn.execute(
        "UPDATE memories SET emotions_json = ? WHERE id = ?",
        (_json.dumps({"love": "high"}), m.id),
    )
    store._conn.commit()

    # Must not raise TypeError from the >= comparison.
    results = store.list_by_emotion("love", min_intensity=5.0)
    assert results == []


def test_store_create_and_get_preserves_metadata(store: MemoryStore) -> None:
    """MemoryStore.create and .get round-trip metadata dict."""
    m = _mem("x", metadata={"source_date": "2024-01-01", "supersedes": "abc-123"})
    store.create(m)
    restored = store.get(m.id)
    assert restored is not None
    assert restored.metadata == {"source_date": "2024-01-01", "supersedes": "abc-123"}


def test_store_create_empty_metadata_survives(store: MemoryStore) -> None:
    """Default empty metadata dict round-trips as {} (not None or missing)."""
    m = _mem("x")
    store.create(m)
    restored = store.get(m.id)
    assert restored is not None
    assert restored.metadata == {}


def test_store_update_metadata_field(store: MemoryStore) -> None:
    """update() can mutate the metadata field."""
    m = _mem("x", metadata={"v": 1})
    store.create(m)
    store.update(m.id, metadata={"v": 2, "added": "yes"})

    restored = store.get(m.id)
    assert restored is not None
    assert restored.metadata == {"v": 2, "added": "yes"}


def test_store_update_rejects_unknown_field_still(store: MemoryStore) -> None:
    """Update's unknown-field guard still works after the metadata addition."""
    m = _mem("x")
    store.create(m)
    with pytest.raises(ValueError, match="Unknown update field"):
        store.update(m.id, nonsense_field="oops")


def test_store_row_with_null_metadata_string_returns_empty_dict(store: MemoryStore) -> None:
    """A row whose metadata_json column is the literal JSON null string
    (not the SQL NULL) must NOT materialise Memory.metadata as None.

    Belt-and-braces guard against manually-edited DBs or a future writer
    that mistakenly calls json.dumps(None). Without the type check in
    _safe_load_metadata, `json.loads('null')` → None would silently poison
    every consumer doing memory.metadata.get(...).
    """
    m = _mem("x", metadata={"real": "data"})
    store.create(m)
    # Manually poison the row (simulates a bad external writer).
    store._conn.execute("UPDATE memories SET metadata_json = 'null' WHERE id = ?", (m.id,))
    store._conn.commit()

    restored = store.get(m.id)
    assert restored is not None
    assert restored.metadata == {}  # not None


def test_store_row_with_malformed_metadata_returns_empty_dict(store: MemoryStore) -> None:
    """Malformed JSON in metadata_json falls back to {} rather than raising."""
    m = _mem("x")
    store.create(m)
    store._conn.execute(
        "UPDATE memories SET metadata_json = ? WHERE id = ?", ("{not valid json", m.id)
    )
    store._conn.commit()

    restored = store.get(m.id)
    assert restored is not None
    assert restored.metadata == {}


def test_store_create_reports_memory_id_on_unjsonable_metadata(store: MemoryStore) -> None:
    """Non-JSON-serialisable metadata surfaces TypeError with the memory id.

    Useful for ETL: with 1,141 records going through create(), a bare
    'Object of type datetime is not JSON serializable' is useless. The
    wrapped error names the offending memory id.
    """
    m = _mem("x", metadata={"ts": datetime.now(UTC)})  # datetime not JSON-serialisable
    with pytest.raises(TypeError, match=f"id={m.id!r}"):
        store.create(m)


def test_store_update_empty_metadata_is_explicit_overwrite(store: MemoryStore) -> None:
    """update(metadata={}) is an explicit overwrite, consistent with update's
    other fields. Callers who want to skip the field should omit the kwarg.
    Pinning the semantic — accidental data loss is surfaced by the test.
    """
    m = _mem("x", metadata={"source_date": "2024-01-01", "supersedes": "abc"})
    store.create(m)
    store.update(m.id, metadata={})

    restored = store.get(m.id)
    assert restored is not None
    assert restored.metadata == {}


def test_store_create_and_get_preserves_nested_metadata(store: MemoryStore) -> None:
    """Complex nested metadata (dict-of-dict, list, None values) round-trips."""
    m = _mem(
        "x",
        metadata={
            "nested": {"a": 1, "b": [2, 3]},
            "list": ["x", "y", None],
            "null_value": None,
            "int_val": 42,
            "float_val": 3.14,
        },
    )
    store.create(m)
    restored = store.get(m.id)
    assert restored is not None
    assert restored.metadata == {
        "nested": {"a": 1, "b": [2, 3]},
        "list": ["x", "y", None],
        "null_value": None,
        "int_val": 42,
        "float_val": 3.14,
    }


def test_list_filter_rejects_unknown_column() -> None:
    """_list_filter raises ValueError for column names not in the allowlist."""
    store = MemoryStore(":memory:")
    try:
        with pytest.raises(ValueError, match="Invalid filter column"):
            store._list_filter("created_at; DROP TABLE memories--", "x", True, None)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Phase 1 (v0.0.14-alpha.3 Forgetting): schema migration + fade/unfade/hard_delete + recall bumping
# ---------------------------------------------------------------------------


def test_fresh_store_has_forgetting_columns() -> None:
    store = MemoryStore(":memory:")
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert "state" in cols
    assert "content_snapshot" in cols
    assert "recall_count" in cols
    store.close()


def test_existing_store_migrates_in_missing_forgetting_columns(tmp_path) -> None:
    """Simulate an upgraded persona — manually create the OLD schema, then open MemoryStore."""
    db_path = tmp_path / "memories.db"
    old_schema = """
    CREATE TABLE memories (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        domain TEXT NOT NULL,
        emotions_json TEXT NOT NULL,
        tags_json TEXT NOT NULL,
        importance REAL NOT NULL DEFAULT 0.0,
        score REAL NOT NULL DEFAULT 0.0,
        created_at TEXT NOT NULL,
        last_accessed_at TEXT,
        active INTEGER NOT NULL DEFAULT 1,
        protected INTEGER NOT NULL DEFAULT 0,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    """
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(str(db_path))
    conn.executescript(old_schema)
    conn.execute(
        "INSERT INTO memories (id, content, memory_type, domain, emotions_json, tags_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("mem_legacy", "old body", "episodic", "chat", "{}", "[]", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db_path)
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert "state" in cols
    assert "content_snapshot" in cols
    assert "recall_count" in cols
    # Pre-existing row survives migration with default state.
    row = store._conn.execute(
        "SELECT state, content_snapshot, recall_count FROM memories WHERE id = ?",
        ("mem_legacy",),
    ).fetchone()
    assert row["state"] == "active"
    assert row["content_snapshot"] is None
    assert row["recall_count"] == 0
    store.close()


def test_fade_snapshots_content_and_replaces_with_summary() -> None:
    store = MemoryStore(":memory:")
    m = Memory.create_new(
        content="this is the original long detailed body",
        memory_type="episodic",
        domain="chat",
        emotions={"joy": 5.0},
    )
    store.create(m)
    store.fade(m.id, summary="short summary")
    row = store._conn.execute(
        "SELECT content, content_snapshot, state FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["content"] == "short summary"
    assert row["content_snapshot"] == "this is the original long detailed body"
    assert row["state"] == "fading"
    store.close()


def test_unfade_restores_content_and_clears_snapshot() -> None:
    store = MemoryStore(":memory:")
    m = Memory.create_new(content="original", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.fade(m.id, summary="short")
    store.unfade(m.id)
    row = store._conn.execute(
        "SELECT content, content_snapshot, state FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["content"] == "original"
    assert row["content_snapshot"] is None
    assert row["state"] == "active"
    store.close()


def test_hard_delete_drops_the_row() -> None:
    store = MemoryStore(":memory:")
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.hard_delete(m.id)
    assert store.get(m.id) is None
    assert (
        store._conn.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (m.id,)).fetchone()[0]
        == 0
    )
    store.close()


def test_fade_raises_on_unknown_id() -> None:
    store = MemoryStore(":memory:")
    with pytest.raises(KeyError):
        store.fade("mem_nonexistent", summary="x")
    store.close()


def test_unfade_with_null_snapshot_is_noop_with_warning(caplog) -> None:
    import logging

    store = MemoryStore(":memory:")
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    # No fade — content_snapshot is NULL. unfade should warn but not crash.
    with caplog.at_level(logging.WARNING, logger="brain.memory.store"):
        store.unfade(m.id)
    assert "content_snapshot" in caplog.text.lower() or "null" in caplog.text.lower()
    # Content unchanged, state stays active.
    row = store._conn.execute(
        "SELECT content, state FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["content"] == "x"
    assert row["state"] == "active"
    store.close()


def test_hard_delete_raises_on_unknown_id() -> None:
    store = MemoryStore(":memory:")
    with pytest.raises(KeyError):
        store.hard_delete("mem_nonexistent")
    store.close()


# ---------------------------------------------------------------------------
# F1 #259 step 5: content-mutation invalidation. fade() / update(content=) /
# unfade() must synchronously re-embed (row + warm matrix); hard_delete()
# must evict the matrix entry. A failed re-embed clears the row's stale
# vector rather than leaving it describing old content.
#
# These use a REAL tmp_path db FILE, never MemoryStore(":memory:") — the
# warm matrix's lazy build opens its OWN separate sqlite3 connection to
# `store.db_path` and reloads straight from disk; an in-memory-only store's
# writes are invisible to that connection (two independent `:memory:`
# databases), so a `put()` would look like it landed but silently vanish on
# the next `ensure_built()`.
# ---------------------------------------------------------------------------


def _use_matrix_dim_fake_provider(monkeypatch: pytest.MonkeyPatch):
    """Override the process-cached embedding provider to a
    `FakeEmbeddingProvider` sized to `model_tier.MODEL_EMBEDDING_DIM` and
    align `model_tier`'s embedding tier to its model id.

    Two separate reasons this alignment is needed, both load-bearing for
    every test below:
      (1) DIMENSION — the suite-wide autouse fixture fakes the provider to
          `FakeEmbeddingProvider(dim=256)`, but `EmbeddingMatrix` expects a
          blob width derived from `model_tier.MODEL_EMBEDDING_DIM` (F1 #259
          increment 7 — no longer a hardcoded literal, but still must MATCH
          whatever that constant currently is) and silently SKIPS any
          other-width row — a 256-dim embed would never appear in the
          matrix no matter what. Sizing this fixture off the same constant
          (rather than a literal) means it keeps matching automatically if
          `MODEL_EMBEDDING_DIM` is ever repointed (e.g. a multilingual model
          swap).
      (2) MODEL ID — `embed_row` embeds via the process-cached provider, but
          the matrix's lazy-build filter is sourced from
          `model_tier.model_for_tier(TIER_EMBEDDING)` (F1 #259 step 0) — a
          SEPARATE lookup that must be aligned or the matrix's first read
          reloads from disk filtered to the wrong model id and finds
          nothing.
    """
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.FakeEmbeddingProvider(dim=model_tier.MODEL_EMBEDDING_DIM)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())
    return provider


def test_fade_reembeds_row_and_matrix_with_new_content(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails pre-fix: without `_reembed_or_clear` wired into `fade()`, the
    row's embedding would stay the ORIGINAL content's vector (embed_row was
    only ever called once, at simulated embed-on-write time) instead of the
    faded summary's — a stale vector under new content."""
    import numpy as np

    from brain.memory.embedding_matrix import build_embedding_matrix

    provider = _use_matrix_dim_fake_provider(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(content="original long body", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.embed_row(m.id, m.content)  # simulate embed-on-write already having run

    store.fade(m.id, summary="short summary")

    row = store._conn.execute(
        "SELECT embedding, embedding_model_id FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["embedding"] is not None
    expected = provider.embed("short summary").astype(np.float32).tobytes()
    assert row["embedding"] == expected, "the row vector must reflect the NEW (faded) content, not the original"
    assert row["embedding_model_id"] == provider.model_id()

    matrix = build_embedding_matrix(store.db_path)
    np.testing.assert_array_equal(matrix.get(m.id), provider.embed("short summary").astype(np.float32))
    store.close()


def test_update_content_reembeds_row_and_matrix(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails pre-fix: `update(memory_id, content=...)` never re-embedded, so
    the row kept the OLD content's vector after a content mutation."""
    import numpy as np

    from brain.memory.embedding_matrix import build_embedding_matrix

    provider = _use_matrix_dim_fake_provider(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(content="old content", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.embed_row(m.id, m.content)

    store.update(m.id, content="brand new content")

    row = store._conn.execute("SELECT embedding FROM memories WHERE id = ?", (m.id,)).fetchone()
    expected = provider.embed("brand new content").astype(np.float32).tobytes()
    assert row["embedding"] == expected

    matrix = build_embedding_matrix(store.db_path)
    np.testing.assert_array_equal(matrix.get(m.id), provider.embed("brand new content").astype(np.float32))
    store.close()


def test_update_without_content_does_not_reembed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-content field update must NOT trigger a re-embed — only a
    `content` change invalidates the vector."""
    _use_matrix_dim_fake_provider(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(content="stable content", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.embed_row(m.id, m.content)
    before = store._conn.execute(
        "SELECT embedding FROM memories WHERE id = ?", (m.id,)
    ).fetchone()["embedding"]

    store.update(m.id, importance=7.0)

    after = store._conn.execute(
        "SELECT embedding FROM memories WHERE id = ?", (m.id,)
    ).fetchone()["embedding"]
    assert after == before, "a non-content update must never touch the row's embedding"
    store.close()


def test_unfade_reembeds_row_and_matrix_with_restored_content(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fails pre-fix: `unfade()` never re-embedded, so the row kept the
    fade-summary's vector after content was restored to the full body."""
    provider = _use_matrix_dim_fake_provider(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(
        content="the full original body", memory_type="episodic", domain="chat", emotions={}
    )
    store.create(m)
    store.embed_row(m.id, m.content)
    store.fade(m.id, summary="short summary")  # row now embeds "short summary"

    store.unfade(m.id)

    row = store._conn.execute("SELECT embedding FROM memories WHERE id = ?", (m.id,)).fetchone()
    expected = provider.embed("the full original body").astype("float32").tobytes()
    assert row["embedding"] == expected, (
        "unfade must re-embed the RESTORED content, not leave the fade-summary's vector"
    )
    store.close()


def test_hard_delete_evicts_matrix_entry(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails pre-fix: hard_delete dropped the row but never evicted the
    matrix entry, leaving a deleted memory's vector resurrectable in the
    warm cache."""
    from brain.memory.embedding_matrix import build_embedding_matrix

    _use_matrix_dim_fake_provider(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(content="to be deleted", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.embed_row(m.id, m.content)

    matrix = build_embedding_matrix(store.db_path)
    assert matrix.get(m.id) is not None  # sanity: present before delete

    store.hard_delete(m.id)

    assert matrix.get(m.id) is None, "hard_delete must evict the warm-matrix entry"
    store.close()


def test_embed_row_succeeds_even_if_warm_matrix_put_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX D (F1 #259 increment-3 red-team, F3): once the row's own UPDATE
    has committed, the row IS durably embedded — the DB is the source of
    truth, the matrix is a cache that self-heals on rebuild. A `put`
    failure AFTER that commit must be logged, never raised out of
    `embed_row`, and must never look like the embed itself failed."""
    from brain.memory import embedding_matrix as embedding_matrix_mod

    provider = _use_matrix_dim_fake_provider(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(
        content="content that gets embedded", memory_type="episodic", domain="chat", emotions={}
    )
    store.create(m)

    def _boom_put(self, memory_id: str, vector) -> None:  # noqa: ANN001, ARG001
        raise RuntimeError("simulated warm-matrix put failure")

    monkeypatch.setattr(embedding_matrix_mod.EmbeddingMatrix, "put", _boom_put)

    store.embed_row(m.id, m.content)  # must NOT raise

    row = store._conn.execute(
        "SELECT embedding, embedding_model_id FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["embedding"] is not None
    assert row["embedding_model_id"] == provider.model_id()
    store.close()


def test_reembed_or_clear_clears_row_and_evicts_matrix_on_embed_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the re-embed itself fails (provider/model error), the row's stale
    vector must be CLEARED (never left describing the OLD content) and the
    matrix entry evicted — a NULL row (picked up by the later idle backfill)
    is always safer than a vector silently describing stale content."""
    from brain.memory.embedding_matrix import build_embedding_matrix

    provider = _use_matrix_dim_fake_provider(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(content="original", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.embed_row(m.id, m.content)

    def _boom_embed(text: str):
        raise RuntimeError("simulated embed failure")

    monkeypatch.setattr(provider, "embed", _boom_embed)

    store.update(m.id, content="new content that cannot be embedded")

    row = store._conn.execute(
        "SELECT embedding, embedding_model_id FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["embedding"] is None
    assert row["embedding_model_id"] is None

    matrix = build_embedding_matrix(store.db_path)
    assert matrix.get(m.id) is None
    store.close()


# ---------------------------------------------------------------------------
# Increment-2 cold red-team FIX 1 (crash-window data integrity): the
# embedding/embedding_model_id columns must be NULLed in the SAME
# UPDATE/commit as the content change on update(content=...)/fade()/
# unfade() — not in the later, separate re-embed commit. Otherwise a crash
# between "content committed" and "re-embed committed" durably leaves
# {new content, OLD embedding}: a stale vector with no `embedding IS NULL`
# signal for the idle backfill to catch, so recall could keep surfacing the
# memory on its pre-mutation content indefinitely.
#
# Each test below patches the embedding provider's `embed()` so that, when
# it is called (necessarily AFTER the content UPDATE has already committed —
# `embed_row`/`_reembed_or_clear` run as a separate step following the
# content write), it FIRST captures the row's current embedding columns
# in-flight before letting the real embed proceed. This directly observes
# the durable intermediate state a crash at that instant would leave behind,
# while the happy path (no failure injected) still completes normally so the
# same test also proves the final row/matrix state is correct.
# ---------------------------------------------------------------------------


def _capture_embedding_mid_reembed(provider, store, memory_id: str, captured: dict):
    """Wrap `provider.embed` to snapshot memory_id's `embedding` /
    `embedding_model_id` columns into `captured` the instant it is called,
    then delegate to the original embed. Returns the original (unwrapped)
    embed callable so a test can independently recompute an "expected"
    vector without re-triggering the capture.
    """
    original_embed = provider.embed

    def _embed(text: str):
        row = store._conn.execute(
            "SELECT embedding, embedding_model_id FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        captured["embedding"] = row["embedding"]
        captured["embedding_model_id"] = row["embedding_model_id"]
        return original_embed(text)

    provider.embed = _embed
    return original_embed


def test_update_content_nulls_embedding_in_same_commit_before_reembed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fails pre-fix: pre-fix, the content UPDATE and the embedding-NULLing
    happened in two separate commits (content first, embedding cleared only
    on re-embed failure) — at the instant re-embed's `provider.embed` runs,
    the row still held the OLD content's vector, not NULL. Post-fix, the
    embedding/embedding_model_id columns are NULLed in the SAME commit as
    the content change, so the row is already NULL/backfill-eligible by
    the time re-embed even starts.
    """
    import numpy as np

    from brain.memory.embedding_matrix import build_embedding_matrix

    provider = _use_matrix_dim_fake_provider(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(content="old content", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.embed_row(m.id, m.content)
    old_embedding = store._conn.execute(
        "SELECT embedding FROM memories WHERE id = ?", (m.id,)
    ).fetchone()["embedding"]
    assert old_embedding is not None

    captured: dict = {}
    original_embed = _capture_embedding_mid_reembed(provider, store, m.id, captured)

    store.update(m.id, content="brand new content")

    assert captured["embedding"] is None, (
        "the row's embedding must already be NULL by the time re-embed runs, "
        "not the stale OLD vector"
    )
    assert captured["embedding_model_id"] is None

    # Happy path (no failure injected): the row ends up byte-equal to the
    # new content's embedding, and the matrix reflects it too.
    row = store._conn.execute("SELECT embedding FROM memories WHERE id = ?", (m.id,)).fetchone()
    expected = original_embed("brand new content").astype(np.float32).tobytes()
    assert row["embedding"] == expected
    matrix = build_embedding_matrix(store.db_path)
    np.testing.assert_array_equal(
        matrix.get(m.id), original_embed("brand new content").astype(np.float32)
    )
    store.close()


def test_fade_nulls_embedding_in_same_commit_before_reembed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same crash-window proof as above, for `fade()`."""
    import numpy as np

    from brain.memory.embedding_matrix import build_embedding_matrix

    provider = _use_matrix_dim_fake_provider(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(content="original long body", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.embed_row(m.id, m.content)
    old_embedding = store._conn.execute(
        "SELECT embedding FROM memories WHERE id = ?", (m.id,)
    ).fetchone()["embedding"]
    assert old_embedding is not None

    captured: dict = {}
    original_embed = _capture_embedding_mid_reembed(provider, store, m.id, captured)

    store.fade(m.id, summary="short summary")

    assert captured["embedding"] is None, (
        "the row's embedding must already be NULL by the time re-embed runs, "
        "not the stale pre-fade vector"
    )
    assert captured["embedding_model_id"] is None

    row = store._conn.execute("SELECT embedding FROM memories WHERE id = ?", (m.id,)).fetchone()
    expected = original_embed("short summary").astype(np.float32).tobytes()
    assert row["embedding"] == expected
    matrix = build_embedding_matrix(store.db_path)
    np.testing.assert_array_equal(
        matrix.get(m.id), original_embed("short summary").astype(np.float32)
    )
    store.close()


def test_unfade_nulls_embedding_in_same_commit_before_reembed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same crash-window proof as above, for `unfade()`."""
    import numpy as np

    from brain.memory.embedding_matrix import build_embedding_matrix

    provider = _use_matrix_dim_fake_provider(monkeypatch)
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(
        content="the full original body", memory_type="episodic", domain="chat", emotions={}
    )
    store.create(m)
    store.embed_row(m.id, m.content)
    store.fade(m.id, summary="short summary")  # row now embeds "short summary"
    old_embedding = store._conn.execute(
        "SELECT embedding FROM memories WHERE id = ?", (m.id,)
    ).fetchone()["embedding"]
    assert old_embedding is not None

    captured: dict = {}
    original_embed = _capture_embedding_mid_reembed(provider, store, m.id, captured)

    store.unfade(m.id)

    assert captured["embedding"] is None, (
        "the row's embedding must already be NULL by the time re-embed runs, "
        "not the stale fade-summary vector"
    )
    assert captured["embedding_model_id"] is None

    row = store._conn.execute("SELECT embedding FROM memories WHERE id = ?", (m.id,)).fetchone()
    expected = original_embed("the full original body").astype(np.float32).tobytes()
    assert row["embedding"] == expected
    matrix = build_embedding_matrix(store.db_path)
    np.testing.assert_array_equal(
        matrix.get(m.id), original_embed("the full original body").astype(np.float32)
    )
    store.close()


def test_get_bumps_last_accessed_at_and_recall_count() -> None:
    store = MemoryStore(":memory:")
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    # Before any get: recall_count=0, last_accessed_at=None
    row = store._conn.execute(
        "SELECT recall_count, last_accessed_at FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["recall_count"] == 0
    assert row["last_accessed_at"] is None
    # First get
    store.get(m.id)
    row = store._conn.execute(
        "SELECT recall_count, last_accessed_at FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["recall_count"] == 1
    assert row["last_accessed_at"] is not None
    # Second get
    store.get(m.id)
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == 2
    store.close()


def test_search_text_bumps_recall_count_for_each_hit() -> None:
    store = MemoryStore(":memory:")
    a = Memory.create_new(
        content="apple banana", memory_type="episodic", domain="chat", emotions={}
    )
    b = Memory.create_new(content="cherry pear", memory_type="episodic", domain="chat", emotions={})
    store.create(a)
    store.create(b)
    store.search_text("apple")  # hits only `a`
    row_a = store._conn.execute(
        "SELECT recall_count FROM memories WHERE id = ?", (a.id,)
    ).fetchone()
    row_b = store._conn.execute(
        "SELECT recall_count FROM memories WHERE id = ?", (b.id,)
    ).fetchone()
    assert row_a["recall_count"] == 1
    assert row_b["recall_count"] == 0  # not hit, not bumped
    store.close()


def test_get_of_unknown_id_does_not_create_phantom_row() -> None:
    store = MemoryStore(":memory:")
    assert store.get("mem_nonexistent") is None
    # No phantom row.
    assert store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    store.close()


def test_search_text_includes_fading_memories_by_default() -> None:
    store = MemoryStore(":memory:")
    m = Memory.create_new(
        content="apple unique-token", memory_type="episodic", domain="chat", emotions={}
    )
    store.create(m)
    store.fade(m.id, summary="apple summary")
    # Default include_fading=True
    hits = store.search_text("apple")
    assert len(hits) == 1
    assert hits[0].state == "fading"  # state field exposed on Memory dataclass (Task 1.5)
    store.close()


def test_search_text_excludes_fading_when_opted_out() -> None:
    store = MemoryStore(":memory:")
    m = Memory.create_new(
        content="apple unique-token", memory_type="episodic", domain="chat", emotions={}
    )
    store.create(m)
    store.fade(m.id, summary="apple summary")
    hits = store.search_text("apple", include_fading=False)
    assert hits == []
    store.close()


# ---------------------------------------------------------------------------
# Phase 1 review regression: mutations must not bump recall_count (Blocker)
# and double-fade must preserve original content_snapshot (Minor)
# ---------------------------------------------------------------------------


def test_fade_does_not_bump_recall_count():
    """Mutations are not recalls — fade/unfade/hard_delete must not inflate recall_count."""
    store = MemoryStore(":memory:")
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    # recall_count starts at 0
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == 0
    store.fade(m.id, summary="s")
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == 0  # fade is NOT a recall
    store.unfade(m.id)
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == 0  # unfade is NOT a recall either
    store.close()


def test_hard_delete_does_not_bump_recall_count_on_other_rows():
    """hard_delete's existence check must not touch any row's recall_count."""
    store = MemoryStore(":memory:")
    keeper = Memory.create_new(content="keeper", memory_type="episodic", domain="chat", emotions={})
    target = Memory.create_new(content="target", memory_type="episodic", domain="chat", emotions={})
    store.create(keeper)
    store.create(target)
    store.hard_delete(target.id)
    row = store._conn.execute(
        "SELECT recall_count FROM memories WHERE id = ?", (keeper.id,)
    ).fetchone()
    assert row["recall_count"] == 0
    store.close()


# ---------------------------------------------------------------------------
# ND-1 follow-up: update()/deactivate()'s internal existence-check must not
# bump recall_count either (closes the last leak — only a genuine full-read
# via get(bump=True, the default) counts as engagement).
# ---------------------------------------------------------------------------


def test_update_does_not_bump_recall_count():
    """update()'s internal existence-check must not inflate recall_count."""
    store = MemoryStore(":memory:")
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == 0
    store.update(m.id, content="modified")
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == 0  # a maintenance write is NOT a recall
    store.close()


def test_deactivate_does_not_bump_recall_count():
    """deactivate()'s internal existence-check must not inflate recall_count."""
    store = MemoryStore(":memory:")
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == 0
    store.deactivate(m.id)
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == 0  # deactivating is NOT a recall
    store.close()


def test_genuine_full_read_still_bumps_recall_count_after_nd1():
    """Regression: get()'s default path (bump=True) is untouched by the ND-1 fix
    — a deliberate full-read still bumps recall_count, even after update()/
    deactivate() calls that themselves must not bump it."""
    store = MemoryStore(":memory:")
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.update(m.id, content="modified")
    store.deactivate(m.id)
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == 0  # confirm the maintenance writes above didn't bump it
    restored = store.get(m.id)  # genuine full-read, default bump=True
    assert restored is not None
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == 1  # only the full-read counted
    store.close()


# ---------------------------------------------------------------------------
# recall-reinforcement: fractional recall_count bump (CHANGE 1, G4/G5/G14)
# ---------------------------------------------------------------------------


def test_bump_recall_accumulates_fractional_values():
    """G4: bump_recall stores and accumulates fractional amounts — two 0.8
    bumps read back as ~1.6 (float), not 1 or 2."""
    store = MemoryStore(":memory:")
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)

    store.bump_recall(m.id, 0.8)
    store.bump_recall(m.id, 0.8)

    restored = store.get(m.id, bump=False)
    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == pytest.approx(1.6)
    assert restored.recall_count == pytest.approx(1.6)  # round-trips through _row_to_memory too
    store.close()


# Pre-REAL personas (e.g. the v0.0.33 schema in tests/recovery/test_source_reader.py)
# declared recall_count with INTEGER column affinity. When such a DB is opened by
# MemoryStore, `CREATE TABLE IF NOT EXISTS` leaves the existing table alone and the
# column-migration only adds recall_count when absent, so the INTEGER affinity is
# preserved on the live table. This helper reproduces that exact starting state so
# G5 tests the real back-compat path, not a fresh REAL-affinity column.
def _mk_integer_affinity_store(db_path, *, seed_recall_count=3):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL,"
        " memory_type TEXT NOT NULL, domain TEXT NOT NULL, emotions_json TEXT NOT NULL,"
        " tags_json TEXT NOT NULL, importance REAL NOT NULL DEFAULT 0.0,"
        " score REAL NOT NULL DEFAULT 0.0, created_at TEXT NOT NULL, last_accessed_at TEXT,"
        " active INTEGER NOT NULL DEFAULT 1, protected INTEGER NOT NULL DEFAULT 0,"
        " metadata_json TEXT NOT NULL DEFAULT '{}', state TEXT NOT NULL DEFAULT 'active',"
        " content_snapshot TEXT, recall_count INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO memories (id, content, memory_type, domain, emotions_json, tags_json,"
        " created_at, recall_count) VALUES ('m1','x','episodic','chat','{}','[]',"
        " '2026-01-01T00:00:00+00:00', ?)",
        (seed_recall_count,),
    )
    conn.commit()
    conn.close()
    return MemoryStore(str(db_path))


def test_bump_recall_on_integer_affinity_column_accumulates_to_float(tmp_path):
    """G5 (back-compat): a persona whose recall_count column genuinely has
    INTEGER affinity (holding integer 3) accepts a subsequent fractional bump
    through the store's real UPDATE path and reads back as 3.8 (float) — no
    crash, no truncation to 3 or 4. Fails if SQLite were to truncate a
    fractional bump on an integer-affinity column."""
    store = _mk_integer_affinity_store(tmp_path / "memories.db")
    # Confirm we are actually testing INTEGER affinity, not a REAL column.
    affinity = {
        row["name"]: row["type"]
        for row in store._conn.execute("PRAGMA table_info(memories)")
    }["recall_count"]
    assert affinity == "INTEGER"

    store.bump_recall("m1", 0.8)

    row = store._conn.execute(
        "SELECT recall_count, typeof(recall_count) AS t FROM memories WHERE id = 'm1'"
    ).fetchone()
    assert row["recall_count"] == pytest.approx(3.8)
    assert row["t"] == "real"  # stored as a real, not truncated to an integer
    store.close()


def test_recall_count_bumps_from_different_sites_accumulate_additively():
    """G14 (no-lost-update): a full-read bump (+1.0, via get()) and a passive-
    recall fractional bump (+0.8, via bump_recall()) both land as a SUM — the
    second write does not clobber the first. Each bump site issues a single
    atomic `recall_count = recall_count + ?` UPDATE, so this must FAIL against
    a non-additive/overwrite implementation (e.g. one that read-then-wrote a
    computed total in Python)."""
    store = MemoryStore(":memory:")
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)

    store.get(m.id)  # +1.0
    store.bump_recall(m.id, 0.8)  # +0.8
    store.get(m.id)  # +1.0

    row = store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()
    assert row["recall_count"] == pytest.approx(2.8)
    store.close()


def test_double_fade_preserves_original_content_snapshot(caplog):
    """fade called twice must NOT overwrite the original snapshot."""
    import logging

    store = MemoryStore(":memory:")
    m = Memory.create_new(
        content="original detailed body", memory_type="episodic", domain="chat", emotions={}
    )
    store.create(m)
    store.fade(m.id, summary="first summary")
    with caplog.at_level(logging.WARNING, logger="brain.memory.store"):
        store.fade(m.id, summary="second summary")  # noop expected
    row = store._conn.execute(
        "SELECT content, content_snapshot FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["content_snapshot"] == "original detailed body"  # NOT "first summary"
    assert row["content"] == "first summary"  # unchanged by the no-op second fade
    assert "already in fading state" in caplog.text.lower() or "noop" in caplog.text.lower()
    store.close()


def test_list_since_iso_includes_fading_by_default():
    """list_since_iso returns active + fading memories with created_at > cutoff."""
    store = MemoryStore(":memory:")
    a = Memory.create_new(content="A", memory_type="episodic", domain="chat", emotions={})
    b = Memory.create_new(content="B", memory_type="episodic", domain="chat", emotions={})
    store.create(a)
    store.create(b)
    store.fade(b.id, summary="B (summary)")
    results = store.list_since_iso("2000-01-01T00:00:00+00:00")
    assert len(results) == 2
    states = {m.state for m in results}
    assert "fading" in states
    store.close()


def test_list_since_iso_excludes_fading_when_disabled():
    """include_fading=False filters out fading rows."""
    store = MemoryStore(":memory:")
    a = Memory.create_new(content="A", memory_type="episodic", domain="chat", emotions={})
    b = Memory.create_new(content="B", memory_type="episodic", domain="chat", emotions={})
    store.create(a)
    store.create(b)
    store.fade(b.id, summary="B (summary)")
    results = store.list_since_iso("2000-01-01T00:00:00+00:00", include_fading=False)
    assert len(results) == 1
    assert results[0].state == "active"
    store.close()


# ---------------------------------------------------------------------------
# v0.0.33 Track 3: monotone peak_emotion_intensity
# ---------------------------------------------------------------------------


def test_peak_captured_on_create(store):
    mem = Memory.create_new("loopy chased the ball", "conversation", "us",
                            emotions={"joy": 6.0, "warmth": 3.0})
    store.create(mem)
    got = store.get(mem.id)
    assert got.peak_emotion_intensity == 6.0


def test_peak_is_monotone_under_update(store):
    mem = Memory.create_new("m", "conversation", "us", emotions={"joy": 6.0})
    store.create(mem)
    # Decay-shaped write: lower intensities must NOT lower the peak.
    store.update(mem.id, emotions={"joy": 0.5})
    assert store.get(mem.id).peak_emotion_intensity == 6.0
    # A hotter later delta raises it.
    store.update(mem.id, emotions={"joy": 8.5})
    assert store.get(mem.id).peak_emotion_intensity == 8.5


def test_peak_survives_emotions_emptied(store):
    """The noise-floor deletion path writes an empty/reduced dict — the peak
    must survive it (the entire point of v0.0.33 Track 3)."""
    mem = Memory.create_new("m", "conversation", "us", emotions={"joy": 6.0})
    store.create(mem)
    store.update(mem.id, emotions={})
    got = store.get(mem.id)
    assert got.emotions == {}
    assert got.peak_emotion_intensity == 6.0


def test_peak_defaults_zero_without_emotions(store):
    mem = Memory.create_new("flat", "conversation", "us")
    store.create(mem)
    assert store.get(mem.id).peak_emotion_intensity == 0.0


def test_peak_migration_seeds_from_current_intensities(tmp_path):
    """Open a pre-v0.0.33 DB (no peak column) containing rows — reopening
    must add the column and seed peak = max(current intensities); empty or
    corrupt emotions seed 0.0 (honest zero — spec D4)."""
    db = tmp_path / "memories.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    # _SCHEMA now HAS the column — simulate the pre-v0.0.33 schema by
    # rebuilding the table without it:
    conn.executescript(
        """
        CREATE TABLE old_memories AS
            SELECT id, content, memory_type, domain, emotions_json, tags_json,
                   importance, score, created_at, last_accessed_at, active,
                   protected, metadata_json, state, content_snapshot, recall_count
            FROM memories;
        DROP TABLE memories;
        ALTER TABLE old_memories RENAME TO memories;
        """
    )
    now = datetime.now(UTC).isoformat()
    for mem_id, content, emotions in (
        ("a" * 36, "warm one", '{"joy": 4.5}'),
        ("b" * 36, "erased one", "{}"),
        ("c" * 36, "corrupt one", "not json"),
    ):
        conn.execute(
            "INSERT INTO memories (id, content, memory_type, domain, emotions_json,"
            " tags_json, importance, score, created_at, active, protected,"
            " metadata_json, state, recall_count)"
            " VALUES (?, ?, 'conversation', 'us', ?, '[]', 0, 0, ?, 1, 0, '{}', 'active', 0)",
            (mem_id, content, emotions, now),
        )
    conn.commit()
    conn.close()

    migrated = MemoryStore(db)  # reopen → column added + seeded
    assert migrated.get("a" * 36).peak_emotion_intensity == 4.5
    assert migrated.get("b" * 36).peak_emotion_intensity == 0.0
    # Corrupt-JSON row: _row_to_memory itself can't parse it, so verify the
    # migrated DB value directly — the seeding must have written honest 0.0.
    row = migrated._conn.execute(
        "SELECT peak_emotion_intensity FROM memories WHERE id = ?", ("c" * 36,)
    ).fetchone()
    assert row["peak_emotion_intensity"] == 0.0


def test_peak_migration_skips_seeding_on_pre_emotions_json_schema(tmp_path):
    """MemoryStore must NOT raise when opening a v0.0.12-shaped DB whose
    memories table uses the old ``emotions`` column (not ``emotions_json``).
    The peak seeding loop must be skipped fail-soft; the column is still added
    with honest DEFAULT 0.0.  Mirrors the roundtrip fixture's old schema."""
    db = tmp_path / "old_schema_memories.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT,
            importance INT,
            memory_type TEXT,
            domain TEXT,
            created_at TEXT,
            emotions TEXT,
            tags TEXT,
            active INT
        )"""
    )
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("id-old-001", "old memory content", 5, "conversation", "us", now, "{}", "[]", 1),
    )
    conn.commit()
    conn.close()

    # Opening must NOT raise — peak seeding must be skipped gracefully
    store = MemoryStore(db)
    try:
        # Column must exist (added by ALTER TABLE)
        cols = {row[1] for row in store._conn.execute("PRAGMA table_info(memories)").fetchall()}
        assert "peak_emotion_intensity" in cols, "peak_emotion_intensity column was not added"
        # The row must have an honest 0.0 (default, not an error)
        row = store._conn.execute(
            "SELECT peak_emotion_intensity FROM memories WHERE id = ?", ("id-old-001",)
        ).fetchone()
        assert row is not None
        assert row[0] == 0.0
    finally:
        store.close()


def test_deferred_d3_peak_is_scalar(store: MemoryStore) -> None:
    """Pin (D3): peak_emotion_intensity is one REAL scalar, not a per-emotion
    vector. A which-emotion-mattered use-case needs the deferred vector design —
    ledger: project_companion_emergence_deferred.md."""
    cols = {row[1]: row[2] for row in store._conn.execute(
        "PRAGMA table_info(memories)").fetchall()}
    assert cols["peak_emotion_intensity"] == "REAL"
    mem = Memory.create_new("m", "conversation", "us", emotions={"joy": 2.0})
    assert isinstance(mem.peak_emotion_intensity, float)


# ---------------------------------------------------------------------------
# F1 (#259) step 1: embedding/cluster columns + cluster_centroids table
# ---------------------------------------------------------------------------


def test_fresh_store_has_embedding_and_cluster_columns() -> None:
    """A brand-new store's `memories` table carries the 4 F1 columns,
    all nullable/no-default — existing rows land NULL."""
    store = MemoryStore(":memory:")
    cols = {row[1]: row for row in store._conn.execute("PRAGMA table_info(memories)").fetchall()}
    for name in ("embedding", "embedding_model_id", "cluster_id", "cluster_model_id"):
        assert name in cols, f"missing column: {name}"
        col = cols[name]
        # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
        assert col[3] == 0, f"{name} must be nullable (notnull=0), got {col[3]}"
        assert col[4] is None, f"{name} must have no default, got {col[4]!r}"
    store.close()


def test_fresh_store_has_cluster_centroids_table() -> None:
    """A brand-new store also creates the relocated `cluster_centroids`
    table (F1 moves it from the old MemoryClusterStore side file into
    memories.db) with the shape mirrored from
    `MemoryClusterStore.memory_cluster_centroids`."""
    store = MemoryStore(":memory:")
    tables = {
        row[0]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "cluster_centroids" in tables
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(cluster_centroids)").fetchall()}
    assert cols == {"model_id", "cluster_id", "centroid", "dim", "updated_at"}
    store.close()


def test_new_memory_row_has_null_embedding_and_cluster_fields(store: MemoryStore) -> None:
    """A freshly-created memory lands with the F1 columns NULL — nothing in
    this step writes them."""
    mem = Memory.create_new("plain content", "conversation", "us")
    store.create(mem)
    row = store._conn.execute(
        "SELECT embedding, embedding_model_id, cluster_id, cluster_model_id"
        " FROM memories WHERE id = ?",
        (mem.id,),
    ).fetchone()
    assert row["embedding"] is None
    assert row["embedding_model_id"] is None
    assert row["cluster_id"] is None
    assert row["cluster_model_id"] is None


def test_existing_store_migrates_in_embedding_and_cluster_columns(tmp_path) -> None:
    """Simulate a pre-F1 persona — manually create the OLD (pre-#259)
    schema (no embedding/cluster columns, no cluster_centroids table), then
    open MemoryStore: the 4 columns + the table must be added without
    error, and a pre-existing row must survive with NULL in all 4."""
    db_path = tmp_path / "memories.db"
    old_schema = """
    CREATE TABLE memories (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        domain TEXT NOT NULL,
        emotions_json TEXT NOT NULL,
        tags_json TEXT NOT NULL,
        importance REAL NOT NULL DEFAULT 0.0,
        score REAL NOT NULL DEFAULT 0.0,
        created_at TEXT NOT NULL,
        last_accessed_at TEXT,
        active INTEGER NOT NULL DEFAULT 1,
        protected INTEGER NOT NULL DEFAULT 0,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        state TEXT NOT NULL DEFAULT 'active',
        content_snapshot TEXT,
        recall_count REAL NOT NULL DEFAULT 0,
        peak_emotion_intensity REAL NOT NULL DEFAULT 0.0
    );
    """
    conn = sqlite3.connect(str(db_path))
    conn.executescript(old_schema)
    conn.execute(
        "INSERT INTO memories (id, content, memory_type, domain, emotions_json, tags_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("mem_pre_f1", "old body", "episodic", "chat", "{}", "[]", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db_path)
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(memories)").fetchall()}
    for name in ("embedding", "embedding_model_id", "cluster_id", "cluster_model_id"):
        assert name in cols
    tables = {
        row[0]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "cluster_centroids" in tables
    row = store._conn.execute(
        "SELECT embedding, embedding_model_id, cluster_id, cluster_model_id"
        " FROM memories WHERE id = ?",
        ("mem_pre_f1",),
    ).fetchone()
    assert tuple(row) == (None, None, None, None)
    store.close()

    # Re-open again (already-migrated DB) — the ALTER-guard must be a no-op,
    # not raise "duplicate column name".
    store2 = MemoryStore(db_path)
    store2.close()


# ---------------------------------------------------------------------------
# F1 (#259) increment 4: MemoryStore.set_cluster_memberships / get_cluster_id
# — the row/table-based successor to MemoryClusterStore.replace_pass
# (brain/memory/clustering.py), unit-tested directly here against small
# hand-built memberships/centroids (brain/memory/clustering.py's own tests
# cover the real k-means-driven end-to-end path).
# ---------------------------------------------------------------------------


def test_get_cluster_id_returns_none_for_unclustered_or_unknown_row(
    store: MemoryStore,
) -> None:
    m = _mem()
    store.create(m)
    assert store.get_cluster_id(m.id) is None  # exists, never clustered
    assert store.get_cluster_id("nonexistent-id") is None  # no such row


def test_set_cluster_memberships_writes_row_and_centroids(store: MemoryStore) -> None:
    m1, m2 = _mem(), _mem()
    store.create(m1)
    store.create(m2)
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])

    store.set_cluster_memberships({m1.id: 0, m2.id: 1}, centroids, model_id="model-a")

    assert store.get_cluster_id(m1.id) == (0, "model-a")
    assert store.get_cluster_id(m2.id) == (1, "model-a")
    rows = store._conn.execute(
        "SELECT cluster_id, dim FROM cluster_centroids WHERE model_id = ? ORDER BY cluster_id",
        ("model-a",),
    ).fetchall()
    assert [tuple(r) for r in rows] == [(0, 2), (1, 2)]


def test_set_cluster_memberships_is_scoped_to_model_id(store: MemoryStore) -> None:
    """A row tagged under one model_id must be invisible to a lookup that
    compares against a different model_id — mirrors the old
    MemoryClusterStore's own model-scoping invariant, ported onto the row."""
    m = _mem()
    store.create(m)
    store.set_cluster_memberships({m.id: 3}, np.zeros((4, 2)), model_id="model-old")

    cluster_id, cluster_model_id = store.get_cluster_id(m.id)
    assert cluster_id == 3
    assert cluster_model_id == "model-old"
    assert cluster_model_id != "model-new"  # the caller's own scoping check


def test_set_cluster_memberships_clears_rows_that_drop_out_of_the_pool(
    store: MemoryStore,
) -> None:
    """Wholesale-replace semantics: a memory id clustered by a PRIOR pass
    but absent from a LATER pass's memberships (same model_id) must end up
    with cluster_id NULL, never a stale tag pointing at a centroid the later
    pass may have deleted — the row-storage mirror of
    MemoryClusterStore.replace_pass's delete-then-reinsert symmetry."""
    m1, m2 = _mem(), _mem()
    store.create(m1)
    store.create(m2)
    store.set_cluster_memberships(
        {m1.id: 0, m2.id: 1}, np.array([[1.0, 0.0], [0.0, 1.0]]), model_id="m"
    )
    assert store.get_cluster_id(m1.id) == (0, "m")
    assert store.get_cluster_id(m2.id) == (1, "m")

    # Second pass: m2 has fallen out of the pool (e.g. its embedding was
    # evicted); only m1 remains, now the sole member of the sole cluster.
    store.set_cluster_memberships({m1.id: 0}, np.array([[1.0, 0.0]]), model_id="m")

    assert store.get_cluster_id(m1.id) == (0, "m")
    assert store.get_cluster_id(m2.id) is None  # dropped, not dangling
    n_centroids = store._conn.execute(
        "SELECT COUNT(*) FROM cluster_centroids WHERE model_id = ?", ("m",)
    ).fetchone()[0]
    assert n_centroids == 1


def test_set_cluster_memberships_only_clears_the_target_model_id(
    store: MemoryStore,
) -> None:
    """A pass for model_id "new" must not touch rows/centroids that belong
    to a DIFFERENT model_id "old" — the clear-before-reinsert step is scoped
    by model_id, not a blanket wipe."""
    m_old, m_new = _mem(), _mem()
    store.create(m_old)
    store.create(m_new)
    store.set_cluster_memberships({m_old.id: 0}, np.array([[1.0, 0.0]]), model_id="old")

    store.set_cluster_memberships({m_new.id: 0}, np.array([[0.0, 1.0]]), model_id="new")

    assert store.get_cluster_id(m_old.id) == (0, "old")  # untouched
    assert store.get_cluster_id(m_new.id) == (0, "new")
    n_old_centroids = store._conn.execute(
        "SELECT COUNT(*) FROM cluster_centroids WHERE model_id = ?", ("old",)
    ).fetchone()[0]
    assert n_old_centroids == 1


def test_set_cluster_memberships_is_atomic_a_failed_write_leaves_prior_state_intact(
    store: MemoryStore,
) -> None:
    """Simulates a crash mid-write: a bad centroid blows up AFTER the
    membership UPDATEs already ran, but every write shares ONE uncommitted
    transaction — rolling back after the failure must undo the membership
    writes too, leaving exactly the last successfully COMMITTED pass's
    state (mirrors MemoryClusterStore.replace_pass's own atomicity test)."""
    m = _mem()
    store.create(m)
    store.set_cluster_memberships({m.id: 0}, np.array([[1.0, 0.0]]), model_id="m")
    assert store.get_cluster_id(m.id) == (0, "m")

    bad_centroids = [None]  # blows up inside the centroid-write loop
    with pytest.raises(AttributeError):
        store.set_cluster_memberships({m.id: 1}, bad_centroids, model_id="m")
    store._conn.rollback()

    # The prior committed pass survives untouched — the failed pass never
    # landed (not even the membership half, despite it running first).
    assert store.get_cluster_id(m.id) == (0, "m")
