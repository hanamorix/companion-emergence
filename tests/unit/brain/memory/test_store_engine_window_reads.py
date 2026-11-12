"""The emotion engine's three read-only window methods of `MemoryStore` (Engine 4, #579).

`list_engine_window(since)`, `creation_stamps_since(since)` and `oldest_created_at()` are plain SELECTs: they bump nothing
(`recall_count`, `last_accessed_at`), fire no hook, and one row whose columns cannot be parsed is logged by id and skipped.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from brain.memory.store import Memory, MemoryStore

T0 = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)


def _memory(
    created: datetime | str,
    *,
    accessed: datetime | str | None = None,
    memory_type: str = "conversation",
    mid: str | None = None,
    active: bool = True,
) -> Memory:
    memory = Memory.from_stored_row(
        id=mid or str(uuid.uuid4()),
        content="a window test memory",
        memory_type=memory_type,
        domain="us",
        created_at=created if isinstance(created, datetime) else datetime.fromisoformat(created),
        emotions={"joy": 5.0},
        tags=[],
        importance=1.0,
        score=5.0,
        metadata={},
        peak_emotion_intensity=5.0,
        last_accessed_at=accessed if isinstance(accessed, datetime) or accessed is None else datetime.fromisoformat(accessed),
        active=active,
    )
    return memory


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    opened = MemoryStore(tmp_path / "memories.db")
    yield opened
    opened.close()


def _put(store: MemoryStore, memory: Memory) -> Memory:
    store.create(memory, fold=False)
    return memory


def _raw_set(store: MemoryStore, memory_id: str, column: str, value: str) -> None:
    store._conn.execute(f"UPDATE memories SET {column} = ? WHERE id = ?", (value, memory_id))  # noqa: S608
    store._conn.commit()


def _snapshot(store: MemoryStore) -> list[tuple]:
    return [
        tuple(row)
        for row in store._conn.execute(
            "SELECT id, recall_count, last_accessed_at, emotions_json FROM memories ORDER BY id"
        ).fetchall()
    ]


# --- list_engine_window ---------------------------------------------------------------------------


def test_the_window_returns_rows_created_or_accessed_on_or_after_since_of_any_type_and_state(store: MemoryStore) -> None:
    created_in = _put(store, _memory(T0 + timedelta(hours=1)))
    accessed_in = _put(store, _memory(T0 - timedelta(days=5), accessed=T0 + timedelta(hours=2)))
    at_the_boundary = _put(store, _memory(T0))
    accessed_at_the_boundary = _put(store, _memory(T0 - timedelta(days=3), accessed=T0))
    other_type = _put(store, _memory(T0 + timedelta(hours=3), memory_type="making"))
    inactive = _put(store, _memory(T0 + timedelta(hours=4), active=False))
    _put(store, _memory(T0 - timedelta(days=5)))
    _put(store, _memory(T0 - timedelta(days=5), accessed=T0 - timedelta(seconds=1)))
    ids = {m.id for m in store.list_engine_window(T0)}
    assert ids == {
        created_in.id,
        accessed_in.id,
        at_the_boundary.id,
        accessed_at_the_boundary.id,
        other_type.id,
        inactive.id,
    }


def test_the_window_gives_each_row_its_columns_as_a_memory(store: MemoryStore) -> None:
    row = _put(store, _memory(T0 + timedelta(hours=1), accessed=T0 + timedelta(hours=2), mid="row-1"))
    (got,) = store.list_engine_window(T0)
    assert (got.id, got.memory_type, got._stored_emotions, got.created_at, got.last_accessed_at) == (
        "row-1",
        "conversation",
        {"joy": 5.0},
        row.created_at,
        row.last_accessed_at,
    )


def test_the_window_compares_parsed_times_not_text(store: MemoryStore) -> None:
    """A row stored with a -05:00 offset: its text sorts before the bound's text, its instant is inside the window."""
    local = timezone(timedelta(hours=-5))
    inside = _put(store, _memory(datetime(2026, 6, 10, 7, 30, tzinfo=local)))  # 12:30 UTC, after T0 = 12:00 UTC
    outside = _put(store, _memory(datetime(2026, 6, 10, 6, 30, tzinfo=local)))  # 11:30 UTC, before T0
    ids = {m.id for m in store.list_engine_window(T0)}
    assert inside.id in ids and outside.id not in ids


def test_the_window_accepts_a_since_with_any_offset(store: MemoryStore) -> None:
    row = _put(store, _memory(T0 + timedelta(hours=1)))
    plus_two = timezone(timedelta(hours=2))
    assert [m.id for m in store.list_engine_window(T0.astimezone(plus_two))] == [row.id]


def test_the_window_bumps_nothing_and_fires_no_hook(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _put(store, _memory(T0 + timedelta(hours=1)))
    _put(store, _memory(T0 - timedelta(days=2), accessed=T0 + timedelta(hours=2)))
    called: list[str] = []
    monkeypatch.setattr(MemoryStore, "emotion_engine", lambda self: called.append("engine") or None)
    before = _snapshot(store)
    store.list_engine_window(T0)
    store.creation_stamps_since(T0)
    store.oldest_created_at()
    assert _snapshot(store) == before
    assert called == []


def test_a_row_that_cannot_be_parsed_is_logged_by_id_and_skipped_the_others_are_returned(
    store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    good = _put(store, _memory(T0 + timedelta(hours=1)))
    bad_json = _put(store, _memory(T0 + timedelta(hours=2), mid="bad-json"))
    bad_time = _put(store, _memory(T0 + timedelta(hours=3), mid="bad-time"))
    _raw_set(store, bad_json.id, "emotions_json", "{not json")
    _raw_set(store, bad_time.id, "created_at", "2026-06-10T25:99:00")
    with caplog.at_level(logging.WARNING):
        got = store.list_engine_window(T0)
    assert [m.id for m in got] == [good.id]
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "bad-json" in text and "bad-time" in text


def test_the_window_rejects_a_since_that_is_not_a_time(store: MemoryStore) -> None:
    with pytest.raises(ValueError):
        store.list_engine_window("yesterday")  # type: ignore[arg-type]


# --- creation_stamps_since --------------------------------------------------------------------------


def test_the_stamps_are_the_creation_times_of_every_row_created_on_or_after_since(store: MemoryStore) -> None:
    _put(store, _memory(T0 - timedelta(days=1)))
    _put(store, _memory(T0 - timedelta(days=4), accessed=T0 + timedelta(hours=1)))  # accessed only: no stamp
    a = _put(store, _memory(T0))
    b = _put(store, _memory(T0 + timedelta(hours=2), memory_type="making"))
    c = _put(store, _memory(T0 + timedelta(hours=1), active=False))
    assert store.creation_stamps_since(T0) == [a.created_at, c.created_at, b.created_at]


def test_the_stamps_compare_parsed_times_and_skip_a_row_that_cannot_be_parsed(
    store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    local = timezone(timedelta(hours=-5))
    inside = _put(store, _memory(datetime(2026, 6, 10, 7, 30, tzinfo=local)))
    _put(store, _memory(datetime(2026, 6, 10, 6, 30, tzinfo=local)))
    broken = _put(store, _memory(T0 + timedelta(hours=1), mid="stamp-bad"))
    _raw_set(store, broken.id, "created_at", "not a time")
    with caplog.at_level(logging.WARNING):
        stamps = store.creation_stamps_since(T0)
    assert stamps == [inside.created_at]
    assert "stamp-bad" in " ".join(r.getMessage() for r in caplog.records)


# --- oldest_created_at -------------------------------------------------------------------------------


def test_the_oldest_created_at_is_the_minimum_over_every_row_of_any_type_and_state(store: MemoryStore) -> None:
    assert store.oldest_created_at() is None
    _put(store, _memory(T0))
    oldest = _put(store, _memory(T0 - timedelta(days=40), memory_type="making", active=False))
    _put(store, _memory(T0 - timedelta(days=2), memory_type="research"))
    assert store.oldest_created_at() == oldest.created_at


def test_an_unparseable_oldest_value_is_logged_and_answers_none(store: MemoryStore, caplog: pytest.LogCaptureFixture) -> None:
    broken = _put(store, _memory(T0))
    _raw_set(store, broken.id, "created_at", "")
    with caplog.at_level(logging.WARNING):
        assert store.oldest_created_at() is None
    assert caplog.records
