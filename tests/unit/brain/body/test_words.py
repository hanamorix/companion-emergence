"""Unit tests for brain/body/words.py — count_words_in_session helper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.body.words import count_words_in_session
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def _now() -> datetime:
    return datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC)


def _seed_assistant_turn(store: MemoryStore, *, content: str, age_hours: float) -> None:
    mem = Memory.create_new(
        memory_type="conversation",
        content=content,
        emotions={},
        domain="general",
        metadata={"speaker": "assistant"},
    )
    mid = store.create(mem)
    backdated = _now() - timedelta(hours=age_hours)
    store._conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        (backdated.isoformat(), mid),
    )
    store._conn.commit()


def _seed_user_turn(store: MemoryStore, *, content: str, age_hours: float) -> None:
    mem = Memory.create_new(
        memory_type="conversation",
        content=content,
        emotions={},
        domain="general",
        metadata={"speaker": "user"},
    )
    mid = store.create(mem)
    backdated = _now() - timedelta(hours=age_hours)
    store._conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        (backdated.isoformat(), mid),
    )
    store._conn.commit()


def test_empty_store_returns_zero(store, tmp_path):
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 0


def test_only_assistant_turns_counted(store, tmp_path):
    _seed_assistant_turn(store, content="one two three four", age_hours=0.5)
    _seed_user_turn(store, content="five six seven eight nine", age_hours=0.5)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 4  # assistant only


def test_window_filter_excludes_old_turns(store, tmp_path):
    # 0.5h ago — inside session window of 2h
    _seed_assistant_turn(store, content="recent words count here", age_hours=0.5)
    # 5h ago — outside
    _seed_assistant_turn(store, content="old turn does not count", age_hours=5.0)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 4


def test_session_hours_zero_falls_back_to_one_hour(store, tmp_path):
    """When CLI mode (no bridge), session_hours=0.0; fall back to 1h window."""
    _seed_assistant_turn(store, content="should count", age_hours=0.5)
    _seed_assistant_turn(store, content="should not count this old turn", age_hours=2.0)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=0.0, now=_now(),
    )
    assert n == 2  # only "should count"


def test_returns_zero_on_store_exception(store, tmp_path, monkeypatch):
    """Fail-safe per spec §3.2 + §7.3 — never propagates."""
    def boom(*a, **k):
        raise RuntimeError("simulated db failure")
    monkeypatch.setattr(store, "list_by_type", boom)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 0
