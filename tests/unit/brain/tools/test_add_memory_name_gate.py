"""add_memory's write gate scores the feelings the name check keeps (#741, #801 step 12).

The store's check (`MemoryStore.create` -> `checked`) drops a feeling name her vocabulary cannot hold. The gate used to score
the raw map first, so a memory could pass it, or take its stored importance, from a feeling that was then thrown away.

Synthetic stores only: an in-memory store (its registry answers, no persona folder) and a tmp persona folder with the bridge faked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.emotion import vocabulary_access
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.impls.add_memory import add_memory

UNKNOWN = "zzq_unregistered_feeling"


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def _add(store: MemoryStore, emotions: dict, importance: int | None = None, persona_dir: Path | None = None) -> dict:
    return add_memory(
        content="Bob told Canary about his week",
        memory_type="conversation",
        domain="us",
        emotions=emotions,
        importance=importance,
        store=store,
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=persona_dir or Path("."),
    )


def _only_row(store: MemoryStore):
    rows = store.list_active()
    assert len(rows) == 1
    return rows[0]


def test_a_memory_whose_only_strong_feeling_is_discarded_is_refused(store: MemoryStore) -> None:
    result = _add(store, {UNKNOWN: 16})
    assert result["created"] is False
    assert "below threshold" in result["reason"]
    assert "emotion_score=0" in result["reason"]
    assert store.count() == 0


def test_a_discarded_feeling_with_a_low_importance_given_is_refused(store: MemoryStore) -> None:
    result = _add(store, {UNKNOWN: 20}, importance=3)
    assert result["created"] is False
    assert store.count() == 0


def test_a_discarded_feeling_with_importance_seven_given_stores_only_the_kept_feelings(store: MemoryStore) -> None:
    result = _add(store, {"joy": 4, UNKNOWN: 16}, importance=7)
    assert result["created"] is True
    row = _only_row(store)
    assert row._stored_emotions == {"joy": 4.0}
    assert row._stored_score == pytest.approx(4.0)
    assert row._stored_peak == pytest.approx(4.0)
    assert row.importance == 7


def test_the_gate_scores_only_the_kept_value(store: MemoryStore) -> None:
    """Kept 10 and discarded 9 sum to 19 raw (passes); the kept 10 alone scores importance 5 and is refused."""
    assert _add(store, {"love": 10, UNKNOWN: 9})["created"] is False
    assert store.count() == 0


def test_the_stored_importance_comes_from_the_kept_score(store: MemoryStore) -> None:
    """Kept 20 and discarded 15 sum to 35 raw (importance 9); the kept 20 alone is importance 7."""
    result = _add(store, {"love": 20, UNKNOWN: 15})
    assert result["created"] is True
    assert result["importance"] == 7
    row = _only_row(store)
    assert row.importance == 7
    assert row.metadata["importance"] == 7
    assert row._stored_emotions == {"love": 20.0}


def test_two_spellings_of_one_feeling_are_scored_as_their_mean(store: MemoryStore) -> None:
    """`Joy` 20 and `joy` 10 average to 15 (importance band 5); the raw sum 30 would have given 9."""
    result = _add(store, {"Joy": 20, "joy": 10})
    assert result["created"] is True
    assert result["importance"] == 5
    row = _only_row(store)
    assert row._stored_emotions == {"joy": 15.0}
    assert row.importance == 5


def test_a_memory_with_only_kept_feelings_is_unchanged(store: MemoryStore) -> None:
    result = _add(store, {"love": 9, "joy": 8})
    assert result["created"] is True
    assert result["importance"] == 5
    assert _only_row(store)._stored_emotions == {"love": 9.0, "joy": 8.0}


def test_the_check_runs_once_per_add_memory(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    real = vocabulary_access.check_feelings

    def spy(persona_dir, raw):
        calls.append(dict(raw))
        return real(persona_dir, raw)

    monkeypatch.setattr(vocabulary_access, "check_feelings", spy)
    assert _add(store, {"love": 9, "joy": 8})["created"] is True
    assert len(calls) == 1


def test_a_name_the_bridge_refuses_is_not_scored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a bridge up, a name the gate refuses comes back `refused` for the one add call and is dropped before the gate."""
    asked: list[dict] = []

    def fake_post(persona_dir, route, body, *args, **kwargs):
        asked.append(body)
        return {"results": [{"name": item["name"], "spelling": None, "outcome": "refused", "changed": False} for item in body["items"]]}

    monkeypatch.setattr(vocabulary_access, "post_to_bridge", fake_post)
    vocabulary_access._ADD_RETRY_AFTER.clear()
    persona_store = MemoryStore(tmp_path / "memories.db")
    try:
        result = _add(persona_store, {UNKNOWN: 16}, persona_dir=tmp_path)
        assert result["created"] is False
        assert persona_store.count() == 0
    finally:
        persona_store.close()
    assert len(asked) == 1
