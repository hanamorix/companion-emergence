"""One thing reads a memory's feelings (Engine 7 ONE4, items 1 and 3).

`Memory`'s stored emotions, score and peak are private (`_stored_*`); a stored row is rebuilt through `Memory.from_stored_row`; the
consolidation merge is one store method; and the readers outside the engine and the store ask for `memory_feelings`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.emotion.engine_utils import memory_feelings
from brain.forgetting import graveyard
from brain.forgetting.salience import SalienceInputs, _emotion_input, _peak_input
from brain.memory.store import Memory, MemoryStore, _merge_emotions
from brain.tools.impls._common import _mem_to_result

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _stored(**overrides) -> Memory:
    fields = {
        "id": "row-1",
        "content": "a stored row",
        "memory_type": "conversation",
        "domain": "us",
        "created_at": NOW,
        "emotions": {"joy": 4.0, "grief": 2.0},
        "importance": 3.0,
        "score": 6.0,
        "peak_emotion_intensity": 7.5,
    }
    fields.update(overrides)
    return Memory.from_stored_row(**fields)


def test_from_stored_row_sets_the_private_fields_from_the_public_names():
    memory = _stored(state="fading", recall_count=2.0, tags=["a"], metadata={"k": 1})
    assert (memory._stored_emotions, memory._stored_score, memory._stored_peak) == ({"joy": 4.0, "grief": 2.0}, 6.0, 7.5)
    assert (memory.state, memory.recall_count, memory.tags, memory.metadata, memory.importance) == ("fading", 2.0, ["a"], {"k": 1}, 3.0)


def test_from_stored_row_defaults_the_optional_values_as_the_dataclass_does():
    memory = Memory.from_stored_row(id="r", content="c", memory_type="meta", domain="us", created_at=NOW)
    assert (memory._stored_emotions, memory._stored_score, memory._stored_peak) == ({}, 0.0, 0.0)
    assert (memory.tags, memory.metadata, memory.state, memory.active, memory.importance) == ([], {}, "active", True, 0.0)


def test_a_memory_no_longer_has_the_public_names():
    memory = _stored()
    for old in ("emotions", "score", "peak_emotion_intensity"):
        assert not hasattr(memory, old)
    with pytest.raises(TypeError):
        Memory(id="r", content="c", memory_type="meta", domain="us", created_at=NOW, emotions={"joy": 1.0})  # type: ignore[call-arg]


def test_to_dict_and_from_dict_keep_the_public_dict_keys():
    memory = _stored()
    data = memory.to_dict()
    assert data["emotions"] == {"joy": 4.0, "grief": 2.0} and data["score"] == 6.0
    back = Memory.from_dict(data)
    assert (back._stored_emotions, back._stored_score) == ({"joy": 4.0, "grief": 2.0}, 6.0)


def test_a_row_read_back_from_the_database_carries_its_stored_values(tmp_path: Path):
    store = MemoryStore(tmp_path / "memories.db")
    try:
        memory = Memory.create_new(content="x", memory_type="conversation", domain="us", emotions={"joy": 4.0, "grief": 2.0})
        store.create(memory)
        got = store.get(memory.id, bump=False)
        assert (got._stored_emotions, got._stored_score, got._stored_peak) == ({"joy": 4.0, "grief": 2.0}, 6.0, 4.0)
    finally:
        store.close()


def test_merge_into_writes_content_and_merged_feelings_in_one_update(tmp_path: Path, monkeypatch):
    store = MemoryStore(tmp_path / "memories.db")
    try:
        target = Memory.create_new(content="a", memory_type="conversation", domain="us", emotions={"Joy": 5.0})
        candidate = Memory.create_new(content="b", memory_type="dream", domain="us", emotions={"joy": 7.0, "awe": 2.0})
        store.create(target, fold=False)
        calls: list[tuple] = []
        real_update = MemoryStore.update
        monkeypatch.setattr(MemoryStore, "update", lambda self, memory_id, **fields: (calls.append((memory_id, fields)), real_update(self, memory_id, **fields))[1])
        store.merge_into(target, candidate, "a and b", engine_live=True)
        assert calls == [(target.id, {"content": "a and b", "emotions": {"joy": 6.0, "awe": 2.0}, "fold": False})]
        got = store.get(target.id, bump=False)
        assert got.content == "a and b" and got._stored_emotions == {"joy": 6.0, "awe": 2.0}
    finally:
        store.close()


def test_merge_into_before_the_switch_writes_the_content_only(tmp_path: Path):
    store = MemoryStore(tmp_path / "memories.db")
    try:
        target = Memory.create_new(content="a", memory_type="conversation", domain="us", emotions={"Joy": 5.0})
        candidate = Memory.create_new(content="b", memory_type="dream", domain="us", emotions={"joy": 7.0})
        store.create(target, fold=False)
        store.merge_into(target, candidate, "a and b", engine_live=False)
        got = store.get(target.id, bump=False)
        assert got.content == "a and b" and got._stored_emotions == {"Joy": 5.0}
    finally:
        store.close()


def test_merge_into_raises_key_error_for_a_missing_target(tmp_path: Path):
    store = MemoryStore(tmp_path / "memories.db")
    try:
        ghost = Memory.create_new(content="a", memory_type="conversation", domain="us", emotions={"joy": 5.0})
        with pytest.raises(KeyError):
            store.merge_into(ghost, ghost, "x", engine_live=True)
    finally:
        store.close()


def test_merge_emotions_lives_in_the_store():
    assert _merge_emotions({"trust": 4, "joy": 7}, {"awe": 6, "Joy": 5}) == {"trust": 4.0, "joy": 6.0, "awe": 6.0}


def test_the_backfill_probe_and_emptiness_check_are_store_methods(tmp_path: Path):
    store = MemoryStore(tmp_path / "memories.db")
    try:
        assert store.has_active_without_emotions() is False  # no rows
        with_feelings = Memory.create_new(content="a", memory_type="conversation", domain="us", emotions={"joy": 4.0})
        store.create(with_feelings, fold=False)
        assert store.has_active_without_emotions() is False
        assert store.has_stored_emotions(with_feelings) is True
        bare = Memory.create_new(content="b", memory_type="conversation", domain="us")
        store.create(bare, fold=False)
        assert store.has_active_without_emotions() is True
        assert store.has_stored_emotions(bare) is False
    finally:
        store.close()


def test_the_stub_engine_offers_the_read_and_holds_a_faded_value(emotion_engine_stub):
    memory = _stored(emotions={"joy": 4.0, "grief": -3.0})
    emotion_engine_stub.faded = {(memory.id, "joy"): 14.0}
    assert emotion_engine_stub.faded_value(memory, "joy") == 10.0
    assert emotion_engine_stub.faded_value(memory, "grief") == 0.0
    assert emotion_engine_stub.memory_feelings(memory).feelings == {"joy": 10.0, "grief": 0.0}


def test_the_memory_view_carries_held_canonical_feelings_with_an_engine_and_without(tmp_path: Path, emotion_engine_stub):
    memory = _stored(emotions={"Joy": 12.0, "grief": -2.0})
    assert _mem_to_result(memory, tmp_path)["emotions"] == {"joy": 10.0, "grief": 0.0}
    assert _mem_to_result(memory, None)["emotions"] == {"joy": 10.0, "grief": 0.0}


def test_the_graveyard_entry_carries_the_read_feelings(tmp_path: Path, emotion_engine_stub):
    memory = _stored(emotions={"Joy": 12.0, "grief": -2.0})
    inputs = SalienceInputs(emotion=0.0, recall=0.0, hebbian=0.0, soul=0.0, freshness=0.0)
    graveyard.append(tmp_path, memory=memory, salience_at_drop=0.05, inputs=inputs, lived_age_hours=1.0, reason="test")
    assert graveyard.read_all(tmp_path)[0]["emotion_at_ingest"] == {"joy": 10.0, "grief": 0.0}


def test_salience_without_an_engine_reads_the_pure_read_and_the_peak_is_held():
    memory = _stored(emotions={"joy": -4.0, "grief": 3.0}, peak_emotion_intensity=-5.0)
    assert _emotion_input(memory, None) == pytest.approx(0.3)
    assert _peak_input(memory, None) == 0.0  # a stored negative peak reads 0, so the peak input is 0


def test_the_consolidation_write_back_writes_the_read_feelings(tmp_path: Path, emotion_engine_stub):
    from brain.engines import consolidation

    emotion_engine_stub.live = True
    emotion_engine_stub.faded = {("w", "joy"): 6.0}
    store = MemoryStore(tmp_path / "memories.db")
    try:
        row = _stored(id="w", emotions={"joy": 9.0}, importance=3.0)
        store.create(row, fold=False)
        consolidation._write_back_faded_emotions(store, tmp_path, row)
        assert store.get("w", bump=False)._stored_emotions == {"joy": 6.0}
    finally:
        store.close()


def test_stored_emotion_names_returns_text_names_only():
    memory = _stored(emotions={"Joy": 1.0, "joy": 2.0, 3: 4.0})  # type: ignore[dict-item]
    assert MemoryStore.stored_emotion_names(memory) == ["Joy", "joy"]
    assert MemoryStore.stored_emotion_names(_stored(emotions={})) == []


def test_the_write_back_skips_a_row_when_the_read_leaves_a_stored_name_out(tmp_path: Path, emotion_engine_stub, monkeypatch):
    from types import SimpleNamespace

    from brain.engines import consolidation

    emotion_engine_stub.live = True
    store = MemoryStore(tmp_path / "memories.db")
    try:
        row = _stored(id="w", emotions={"joy": 9.0, "trust": 5.0}, importance=3.0)
        store.create(row, fold=False)
        # the read drops "trust" (a value that is not a finite number is dropped by the real read)
        monkeypatch.setattr(emotion_engine_stub, "memory_feelings", lambda m: SimpleNamespace(feelings={"joy": 2.0}))
        before = store.get("w", bump=False)
        consolidation._write_back_faded_emotions(store, tmp_path, row)
        after = store.get("w", bump=False)
        assert after._stored_emotions == {"joy": 9.0, "trust": 5.0}  # stored values stay
        assert after.created_at == before.created_at  # and the row is not touched at all
    finally:
        store.close()


@pytest.mark.no_base_failure(
    reason="pins that the new skip guard does not fire on a pooled two-spelling row; the unguarded write-back already writes it"
)
def test_the_write_back_writes_a_row_whose_two_spellings_the_read_pools_into_one(tmp_path: Path, emotion_engine_stub):
    from brain.engines import consolidation

    emotion_engine_stub.live = True
    store = MemoryStore(tmp_path / "memories.db")
    try:
        row = _stored(id="w", emotions={"Joy": 4.0, "joy": 6.0}, importance=3.0)
        store.create(row, fold=False)
        assert len(store.get("w", bump=False)._stored_emotions) == 2  # both spellings are stored
        consolidation._write_back_faded_emotions(store, tmp_path, row)
        assert store.get("w", bump=False)._stored_emotions == {"joy": 5.0}  # pooled as their average, written back
    finally:
        store.close()


def test_the_pure_read_is_what_every_engineless_reader_gets():
    memory = _stored(emotions={"joy": 4.0, "Joy": 2.0}, score=6.0, peak_emotion_intensity=4.0)
    read = memory_feelings(memory, None, None, None)
    assert (read.feelings, read.peak, read.score) == ({"joy": 3.0}, 4.0, 6.0)
