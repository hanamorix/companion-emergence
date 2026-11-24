"""The check on a memory's feeling names at the write (#801 section 6): `MemoryStore.create`, `update`, the grief touch and
`pending.route_write` run it before the write transaction; a merge, a restore and an import keep the names they hold.

Synthetic persona folders in tmp, no bridge (the bridge is faked as unreachable).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.emotion import vocabulary, vocabulary_access
from brain.memory.pending import PendingQueue, route_write
from brain.memory.store import Memory, MemoryStore

FILE = "emotion_vocabulary.json"


@pytest.fixture(autouse=True)
def _no_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vocabulary_access, "post_to_bridge", lambda *a, **k: {"error": "bridge unreachable"})
    vocabulary_access._ADD_RETRY_AFTER.clear()
    vocabulary_access._READ_ONLY_LOADED.clear()


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def _memory(emotions: dict, *, memory_type: str = "conversation", **kwargs) -> Memory:
    return Memory.create_new(content="Bob told Canary about his week", memory_type=memory_type, domain="us", emotions=emotions, **kwargs)


def _stored(store: MemoryStore, memory_id: str) -> Memory:
    got = store.get(memory_id, bump=False)
    assert got is not None
    return got


# --- create --------------------------------------------------------------------------------------------------------------


def test_a_memory_written_with_Joy_is_stored_and_read_back_as_joy(store: MemoryStore) -> None:
    """Spec criterion 6."""
    memory = _memory({"Joy": 6.0})
    store.create(memory)
    assert _stored(store, memory.id)._stored_emotions == {"joy": 6.0}


def test_two_spellings_on_one_memory_are_stored_as_their_mean_and_the_peak_and_score_follow(store: MemoryStore) -> None:
    memory = _memory({"Joy": 4.0, "joy": 8.0, "love": 2.0})
    store.create(memory)
    got = _stored(store, memory.id)
    assert got._stored_emotions == {"joy": 6.0, "love": 2.0}
    assert got._stored_score == pytest.approx(8.0)
    assert got._stored_peak == pytest.approx(6.0)


def test_a_name_her_vocabulary_cannot_hold_is_not_stored_and_leaves_no_peak_or_score(store: MemoryStore) -> None:
    memory = _memory({"joy": 2.0, "never_seen_zz": 9.0, "---": 5.0})
    store.create(memory)
    got = _stored(store, memory.id)
    assert got._stored_emotions == {"joy": 2.0}
    assert got._stored_peak == pytest.approx(2.0) and got._stored_score == pytest.approx(2.0)
    row = store._conn.execute("SELECT peak_emotion_intensity FROM memories WHERE id = ?", (memory.id,)).fetchone()
    assert row[0] == pytest.approx(2.0)  # the peak column was worked out from the checked names


def test_a_memory_the_check_does_not_change_is_written_exactly_as_given(store: MemoryStore) -> None:
    memory = _memory({"joy": 2.0}, importance=7.0)
    memory._stored_score = 99.0  # a stored score that is not the sum is kept when nothing changed
    assert store.checked(memory) is memory
    store.create(memory)
    assert _stored(store, memory.id)._stored_score == 99.0


def test_in_the_owner_a_new_name_that_passes_the_gate_is_added_and_stored(tmp_path: Path, store: MemoryStore) -> None:
    vocabulary.set_bridge_owner(True)  # tests/conftest.py puts the flag back
    memory = _memory({"Brand New Word": 5.0})
    store.create(memory)
    assert _stored(store, memory.id)._stored_emotions == {"brand_new_word": 5.0}
    assert [e["name"] for e in json.loads((tmp_path / FILE).read_text(encoding="utf-8"))["emotions"]] == ["brand_new_word"]


def test_check_names_false_keeps_the_names_as_given(store: MemoryStore) -> None:
    memory = _memory({"Joy": 6.0, "never_seen_zz": 9.0})
    store.create(memory, check_names=False)
    assert _stored(store, memory.id)._stored_emotions == {"Joy": 6.0, "never_seen_zz": 9.0}


def test_an_in_memory_store_checks_against_this_process_registry(register_feelings) -> None:
    register_feelings("kept_here")
    store = MemoryStore(":memory:")
    try:
        memory = _memory({"Kept Here": 3.0, "not_here_zz": 4.0})
        store.create(memory)
        assert _stored(store, memory.id)._stored_emotions == {"kept_here": 3.0}
    finally:
        store.close()


def test_the_engine_is_told_the_checked_memory(store: MemoryStore, emotion_engine_stub) -> None:
    memory = _memory({"Joy": 6.0, "never_seen_zz": 9.0})
    store.create(memory)
    committed = [c for c in emotion_engine_stub.recorded_calls if c[0] == "on_memory_committed"]
    assert len(committed) == 1 and committed[0][1]._stored_emotions == {"joy": 6.0}


# --- update --------------------------------------------------------------------------------------------------------------


def test_update_checks_the_emotions_and_the_peak_follows_the_checked_map(store: MemoryStore) -> None:
    row = _memory({})
    store.create(row)
    store.update(row.id, emotions={"Joy": 4.0, "joy": 6.0, "never_seen_zz": 9.0})
    got = _stored(store, row.id)
    assert got._stored_emotions == {"joy": 5.0}
    peak = store._conn.execute("SELECT peak_emotion_intensity FROM memories WHERE id = ?", (row.id,)).fetchone()[0]
    assert peak == pytest.approx(5.0)


def test_update_with_check_names_false_keeps_the_names_as_given(store: MemoryStore) -> None:
    row = _memory({})
    store.create(row)
    store.update(row.id, emotions={"Joy": 4.0, "never_seen_zz": 9.0}, check_names=False)
    assert _stored(store, row.id)._stored_emotions == {"Joy": 4.0, "never_seen_zz": 9.0}


def test_update_tells_the_engine_the_checked_emotions(store: MemoryStore, emotion_engine_stub) -> None:
    row = _memory({})
    store.create(row, fold=False)
    store.update(row.id, emotions={"Joy": 4.0, "never_seen_zz": 9.0})
    assigned = [c for c in emotion_engine_stub.recorded_calls if c[0] == "on_emotions_assigned"]
    assert len(assigned) == 1 and assigned[0][1]._stored_emotions == {"joy": 4.0}


# --- a merge keeps the names the rows hold ---------------------------------------------------------------------------------


def test_a_merge_with_no_bridge_keeps_a_name_her_file_lacks(store: MemoryStore) -> None:
    target = _memory({"never_seen_zz": 4.0, "joy": 2.0})
    candidate = _memory({"never_seen_zz": 8.0})
    store.create(target, fold=False, check_names=False)
    store.merge_into(target, candidate, "merged text", engine_live=True)
    assert _stored(store, target.id)._stored_emotions == {"never_seen_zz": 6.0, "joy": 2.0}


# --- the queue -----------------------------------------------------------------------------------------------------------


def test_a_queued_memory_is_checked_when_it_is_queued(tmp_path: Path, store: MemoryStore) -> None:
    memory = _memory({"Joy": 6.0, "never_seen_zz": 9.0}, memory_type="dream")
    assert route_write(store, memory, source="test") == memory.id
    queued = PendingQueue(tmp_path).read_recent("dream", limit=1)[0]
    assert queued._stored_emotions == {"joy": 6.0}
    assert "never_seen_zz" not in (tmp_path / "pending_candidates.jsonl").read_text(encoding="utf-8")


# --- the grief touch -----------------------------------------------------------------------------------------------------


def test_a_grief_touch_is_checked_only_when_it_will_be_written(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    def touch() -> Memory:
        return Memory.create_new(
            content="a touch", memory_type="grief_event", domain="us", emotions={"Joy": 3.0, "never_seen_zz": 9.0}, metadata={"grief_referent_id": "ref-1"}
        )

    first = touch()
    assert store.create_grief_touch_unless_recent(first, hours=24.0) == first.id
    assert _stored(store, first.id)._stored_emotions == {"joy": 3.0}
    calls: list[object] = []
    real = MemoryStore.checked
    monkeypatch.setattr(MemoryStore, "checked", lambda self, m: (calls.append(m), real(self, m))[1])
    assert store.create_grief_touch_unless_recent(touch(), hours=24.0) is None  # debounced: nothing is asked
    assert calls == []
