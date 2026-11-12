"""The dream sites read a memory's feelings through the one read (Engine 7 ONE5 item 4).

A `DreamEngine` with a persona directory asks the persona's engine (faded values once switched over); one without asks the pure
read. `dream_seeds` stays pure: it takes the live engine (or None for the pure read) as `eng`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from brain.bridge.provider import FakeProvider
from brain.emotion.state import EmotionalState
from brain.engines import dream_seeds
from brain.engines.dream import DreamEngine
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _mem(mid: str, *, emotions: dict[str, float], mtype: str = "conversation") -> Memory:
    return Memory.from_stored_row(
        id=mid, content=f"content-{mid}", memory_type=mtype, domain="us", created_at=NOW, emotions=emotions, importance=1.0
    )


def _engine(tmp_path: Path, *, persona_dir: Path | None) -> DreamEngine:
    return DreamEngine(
        store=MemoryStore(tmp_path / "memories.db"),
        hebbian=HebbianMatrix(db_path=":memory:"),
        provider=FakeProvider(),
        persona_dir=persona_dir,
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
    )


class _ReadsThrough:
    """Stands in for a live engine: its one per-memory read answers what `feelings_of` says."""

    def __init__(self, feelings_of):
        self._feelings_of = feelings_of

    def memory_feelings(self, memory):
        return SimpleNamespace(feelings=self._feelings_of(memory))


def test_dream_seeds_read_through_the_engine_they_are_given():
    mood = EmotionalState(emotions={"joy": 2.0})
    mem = _mem("a", emotions={"joy": 9.0})
    faded = _ReadsThrough(lambda m: {"joy": 1.0, "memory_grief": 5.0})
    assert dream_seeds.emotional_congruence(mem, mood, eng=None) == 18.0  # no engine: the pure read
    assert dream_seeds.emotional_congruence(mem, mood, eng=faded) == 2.0
    grief = _mem("g", emotions={"memory_grief": 9.0}, mtype="grief_event")
    assert dream_seeds.grief_pull(grief, eng=None) == 0.9
    assert dream_seeds.grief_pull(grief, eng=faded) == 0.5


def test_select_seed_ranks_by_the_engine_read_it_is_given():
    cands = [_mem("g1", emotions={"memory_grief": 9.0}, mtype="grief_event"), _mem("g2", emotions={"memory_grief": 1.0}, mtype="grief_event")]
    mood = EmotionalState()
    assert dream_seeds.select_seed(cands, mood, [], recent_seed_ids=[], eng=None).id == "g1"
    inverted = _ReadsThrough(lambda m: {"memory_grief": 9.0 if m.id == "g2" else 1.0})
    assert dream_seeds.select_seed(cands, mood, [], recent_seed_ids=[], eng=inverted).id == "g2"


def test_a_dream_engine_with_a_persona_dir_reads_through_the_persona_engine(tmp_path: Path, emotion_engine_stub):
    emotion_engine_stub.faded = {("a", "joy"): 3.0, ("b", "joy"): 4.0}
    engine = _engine(tmp_path, persona_dir=tmp_path)
    try:
        seed, neighbour = _mem("a", emotions={"joy": 9.0}), _mem("b", emotions={"joy": 8.0})
        assert engine._feelings_of(seed) == {"joy": 3.0}
        dream = engine._write_dream_memory(seed, [(neighbour, 1.0)], "DREAM: x", eng=None)
        assert dream._stored_emotions == {"joy": 7.0}  # the off-path sum of the engine's read values, not 9 + 8
    finally:
        engine.store.close()


def test_the_initiate_snapshot_vector_is_the_engine_read_of_the_dream_row(tmp_path: Path, emotion_engine_stub, monkeypatch):
    from brain.initiate import emit

    emotion_engine_stub.faded = {("d", "joy"): 2.5}
    seen = []
    monkeypatch.setattr(emit, "emit_initiate_candidate", lambda *a, **k: seen.append(k["emotional_snapshot"]))
    engine = _engine(tmp_path, persona_dir=tmp_path)
    try:
        engine._emit_initiate_candidate(_mem("s", emotions={}), [], _mem("d", emotions={"joy": 9.0}))
        assert [snap.vector for snap in seen] == [{"joy": 2.5}]
    finally:
        engine.store.close()


def test_a_dream_engine_without_a_persona_dir_reads_the_pure_read(tmp_path: Path, emotion_engine_stub):
    emotion_engine_stub.faded = {("a", "joy"): 3.0}  # would show if the engine were asked
    engine = _engine(tmp_path, persona_dir=None)
    try:
        assert engine._feelings_of(_mem("a", emotions={"joy": 9.0, "grief": 12.0})) == {"joy": 9.0, "grief": 10.0}
        assert emotion_engine_stub.recorded_calls == []
    finally:
        engine.store.close()
