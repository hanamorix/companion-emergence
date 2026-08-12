# tests/unit/brain/maker/test_wiring_memory.py
from brain.maker.maker import Making
from brain.maker.wiring import write_making_memory
from brain.memory.store import MemoryStore


def _promote_pending(persona_dir):
    """Drain the consolidation gate with a promote-all classifier.

    `making` is a GATED type (Root-2 stopgap): write_making_memory enqueues the act-memory;
    it reaches memories.db only after a consolidation drain."""
    from brain.engines.consolidation import Decision, run_consolidation

    s = MemoryStore(persona_dir / "memories.db")
    try:
        run_consolidation(
            s, persona_dir=persona_dir, classifier=lambda _c, _ctx: Decision("new")
        )
    finally:
        s.close()


def test_act_memory_written_with_making_type_and_emotions(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    write_making_memory(store, Making("elegy", "For the dog", "Soft paws.", "private", "raw"),
                        emotions={"tenderness": 0.1})
    # making is GATED: enqueues, not written straight to db.
    assert not store.list_by_type("making", active_only=True, limit=5)
    store.close()
    _promote_pending(tmp_path)
    store2 = MemoryStore(tmp_path / "memories.db")
    mems = store2.list_by_type("making", active_only=True, limit=5)
    assert len(mems) == 1
    assert "For the dog" in mems[0].content
    assert mems[0].emotions.get("tenderness") == 0.1
    store2.close()
