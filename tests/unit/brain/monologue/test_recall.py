from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.monologue.recall import recall_monologue
from brain.monologue.trace import write_trace_memory


def _ctx(tmp_path):
    return MemoryStore(tmp_path / "memories.db"), HebbianMatrix(":memory:")


def test_recall_finds_matching_trace_only(tmp_path):
    store, hebbian = _ctx(tmp_path)
    write_trace_memory(store, "the lighthouse kept pulling at me")
    write_trace_memory(store, "something about the train timetable")
    # A non-trace memory containing the keyword must NOT be returned.
    store.create(Memory.create_new(content="lighthouse facts", memory_type="episodic", domain="chat"))

    out = recall_monologue("lighthouse", store=store, hebbian=hebbian, persona_dir=tmp_path)
    bodies = [m["content"] for m in out["monologues"]]
    assert any("lighthouse kept pulling" in b for b in bodies)
    assert "lighthouse facts" not in bodies


def test_recall_bumps_recall_count_keeping_it_sharp(tmp_path):
    store, hebbian = _ctx(tmp_path)
    write_trace_memory(store, "the harbour at dusk")
    before = store.list_by_type("monologue_trace")[0].recall_count
    recall_monologue("harbour", store=store, hebbian=hebbian, persona_dir=tmp_path)
    after = store.list_by_type("monologue_trace")[0].recall_count
    assert after == before + 1


def test_recall_monologue_dispatches(tmp_path):
    from brain.tools.dispatch import dispatch

    store, hebbian = _ctx(tmp_path)
    write_trace_memory(store, "the kettle and the rain")
    out = dispatch(
        "recall_monologue", {"query": "kettle"}, store=store, hebbian=hebbian, persona_dir=tmp_path
    )
    assert out["count"] == 1

