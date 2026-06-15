# tests/unit/brain/maker/test_wiring_memory.py
from brain.maker.maker import Making
from brain.maker.wiring import write_making_memory
from brain.memory.store import MemoryStore


def test_act_memory_written_with_making_type_and_emotions(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    write_making_memory(store, Making("elegy", "For the dog", "Soft paws.", "private", "raw"),
                        emotions={"tenderness": 0.1})
    mems = store.list_by_type("making", active_only=True, limit=5)
    assert len(mems) == 1
    assert "For the dog" in mems[0].content
    assert mems[0].emotions.get("tenderness") == 0.1
    store.close()
