import json

from brain.chat.monologue_capture import capture_monologue
from brain.memory.store import Memory, MemoryStore
from brain.monologue.trace import MONOLOGUE_TRACE_TYPE, write_trace_memory


def test_write_trace_memory_persists_verbatim_as_active(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    mem_id = write_trace_memory(store, "i kept thinking about the lighthouse")
    mem = store.get(mem_id)
    assert mem is not None
    assert mem.content == "i kept thinking about the lighthouse"
    assert mem.memory_type == MONOLOGUE_TRACE_TYPE
    assert mem.domain == "monologue"
    assert mem.state == "active"


def test_write_trace_memory_seeds_current_emotion_aggregate(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    # Seed an emotionally-charged active memory so the aggregate is non-empty.
    # Use a baseline (registered) emotion — aggregate_state skips unregistered names.
    store.create(
        Memory.create_new(
            content="a charged moment",
            memory_type="episodic",
            domain="chat",
            emotions={"love": 8.0},
        )
    )
    mem_id = write_trace_memory(store, "drifting again")
    mem = store.get(mem_id)
    assert mem.emotions, "trace should inherit the current emotional aggregate"
    assert "love" in mem.emotions


def test_capture_writes_trace_memory_and_surfaced_digest(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    out = capture_monologue(
        persona_dir=tmp_path,
        store=store,
        monologue="the raw drift",
        feed_digest="she drifted",
        surface=True,
    )
    assert out == "the raw drift"
    # Tier 2: a trace memory exists.
    traces = store.list_by_type(MONOLOGUE_TRACE_TYPE)
    assert len(traces) == 1
    assert traces[0].content == "the raw drift"
    # Tier 3: the digest line carries surfaced.
    line = (tmp_path / "monologue_digest.jsonl").read_text().splitlines()[0]
    obj = json.loads(line)
    assert obj["digest"] == "she drifted"
    assert obj["surfaced"] is True


def test_capture_trace_write_failure_does_not_raise(tmp_path):
    # A closed store raises on create; capture must swallow it and still return.
    store = MemoryStore(tmp_path / "memories.db")
    store.close()
    out = capture_monologue(
        persona_dir=tmp_path,
        store=store,
        monologue="drift",
        feed_digest="gist",
        surface=False,
    )
    assert out == "drift"
    line = json.loads((tmp_path / "monologue_digest.jsonl").read_text().splitlines()[0])
    assert line["surfaced"] is False
