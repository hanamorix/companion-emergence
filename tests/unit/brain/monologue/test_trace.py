import json

from brain.chat.monologue_capture import capture_monologue
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore
from brain.monologue.trace import MONOLOGUE_TRACE_TYPE, write_trace_memory

# memory-consolidation migration: monologue_trace is a GATED type. write_trace_memory
# now enqueues a pending candidate instead of writing to memories.db. These tests
# verify the produced trace's fields against the queue entry (read_recent yields
# Memory objects with the same content/type/domain/emotions/state). The trace's
# DB-fade lifecycle is a separate concern — see test_monologue_trace_lifecycle.


def _trace_candidate(store):
    """The single most-recent monologue_trace candidate in the pending queue."""
    return PendingQueue(store.persona_dir).read_recent(MONOLOGUE_TRACE_TYPE, limit=1)[0]


def test_write_trace_memory_persists_verbatim_as_active(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    mem_id = write_trace_memory(store, "i kept thinking about the lighthouse")
    # Enqueued as a candidate, not a memories.db row.
    assert store.get(mem_id) is None
    mem = _trace_candidate(store)
    assert mem.id == mem_id
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
    write_trace_memory(store, "drifting again")
    mem = _trace_candidate(store)
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
    # Tier 2: a trace candidate exists in the pending queue (gated type).
    traces = PendingQueue(store.persona_dir).read_recent(MONOLOGUE_TRACE_TYPE, limit=10)
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
