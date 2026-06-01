"""monologue_trace is picked up by the forgetting engine like any memory:
FADE blurs verbatim→summary, LOSE forgets it (graveyard + grief)."""
import json
from datetime import UTC, datetime, timedelta

from brain.forgetting import run_pass
from brain.memory.store import MemoryStore
from brain.monologue.trace import write_trace_memory


class _Bus:
    def __init__(self):
        self.events = []

    def publish(self, e):
        self.events.append(e)


def _age_memory(store, mem_id, days):
    """Backdate created_at + last_accessed_at so the recent-buffer exemption lifts."""
    old = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    store._conn.execute(
        "UPDATE memories SET created_at = ?, last_accessed_at = ? WHERE id = ?",
        (old, old, mem_id),
    )
    store._conn.commit()


def test_unrevisited_trace_fades_then_is_lost(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    mem_id = write_trace_memory(store, "an idle unremarkable drift")
    store.update(mem_id, emotions={})  # strip emotion seed → low salience
    _age_memory(store, mem_id, days=40)
    store.close()

    # Non-cold lived-age, else the recent-buffer exemption protects everything.
    (tmp_path / "felt_time_state.json").write_text(json.dumps({"lived_age_hours": 9999.0}))

    bus = _Bus()
    # Pass 1: active → fading (FADE). Pass 2: fading → lost (consecutive low >= 2).
    for _ in range(3):
        run_pass(tmp_path, event_bus=bus)

    store2 = MemoryStore(tmp_path / "memories.db")
    assert store2.get(mem_id) is None  # lost
    graveyard = tmp_path / "forgotten_memories.jsonl"
    assert graveyard.exists()
    assert "an idle unremarkable drift" in graveyard.read_text()  # stored under "summary"


def test_recalled_trace_stays(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    mem_id = write_trace_memory(store, "a thought she keeps returning to")
    _age_memory(store, mem_id, days=40)
    # Simulate repeated recall — bumps recall_count well above the salience floor.
    for _ in range(12):
        store.get(mem_id)
    store.close()

    (tmp_path / "felt_time_state.json").write_text(json.dumps({"lived_age_hours": 9999.0}))

    bus = _Bus()
    for _ in range(3):
        run_pass(tmp_path, event_bus=bus)

    store2 = MemoryStore(tmp_path / "memories.db")
    assert store2.get(mem_id) is not None  # survived
