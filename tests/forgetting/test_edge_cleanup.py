import json
from datetime import UTC, datetime, timedelta

from brain.forgetting import FORGETTING_STATE_FILENAME, run_pass
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


class _Bus:
    def __init__(self):
        self.events = []

    def publish(self, e):
        self.events.append(e)


def _old(days):
    return datetime.now(UTC) - timedelta(days=days)


def test_lose_removes_edges_and_records_neighbors(tmp_path):
    # Survivor S (high emotion → salience > FADE), victim V (will be LOST).
    store = MemoryStore(tmp_path / "memories.db")
    store.create(Memory(id="S", content="survivor", memory_type="conversation",
                        domain="us", created_at=_old(400), importance=10.0,
                        emotions={"love": 9.0}, score=9.0))
    store.create(Memory(id="V", content="victim", memory_type="conversation",
                        domain="us", created_at=_old(400), importance=0.0))
    # create() does NOT persist `state`; force V into 'fading' so one pass can LOSE it.
    store._conn.execute("UPDATE memories SET state='fading' WHERE id='V'")
    store._conn.commit()
    store.close()

    h = HebbianMatrix(tmp_path / "hebbian.db")
    h.strengthen("S", "V", delta=0.6)
    h.close()

    # lived_age > 0 so is_exempt's cold-start branch doesn't exempt everything.
    (tmp_path / "felt_time_state.json").write_text(json.dumps({"lived_age_hours": 9999.0}))
    # Pre-seed the consecutive-low counter so a single pass crosses LOST_PASS_COUNT.
    (tmp_path / FORGETTING_STATE_FILENAME).write_text(json.dumps({"V": 2}))

    run_pass(tmp_path, event_bus=_Bus())

    store = MemoryStore(tmp_path / "memories.db")
    assert store.get("V") is None        # victim lost
    assert store.get("S") is not None    # survivor kept
    store.close()

    h = HebbianMatrix(tmp_path / "hebbian.db")
    assert h.neighbors("S") == []        # NO dangling edge to V (fails before the fix)
    h.close()

    grave = [json.loads(line) for line in
             (tmp_path / "forgotten_memories.jsonl").read_text().splitlines()]
    v_entry = next(e for e in grave if e["memory_id"] == "V")
    assert ["S", 0.6] in v_entry["hebbian_neighbors"]
