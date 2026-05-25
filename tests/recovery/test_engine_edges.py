from datetime import UTC, datetime

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.recovery.engine import _apply_memory_restores, _build_restore_plan, _repair_edges


def _mem(mid):
    return Memory(id=mid, content=mid, memory_type="conversation", domain="us",
                  created_at=datetime(2026, 4, 1, tzinfo=UTC))


def test_repair_restores_and_sweeps(tmp_path):
    persona = tmp_path / "P"
    persona.mkdir()
    s = MemoryStore(persona / "memories.db")
    s.create(_mem("S"))
    s.close()
    h = HebbianMatrix(persona / "hebbian.db")
    h.strengthen("S", "V", 0.6)        # legacy dangling: V was forgotten
    h.strengthen("S", "ghost", 0.2)    # unrecoverable: ghost not in source
    h.close()

    src = tmp_path / "src"
    src.mkdir()
    ss = MemoryStore(src / "memories.db")
    ss.create(_mem("S"))
    ss.create(_mem("V"))
    ss.close()
    sh = HebbianMatrix(src / "hebbian.db")
    sh.strengthen("S", "V", 0.6)
    sh.close()

    plan = _build_restore_plan(persona, source_dir=src)
    _apply_memory_restores(persona, plan)          # restores V
    repaired, pruned = _repair_edges(persona, plan)

    h = HebbianMatrix(persona / "hebbian.db")
    names = {nid for nid, _ in h.neighbors("S")}
    assert "V" in names                            # edge to V resolves (auto-healed)
    assert "ghost" not in names                    # unrecoverable edge pruned
    h.close()
    assert pruned >= 1


def test_source_mode_restores_edges_with_empty_current_graph(tmp_path):
    # Regression: after forgetting removes a memory's edges, the current
    # hebbian.db is EMPTY (no dangling edge to auto-heal). Recovery must
    # re-create the edge from the source via plan.source_edges.
    persona = tmp_path / "PE"
    persona.mkdir()
    s = MemoryStore(persona / "memories.db")
    s.create(_mem("S"))
    s.close()
    HebbianMatrix(persona / "hebbian.db").close()   # EMPTY — no edges at all

    src = tmp_path / "srcE"
    src.mkdir()
    ss = MemoryStore(src / "memories.db")
    ss.create(_mem("S"))
    ss.create(_mem("V"))
    ss.close()
    sh = HebbianMatrix(src / "hebbian.db")
    sh.strengthen("S", "V", 0.6)
    sh.close()

    plan = _build_restore_plan(persona, source_dir=src)
    assert plan.source_edges, "plan must carry the source's edges"
    _apply_memory_restores(persona, plan)
    repaired, _ = _repair_edges(persona, plan)

    h = HebbianMatrix(persona / "hebbian.db")
    assert "V" in {nid for nid, _ in h.neighbors("S")}   # edge genuinely re-created
    h.close()
    assert repaired >= 1


def test_graveyard_mode_rebuilds_edges_from_neighbors(tmp_path):
    import json
    persona = tmp_path / "G"
    persona.mkdir()
    s = MemoryStore(persona / "memories.db")
    s.create(_mem("alive"))
    s.close()
    HebbianMatrix(persona / "hebbian.db").close()
    grave = {"memory_id": "gone", "summary": "s", "domain": "us",
             "memory_type": "conversation",
             "created_at_iso": datetime(2026, 4, 1, tzinfo=UTC).isoformat(),
             "emotion_at_ingest": {}, "hebbian_neighbors": [["alive", 0.5], ["vanished", 0.9]]}
    (persona / "forgotten_memories.jsonl").write_text(json.dumps(grave) + "\n")

    plan = _build_restore_plan(persona, source_dir=None)
    _apply_memory_restores(persona, plan)          # restores "gone"
    repaired, pruned = _repair_edges(persona, plan)

    h = HebbianMatrix(persona / "hebbian.db")
    names = {nid for nid, _ in h.neighbors("gone")}
    assert "alive" in names          # neighbour exists → edge rebuilt
    assert "vanished" not in names   # neighbour absent → not created
    h.close()
    assert repaired >= 1
