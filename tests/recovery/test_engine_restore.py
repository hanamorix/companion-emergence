from datetime import UTC, datetime
from pathlib import Path

from brain.memory.store import Memory, MemoryStore
from brain.recovery.engine import _apply_memory_restores, _build_restore_plan


def _seed_current(persona: Path):
    persona.mkdir(parents=True, exist_ok=True)
    s = MemoryStore(persona / "memories.db")
    s.create(Memory(id="keep", content="kept", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 2, tzinfo=UTC)))
    s.create(Memory(id="faded", content="short summary", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 2, tzinfo=UTC)))
    # create() ignores `state`; force 'faded' into fading state via SQL.
    s._conn.execute("UPDATE memories SET state='fading' WHERE id='faded'")
    s._conn.commit()
    s.close()


def _seed_source(src: Path):
    src.mkdir(parents=True, exist_ok=True)
    s = MemoryStore(src / "memories.db")
    s.create(Memory(id="keep", content="kept", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 2, tzinfo=UTC)))
    s.create(Memory(id="faded", content="ORIGINAL long text", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 2, tzinfo=UTC)))
    s.create(Memory(id="lost", content="forgotten original", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 2, tzinfo=UTC)))
    s.close()


def test_source_mode_restores_missing_and_unfades(tmp_path):
    persona = tmp_path / "Phoebe"
    _seed_current(persona)
    src = tmp_path / "src"
    _seed_source(src)

    plan = _build_restore_plan(persona, source_dir=src)
    assert set(plan.missing) == {"lost"}
    assert set(plan.unfade) == {"faded"}

    counts = _apply_memory_restores(persona, plan)
    assert counts["restored_full"] == 1
    assert counts["unfaded"] == 1

    s = MemoryStore(persona / "memories.db")
    assert s.get("lost").content == "forgotten original"
    refaded = s.get("faded")
    assert refaded.state == "active"
    assert refaded.content == "ORIGINAL long text"
    assert s.get("keep").content == "kept"
    s.close()


def test_graveyard_mode_restores_from_summary(tmp_path):
    import json
    persona = tmp_path / "Nova"
    persona.mkdir(parents=True)
    s = MemoryStore(persona / "memories.db")
    s.create(Memory(id="alive", content="alive", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 2, tzinfo=UTC)))
    s.close()
    # graveyard has one forgotten memory (summary only) + tombstoned neighbour
    grave = {
        "memory_id": "gone", "summary": "compressed summary", "domain": "us",
        "memory_type": "conversation",
        "created_at_iso": datetime(2026, 4, 1, tzinfo=UTC).isoformat(),
        "emotion_at_ingest": {}, "hebbian_neighbors": [["alive", 0.5]],
    }
    (persona / "forgotten_memories.jsonl").write_text(json.dumps(grave) + "\n")

    plan = _build_restore_plan(persona, source_dir=None)
    assert plan.mode == "graveyard"
    assert set(plan.missing_summaries) == {"gone"}
    assert plan.graveyard_neighbors["gone"] == [("alive", 0.5)]

    counts = _apply_memory_restores(persona, plan)
    assert counts["restored_summary"] == 1
    s = MemoryStore(persona / "memories.db")
    assert s.get("gone").content == "compressed summary"
    s.close()
