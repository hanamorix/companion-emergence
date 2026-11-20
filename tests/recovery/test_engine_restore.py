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
    s._conn.execute(
        "UPDATE memories SET state='fading', content_snapshot='stale snapshot' WHERE id='faded'"
    )
    s._conn.commit()
    s.close()


def _seed_source(src: Path):
    src.mkdir(parents=True, exist_ok=True)
    s = MemoryStore(src / "memories.db")
    s.create(Memory(id="keep", content="kept", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 2, tzinfo=UTC)))
    s.create(Memory(id="faded", content="ORIGINAL long text", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 2, tzinfo=UTC)))
    # Older than every row the persona holds, so a restore that told the engine (a fold) would move its persona start and
    # change the state file.
    s.create(Memory(id="lost", content="forgotten original", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 3, 20, tzinfo=UTC)))
    s.close()


def test_source_mode_restores_missing_and_unfades(tmp_path):
    persona = tmp_path / "Canary"
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
    assert refaded.content_snapshot is None
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


def _told_to_the_engine(persona: Path):
    """What the engine has been told so far: the bytes of its queue file and of its state file (None when a file does not exist).

    A commit is applied to the state (Engine 4) and a chat change or showing is queued; a restore must change neither.
    """
    from brain.emotion import engine_files

    def read(path: Path):
        return path.read_bytes() if path.exists() else None

    return read(engine_files.queue_path(persona)), read(engine_files.state_path(persona))


def test_a_restore_is_not_new_experience_and_tells_nothing_to_the_emotion_engine(tmp_path, monkeypatch):
    persona = tmp_path / "Canary"
    _seed_current(persona)
    src = tmp_path / "src"
    _seed_source(src)
    told_by_seeding = _told_to_the_engine(persona)  # the seeding created rows with the default fold
    plan = _build_restore_plan(persona, source_dir=src)

    seen = []
    real_create = MemoryStore.create

    def recording_create(self, memory, **kwargs):
        seen.append(kwargs)
        return real_create(self, memory, **kwargs)

    monkeypatch.setattr(MemoryStore, "create", recording_create)
    _apply_memory_restores(persona, plan)

    assert seen and all(kwargs.get("fold") is False for kwargs in seen)
    assert _told_to_the_engine(persona) == told_by_seeding


def test_a_graveyard_restore_is_not_new_experience_either(tmp_path, monkeypatch):
    import json

    persona = tmp_path / "Nova"
    persona.mkdir(parents=True)
    s = MemoryStore(persona / "memories.db")
    s.create(Memory(id="alive", content="alive", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 2, tzinfo=UTC)))
    s.close()
    told_by_seeding = _told_to_the_engine(persona)
    grave = {
        "memory_id": "gone", "summary": "compressed summary", "domain": "us",
        "memory_type": "conversation",
        "created_at_iso": datetime(2026, 4, 1, tzinfo=UTC).isoformat(),
        "emotion_at_ingest": {}, "hebbian_neighbors": [["alive", 0.5]],
    }
    (persona / "forgotten_memories.jsonl").write_text(json.dumps(grave) + "\n")
    plan = _build_restore_plan(persona, source_dir=None)
    assert set(plan.missing_summaries) == {"gone"}

    seen = []
    real_create = MemoryStore.create

    def recording_create(self, memory, **kwargs):
        seen.append(kwargs)
        return real_create(self, memory, **kwargs)

    monkeypatch.setattr(MemoryStore, "create", recording_create)
    _apply_memory_restores(persona, plan)

    assert len(seen) == 1 and seen[0].get("fold") is False
    assert _told_to_the_engine(persona) == told_by_seeding
