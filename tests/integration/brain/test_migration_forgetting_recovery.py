import json
from datetime import UTC, datetime, timedelta

from brain.forgetting import FORGETTING_STATE_FILENAME, run_pass
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.migrator.companion_emergence import (
    CompanionEmergenceMigrateArgs,
    migrate_companion_emergence,
)
from brain.paths import get_persona_dir
from brain.recovery.engine import run_recovery


class _Bus:
    def publish(self, e):
        pass


def _old(days):
    return datetime.now(UTC) - timedelta(days=days)


def test_phoebe_loop_heals(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))

    # pristine v0.0.11-ish source: importance-10 anchor + 3 low-salience neighbours
    src = tmp_path / "Downloads" / "Phoebe"
    src.mkdir(parents=True)
    s = MemoryStore(src / "memories.db")
    s.create(Memory(id="anchor", content="the important one", memory_type="conversation",
                    domain="us", created_at=_old(400), importance=10.0,
                    emotions={"love": 9.0}, score=9.0))
    for n in ("n1", "n2", "n3"):
        s.create(Memory(id=n, content=f"linked {n}", memory_type="conversation",
                        domain="us", created_at=_old(400), importance=0.0))
    s.close()
    h = HebbianMatrix(src / "hebbian.db")
    for n in ("n1", "n2", "n3"):
        h.strengthen("anchor", n, 0.6)
    h.close()
    (src / "persona_config.json").write_text(json.dumps({"user_name": "zero"}))

    # migrate (CE→CE copytree)
    migrate_companion_emergence(CompanionEmergenceMigrateArgs(
        input_dir=src, install_as="Phoebe", force=False))
    p = get_persona_dir("Phoebe")

    # Force the neighbours to LOSE on the next pass:
    #  - disable import grace (empty manifest)
    #  - large lived-age so cold-start doesn't exempt everything
    #  - mark neighbours 'fading' and pre-seed the consecutive-low counter
    (p / "source-manifest.json").write_text("{}")
    (p / "felt_time_state.json").write_text(json.dumps({"lived_age_hours": 9999.0}))
    store = MemoryStore(p / "memories.db")
    for n in ("n1", "n2", "n3"):
        store._conn.execute("UPDATE memories SET state='fading' WHERE id=?", (n,))
    store._conn.commit()
    store.close()
    (p / FORGETTING_STATE_FILENAME).write_text(json.dumps({"n1": 2, "n2": 2, "n3": 2}))

    run_pass(p, event_bus=_Bus())

    # BROKEN STATE: neighbours forgotten, anchor survives, NO dangling edges
    store = MemoryStore(p / "memories.db")
    assert store.get("anchor") is not None
    assert store.get("n1") is None
    store.close()
    h = HebbianMatrix(p / "hebbian.db")
    assert h.neighbors("anchor") == []   # edges removed with the lost memories
    h.close()

    # RECOVER from the pristine Downloads source
    report = run_recovery(p, source_dir=src, dry_run=False)
    assert report.memories_restored_full == 3

    store = MemoryStore(p / "memories.db")
    assert store.get("n1").content == "linked n1"
    store.close()
    h = HebbianMatrix(p / "hebbian.db")
    names = {nid for nid, _ in h.neighbors("anchor")}
    assert {"n1", "n2", "n3"} <= names   # links followable again
    h.close()
