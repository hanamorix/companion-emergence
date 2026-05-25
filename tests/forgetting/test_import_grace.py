import json
from datetime import UTC, datetime

from brain.forgetting import run_pass
from brain.memory.store import Memory, MemoryStore


class _Bus:
    def publish(self, e):
        pass


def test_migrated_memory_within_grace_is_exempt(tmp_path):
    # install has accrued 10 lived-h; migrated at 5 lived-h → 5 elapsed < 168.
    (tmp_path / "felt_time_state.json").write_text(json.dumps({"lived_age_hours": 10.0}))
    (tmp_path / "source-manifest.json").write_text(json.dumps({
        "migrated_at_utc": datetime(2026, 5, 1, tzinfo=UTC).isoformat().replace("+00:00", "Z"),
        "lived_age_hours_at_migration": 5.0,
    }))
    store = MemoryStore(tmp_path / "memories.db")
    # ancient, zero-salience, created BEFORE migration → would normally fade
    store.create(Memory(id="V", content="victim", memory_type="conversation",
                        domain="us", created_at=datetime(2026, 4, 1, tzinfo=UTC),
                        importance=0.0))
    store.close()

    summary = run_pass(tmp_path, event_bus=_Bus())

    store = MemoryStore(tmp_path / "memories.db")
    mem = store.get("V")
    store.close()
    assert mem is not None
    assert mem.state == "active"        # exempt — not faded
    assert summary["exempt"] >= 1


def test_post_migration_memory_not_grace_exempt_and_fades(tmp_path):
    # Same setup, but memory created AFTER migration → grace does not apply.
    (tmp_path / "felt_time_state.json").write_text(json.dumps({"lived_age_hours": 10.0}))
    (tmp_path / "source-manifest.json").write_text(json.dumps({
        "migrated_at_utc": datetime(2026, 5, 1, tzinfo=UTC).isoformat().replace("+00:00", "Z"),
        "lived_age_hours_at_migration": 5.0,
    }))
    store = MemoryStore(tmp_path / "memories.db")
    store.create(Memory(id="N", content="native", memory_type="conversation",
                        domain="us", created_at=datetime(2026, 5, 15, tzinfo=UTC),
                        importance=0.0))
    store.close()

    run_pass(tmp_path, event_bus=_Bus())

    store = MemoryStore(tmp_path / "memories.db")
    mem = store.get("N")
    store.close()
    # native zero-salience memory is NOT import-exempt → fades to 'fading'
    assert mem.state == "fading"
