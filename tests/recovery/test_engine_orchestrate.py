import hashlib
from datetime import UTC, datetime

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.recovery.engine import preflight_recovery, run_recovery


def _persona_with_dangling(tmp_path):
    p = tmp_path / "Phoebe"
    p.mkdir()
    s = MemoryStore(p / "memories.db")
    s.create(Memory(id="S", content="survivor", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 1, tzinfo=UTC), importance=10.0))
    s.close()
    h = HebbianMatrix(p / "hebbian.db")
    h.strengthen("S", "V", 0.6)  # dangling
    h.close()
    (p / "source-manifest.json").write_text(
        '{"migrated_at_utc":"2026-05-01T00:00:00Z","lived_age_hours_at_migration":5.0}')
    return p


def _source_with_v(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    s = MemoryStore(src / "memories.db")
    s.create(Memory(id="S", content="survivor", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 1, tzinfo=UTC)))
    s.create(Memory(id="V", content="the linked memory", memory_type="conversation",
                    domain="us", created_at=datetime(2026, 4, 1, tzinfo=UTC)))
    s.close()
    h = HebbianMatrix(src / "hebbian.db")
    h.strengthen("S", "V", 0.6)
    h.close()
    return src


def test_source_equals_target_refused(tmp_path):
    p = _persona_with_dangling(tmp_path)
    import pytest
    with pytest.raises(ValueError):
        run_recovery(p, source_dir=p, dry_run=False)


def test_real_run_restores_links_and_preserves_source(tmp_path):
    p = _persona_with_dangling(tmp_path)
    src = _source_with_v(tmp_path)
    src_before = hashlib.sha256((src / "memories.db").read_bytes()).hexdigest()

    report = run_recovery(p, source_dir=src, dry_run=False)

    s = MemoryStore(p / "memories.db")
    v = s.get("V")
    assert v is not None and v.content == "the linked memory"
    s.close()
    h = HebbianMatrix(p / "hebbian.db")
    assert "V" in {nid for nid, _ in h.neighbors("S")}      # link followable again
    h.close()
    assert report.backup_path is not None
    assert report.memories_restored_full == 1
    assert hashlib.sha256((src / "memories.db").read_bytes()).hexdigest() == src_before


def test_dry_run_changes_nothing(tmp_path):
    p = _persona_with_dangling(tmp_path)
    src = _source_with_v(tmp_path)
    before = hashlib.sha256((p / "memories.db").read_bytes()).hexdigest()
    report = run_recovery(p, source_dir=src, dry_run=True)
    after = hashlib.sha256((p / "memories.db").read_bytes()).hexdigest()
    assert before == after
    assert report.memories_restored_full == 1
    assert report.dry_run is True


def test_preflight_is_pure_read(tmp_path):
    p = _persona_with_dangling(tmp_path)
    src = _source_with_v(tmp_path)
    before = hashlib.sha256((p / "memories.db").read_bytes()).hexdigest()
    info = preflight_recovery(p, source_dir=src)
    after = hashlib.sha256((p / "memories.db").read_bytes()).hexdigest()
    assert before == after
    assert info["missing"] == 1
