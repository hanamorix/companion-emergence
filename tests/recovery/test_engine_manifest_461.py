"""#461: `run_recovery` must not rewrite source-manifest.json as empty after a read it could not do.

The manifest was read with the fall-back-to-empty policy and then written back with the recovery keys, so a transient I/O fault or a
damaged file replaced the manifest's real contents with `{}`. The read now heals (a damaged file is kept aside as
`source-manifest.json.corrupt-<ts>` and a valid backup is restored) and raises on an I/O fault, and it happens BEFORE the backup and
the memory restores, so an unreadable manifest aborts recovery before the persona is changed.

Synthetic names only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from brain.health.json_io import JsonFileError
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.recovery.engine import run_recovery

_RECOVERY_KEYS = {"migrated_at_utc", "lived_age_hours_at_migration", "recovered_at"}


def _persona(tmp_path):
    p = tmp_path / "Canary"
    p.mkdir()
    s = MemoryStore(p / "memories.db")
    s.create(Memory(id="S", content="survivor", memory_type="conversation", domain="us",
                    created_at=datetime(2026, 4, 1, tzinfo=UTC), importance=10.0))
    s.close()
    return p


def _source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    s = MemoryStore(src / "memories.db")
    for mid, content in (("S", "survivor"), ("V", "the linked memory")):
        s.create(Memory(id=mid, content=content, memory_type="conversation", domain="us",
                        created_at=datetime(2026, 4, 1, tzinfo=UTC)))
    s.close()
    h = HebbianMatrix(src / "hebbian.db")
    h.strengthen("S", "V", 0.6)
    h.close()
    return src


def _count(p) -> int:
    s = MemoryStore(p / "memories.db")
    try:
        return s.count(active_only=False)
    finally:
        s.close()


def _corrupt_copies(p):
    return sorted(p.glob("source-manifest.json.corrupt-*"))


def test_an_unreadable_manifest_aborts_recovery_before_anything_changes(tmp_path):
    # C461-a: an I/O fault (a directory where the manifest should be).
    p, src = _persona(tmp_path), _source(tmp_path)
    (p / "source-manifest.json").mkdir()

    with pytest.raises(JsonFileError) as excinfo:
        run_recovery(p, source_dir=src, dry_run=False)

    assert excinfo.value.kind == "io"
    assert not list(p.glob("recover-backup-*"))
    assert _count(p) == 1  # V was not restored
    assert (p / "source-manifest.json").is_dir()


def test_a_malformed_manifest_is_kept_aside_and_recovery_completes(tmp_path):
    # C461-b
    p, src = _persona(tmp_path), _source(tmp_path)
    (p / "source-manifest.json").write_bytes(b'{"a": ')

    report = run_recovery(p, source_dir=src, dry_run=False)

    assert report.memories_restored_full == 1
    copies = _corrupt_copies(p)
    assert len(copies) == 1 and copies[0].read_bytes() == b'{"a": '
    live = json.loads((p / "source-manifest.json").read_text())
    assert _RECOVERY_KEYS <= set(live)


def test_a_manifest_of_the_wrong_shape_is_kept_aside_too(tmp_path):
    # C461-c
    p, src = _persona(tmp_path), _source(tmp_path)
    (p / "source-manifest.json").write_text("[1, 2]")

    run_recovery(p, source_dir=src, dry_run=False)

    copies = _corrupt_copies(p)
    assert len(copies) == 1 and copies[0].read_text() == "[1, 2]"
    assert _RECOVERY_KEYS <= set(json.loads((p / "source-manifest.json").read_text()))


def test_a_malformed_manifest_with_a_valid_backup_is_restored_from_it(tmp_path):
    # C461-d
    p, src = _persona(tmp_path), _source(tmp_path)
    (p / "source-manifest.json").write_bytes(b'{"a": ')
    (p / "source-manifest.json.bak1").write_text(json.dumps({"keep": "me", "migrated_at_utc": "2026-05-01T00:00:00Z"}))

    run_recovery(p, source_dir=src, dry_run=False)

    live = json.loads((p / "source-manifest.json").read_text())
    assert live["keep"] == "me" and _RECOVERY_KEYS <= set(live)
    assert len(_corrupt_copies(p)) == 1


def test_a_healthy_manifest_keeps_its_other_keys(tmp_path):
    # C461-e PIN
    p, src = _persona(tmp_path), _source(tmp_path)
    (p / "source-manifest.json").write_text(json.dumps({"keep": "me"}))

    run_recovery(p, source_dir=src, dry_run=False)

    live = json.loads((p / "source-manifest.json").read_text())
    assert live["keep"] == "me" and _RECOVERY_KEYS <= set(live) and not _corrupt_copies(p)


def test_an_absent_manifest_is_created(tmp_path):
    # C461-e PIN
    p, src = _persona(tmp_path), _source(tmp_path)

    run_recovery(p, source_dir=src, dry_run=False)

    assert _RECOVERY_KEYS <= set(json.loads((p / "source-manifest.json").read_text()))


def test_a_dry_run_does_not_touch_a_malformed_manifest(tmp_path):
    # C461-e PIN: the read happens after the dry-run early return.
    p, src = _persona(tmp_path), _source(tmp_path)
    (p / "source-manifest.json").write_bytes(b'{"a": ')
    before = sorted(x.name for x in p.iterdir())

    run_recovery(p, source_dir=src, dry_run=True)

    assert sorted(x.name for x in p.iterdir()) == before
    assert (p / "source-manifest.json").read_bytes() == b'{"a": '
