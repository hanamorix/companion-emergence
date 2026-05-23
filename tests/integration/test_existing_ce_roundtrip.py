"""End-to-end: a 50-memory v0.0.12 persona survives migrate_companion_emergence
intact, queried back through the real MemoryStore the bridge would use.

The migrator does a straight shutil.copytree, so the source memories.db is
copied verbatim.  MemoryStore opens any SQLite file whose ``memories`` table
exists and runs ALTER TABLE ADD COLUMN for any new-schema columns — meaning
an old v0.0.12-shaped table is readable without a schema migration step.

This test proves the full roundtrip end-to-end:
  1. Build a 50-memory v0.0.12-shaped persona dir in a temp location.
  2. Call migrate_companion_emergence (the real migrator, no mocks).
  3. Open the migrated memories.db through the real MemoryStore.
  4. Assert count() == 50 and that app_config.json has the right persona.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from brain.memory.store import MemoryStore
from brain.migrator.companion_emergence import (
    CompanionEmergenceMigrateArgs,
    migrate_companion_emergence,
)


def _build_v012_persona(root: Path, name: str = "phoebe", memory_count: int = 50) -> Path:
    """Build a v0.0.12-shaped persona dir with ``memory_count`` memories.

    Uses the minimal column set that v0.0.12 produced — MemoryStore's
    idempotent ALTER TABLE logic migrates the schema on first open.
    """
    persona_dir = root / name
    persona_dir.mkdir(parents=True)

    conn = sqlite3.connect(persona_dir / "memories.db")
    conn.execute(
        """CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT,
            importance INT,
            memory_type TEXT,
            domain TEXT,
            created_at TEXT,
            emotions TEXT,
            tags TEXT,
            active INT
        )"""
    )
    for i in range(memory_count):
        conn.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"mem-{i:05d}",
                f"memory content for index {i}: phoebe recalls the shape of the evening",
                5 + (i % 5),
                "conversation",
                "us",
                f"2026-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00Z",
                "{}",
                "[]",
                1,
            ),
        )
    conn.commit()
    conn.close()

    # Companion-emergence requires these sibling DBs to exist (even empty).
    sqlite3.connect(persona_dir / "hebbian.db").close()
    sqlite3.connect(persona_dir / "crystallizations.db").close()

    (persona_dir / "persona_config.json").write_text(
        json.dumps({
            "persona_name": name,
            "user_name": "zero",
            "voice_template": "nell-example",
            "provider": "claude-cli",
            "model": "sonnet",
        }),
        encoding="utf-8",
    )
    return persona_dir


def test_existing_ce_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """50-memory v0.0.12 persona migrates intact; MemoryStore reads 50 back."""
    src = _build_v012_persona(tmp_path / "old_install", name="phoebe", memory_count=50)

    new_home = tmp_path / "new_kindled_home"
    monkeypatch.setenv("KINDLED_HOME", str(new_home))

    args = CompanionEmergenceMigrateArgs(
        input_dir=src,
        install_as="phoebe",
        force=False,
    )
    report = migrate_companion_emergence(args)

    # --- migrator-level assertions ---
    assert report.memories_migrated == 50
    assert report.source_kind == "companion-emergence"
    assert report.bytes_copied > 0

    # --- persona dir structure ---
    target = new_home / "personas" / "phoebe"
    assert target.is_dir(), f"persona dir not created: {target}"
    assert (target / "memories.db").is_file()
    assert (target / "persona_config.json").is_file()
    assert (target / "source-manifest.json").is_file()

    # --- real MemoryStore round-trip ---
    store = MemoryStore(db_path=target / "memories.db")
    try:
        # count(active_only=True) — all 50 seeds have active=1
        assert store.count(active_only=True) == 50, (
            f"expected 50 active memories, got {store.count(active_only=True)}"
        )
        # count(active_only=False) must match too (no inactive seeds)
        assert store.count(active_only=False) == 50
    finally:
        store.close()

    # --- app_config.json points at the migrated persona ---
    app_cfg_path = new_home / "app_config.json"
    assert app_cfg_path.is_file(), "app_config.json not written by migrator"
    cfg = json.loads(app_cfg_path.read_text(encoding="utf-8"))
    assert cfg["selected_persona"] == "phoebe"

    # --- source-manifest.json correctness ---
    manifest = json.loads((target / "source-manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_kind"] == "companion-emergence"
    assert manifest["memory_count"] == 50
