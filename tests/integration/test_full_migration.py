"""End-to-end migrator test — small fixture, full pipeline, real SQLite verification."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from brain.migrator.cli import MigrateArgs, run_migrate


@pytest.fixture
def og_mini(tmp_path: Path) -> Path:
    """Five OG-shaped memories + a 5x5 hebbian matrix. One memory is malformed."""
    og = tmp_path / "og_mini"
    og.mkdir()

    memories = [
        {
            "id": "m1",
            "content": "cold coffee, warm hana",
            "memory_type": "conversation",
            "domain": "us",
            "created_at": "2024-03-01T10:00:00+00:00",
            "emotions": {"love": 9.0, "tenderness": 8.0},
            "emotion_score": 17.0,
            "importance": 8.0,
            "tags": ["first", "important"],
            "source_date": "2024-03-01",
            "source_summary": "first kiss equivalent",
            "supersedes": None,
        },
        {
            "id": "m2",
            "content": "the evening has a shape to it now",
            "memory_type": "conversation",
            "domain": "us",
            "created_at": "2024-04-10T20:00:00+00:00",
            "emotions": {"anchor_pull": 7.0, "tenderness": 6.0},
            "emotion_score": 13.0,
        },
        {
            "id": "m3",
            "content": "creative hunger strikes unannounced",
            "memory_type": "meta",
            "domain": "craft",
            "created_at": "2024-05-15T14:30:00+00:00",
            "emotions": {"creative_hunger": 8.0, "defiance": 5.0},
            "emotion_score": 13.0,
        },
        # malformed: no content
        {
            "id": "m4",
            "content": "",
            "memory_type": "conversation",
            "domain": "us",
            "created_at": "2024-06-01T00:00:00+00:00",
            "emotions": {},
        },
        {
            "id": "m5",
            "content": "loose thread",
            "memory_type": "meta",
            "domain": "us",
            "created_at": "2024-07-01T00:00:00+00:00",
            "emotions": {"curiosity": 6.0},
            "emotion_score": 6.0,
        },
    ]
    (og / "memories_v2.json").write_text(json.dumps(memories))

    ids = ["m1", "m2", "m3", "m4", "m5"]
    (og / "connection_matrix_ids.json").write_text(json.dumps(ids))

    matrix = np.zeros((5, 5), dtype=np.float32)
    matrix[0, 1] = 0.8  # m1 - m2
    matrix[1, 2] = 0.3  # m2 - m3
    matrix[0, 4] = 0.5  # m1 - m5
    np.save(og / "connection_matrix.npy", matrix)

    (og / "hebbian_state.json").write_text(json.dumps({"version": 1}))
    return og


def test_full_migration_output_mode_produces_expected_counts(og_mini: Path, tmp_path: Path) -> None:
    """End-to-end: run migrator, open output dbs, verify counts + a specific record."""
    out = tmp_path / "migrated-mini"
    args = MigrateArgs(input_dir=og_mini, output_dir=out, install_as=None, force=False)
    report = run_migrate(args)

    # Report shape
    assert report.memories_migrated == 4  # 5 input - 1 malformed (m4)
    assert len(report.memories_skipped) == 1
    assert report.memories_skipped[0].reason == "missing_content"
    assert report.edges_migrated == 3

    # Artefacts present
    assert (out / "memories.db").exists()
    assert (out / "hebbian.db").exists()
    assert (out / "source-manifest.json").exists()
    assert (out / "migration-report.md").exists()

    # Open memories.db and sanity-check content
    conn = sqlite3.connect(out / "memories.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, content, domain FROM memories ORDER BY id").fetchall()
    conn.close()
    assert len(rows) == 4
    ids = [r["id"] for r in rows]
    assert ids == ["m1", "m2", "m3", "m5"]
    assert "cold coffee" in rows[0]["content"]

    # Open hebbian.db and check edge count
    conn = sqlite3.connect(out / "hebbian.db")
    (edge_count,) = conn.execute("SELECT COUNT(*) FROM hebbian_edges").fetchone()
    conn.close()
    assert edge_count == 3


def test_full_migration_metadata_absorbs_og_only_fields(og_mini: Path, tmp_path: Path) -> None:
    """m1 had source_date + source_summary + supersedes → all in metadata."""
    out = tmp_path / "migrated-mini"
    args = MigrateArgs(input_dir=og_mini, output_dir=out, install_as=None, force=False)
    run_migrate(args)

    conn = sqlite3.connect(out / "memories.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT metadata_json FROM memories WHERE id = 'm1'").fetchone()
    conn.close()

    metadata = json.loads(row["metadata_json"])
    assert metadata["source_date"] == "2024-03-01"
    assert metadata["source_summary"] == "first kiss equivalent"
    assert metadata["supersedes"] is None


def test_full_migration_source_manifest_records_all_files(og_mini: Path, tmp_path: Path) -> None:
    """source-manifest.json records all four OG files with sha256 prefixes."""
    out = tmp_path / "migrated-mini"
    args = MigrateArgs(input_dir=og_mini, output_dir=out, install_as=None, force=False)
    run_migrate(args)

    manifest = json.loads((out / "source-manifest.json").read_text())
    paths = {f["relative_path"] for f in manifest["files"]}
    assert paths == {
        "memories_v2.json",
        "connection_matrix.npy",
        "connection_matrix_ids.json",
        "hebbian_state.json",
    }
    for f in manifest["files"]:
        assert len(f["sha256"]) == 64
        assert f["size_bytes"] > 0


def test_full_migration_never_writes_to_og_dir(og_mini: Path, tmp_path: Path) -> None:
    """After the migrator runs, OG dir contents are byte-identical.

    This is the critical safety invariant. The migrator must never write to
    OG files under any code path — the only guarantor is this test.
    """
    # Snapshot before
    before = {p.name: (p.stat().st_size, p.read_bytes()) for p in og_mini.iterdir()}
    out = tmp_path / "migrated-mini"
    args = MigrateArgs(input_dir=og_mini, output_dir=out, install_as=None, force=False)
    run_migrate(args)

    after = {p.name: (p.stat().st_size, p.read_bytes()) for p in og_mini.iterdir()}
    assert before == after, "OG files must not be mutated"
