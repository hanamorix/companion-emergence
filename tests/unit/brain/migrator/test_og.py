"""Tests for brain.migrator.og — OG data readers + manifest."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from brain.migrator.og import (
    FileManifest,
    LiveLockDetected,
    OGReader,
)


@pytest.fixture
def og_dir(tmp_path: Path) -> Path:
    """Minimal OG-shaped fixture: 2 memories, 2x2 hebbian, ids file."""
    og = tmp_path / "og_data"
    og.mkdir()

    memories = [
        {
            "id": "m1",
            "content": "first",
            "memory_type": "conversation",
            "domain": "us",
            "created_at": "2024-01-01T00:00:00+00:00",
            "emotions": {"love": 9.0},
        },
        {
            "id": "m2",
            "content": "second",
            "memory_type": "meta",
            "domain": "work",
            "created_at": "2024-02-01T00:00:00+00:00",
            "emotions": {},
        },
    ]
    (og / "memories_v2.json").write_text(json.dumps(memories))

    ids = ["m1", "m2"]
    (og / "connection_matrix_ids.json").write_text(json.dumps(ids))

    matrix = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
    np.save(og / "connection_matrix.npy", matrix)

    (og / "hebbian_state.json").write_text(json.dumps({"version": 1}))

    return og


def test_reader_reads_memories(og_dir: Path) -> None:
    """read_memories returns the list of OG memory dicts."""
    reader = OGReader(og_dir)
    memories = reader.read_memories()
    assert len(memories) == 2
    assert memories[0]["id"] == "m1"
    assert memories[0]["content"] == "first"


def test_reader_reads_hebbian_matrix(og_dir: Path) -> None:
    """read_hebbian returns (ids, matrix) tuple."""
    reader = OGReader(og_dir)
    ids, matrix = reader.read_hebbian()
    assert ids == ["m1", "m2"]
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == pytest.approx(0.5)


def test_reader_iter_nonzero_upper_edges(og_dir: Path) -> None:
    """iter_nonzero_upper_edges yields (id_a, id_b, weight) for i<j nonzero."""
    reader = OGReader(og_dir)
    edges = list(reader.iter_nonzero_upper_edges())
    assert len(edges) == 1
    a, b, w = edges[0]
    assert a == "m1"
    assert b == "m2"
    assert w == pytest.approx(0.5)


def test_reader_manifest_lists_all_source_files(og_dir: Path) -> None:
    """manifest() returns a FileManifest for every OG file consumed."""
    reader = OGReader(og_dir)
    reader.read_memories()
    reader.read_hebbian()
    manifest = reader.manifest()

    paths = {m.relative_path for m in manifest}
    assert "memories_v2.json" in paths
    assert "connection_matrix.npy" in paths
    assert "connection_matrix_ids.json" in paths
    assert "hebbian_state.json" in paths


def test_reader_manifest_records_sha256_size_mtime(og_dir: Path) -> None:
    """Each manifest entry has sha256, size_bytes, mtime_utc."""
    reader = OGReader(og_dir)
    reader.read_memories()
    manifest = reader.manifest()
    mem_entry: FileManifest = next(m for m in manifest if m.relative_path == "memories_v2.json")

    expected_sha = hashlib.sha256((og_dir / "memories_v2.json").read_bytes()).hexdigest()
    assert mem_entry.sha256 == expected_sha
    assert mem_entry.size_bytes == (og_dir / "memories_v2.json").stat().st_size
    assert "T" in mem_entry.mtime_utc  # ISO 8601 marker


def test_reader_raises_if_memories_lock_is_recent(og_dir: Path) -> None:
    """If memories_v2.json.lock is recent (< 90s — 1.5x OG's 60s tick),
    raise LiveLockDetected. Bridge is almost certainly actively writing."""
    (og_dir / "memories_v2.json.lock").write_bytes(b"")
    reader = OGReader(og_dir)
    with pytest.raises(LiveLockDetected):
        reader.check_preflight()


def test_reader_preflight_ok_when_lock_is_stale(og_dir: Path) -> None:
    """A lock file older than 90 seconds is treated as stale (no error)."""
    lock = og_dir / "memories_v2.json.lock"
    lock.write_bytes(b"")
    old_time = time.time() - 3600  # 1 hour ago
    os.utime(lock, (old_time, old_time))

    reader = OGReader(og_dir)
    reader.check_preflight()  # should not raise


def test_reader_preflight_threshold_is_90s_not_300s(og_dir: Path) -> None:
    """M-8 (audit-2 follow-up): threshold tightened from 300s to 90s.
    Lock at 120s old must now be considered stale (was previously live)."""
    lock = og_dir / "memories_v2.json.lock"
    lock.write_bytes(b"")
    old_time = time.time() - 120  # 2 minutes ago — under old 300s threshold, over new 90s
    os.utime(lock, (old_time, old_time))

    reader = OGReader(og_dir)
    reader.check_preflight()  # 120s > 90s → stale, no raise


def test_reader_preflight_force_skips_check(og_dir: Path) -> None:
    """check_preflight(force=True) bypasses the lock detection entirely.
    Wired to --force-preflight at the CLI for operators who know the lock
    is stale (e.g. lock file from a prior crash)."""
    (og_dir / "memories_v2.json.lock").write_bytes(b"")  # fresh lock
    reader = OGReader(og_dir)
    # Without force: would raise
    with pytest.raises(LiveLockDetected):
        reader.check_preflight()
    # With force: bypass
    reader.check_preflight(force=True)


def test_reader_preflight_ok_when_no_lock(og_dir: Path) -> None:
    """No lock file → preflight passes silently."""
    reader = OGReader(og_dir)
    reader.check_preflight()  # should not raise


def test_reader_raises_if_og_dir_missing(tmp_path: Path) -> None:
    """OGReader(nonexistent_dir) raises a clear error on first read."""
    with pytest.raises(FileNotFoundError):
        OGReader(tmp_path / "nope").read_memories()
