"""P0 image-tool-route — brain.file_store (general non-image content store).

Mirrors brain.images' idempotent tmp+os.replace+dedup pattern. The on-disk
name is the validated 64-hex content sha ONLY — no client filename/extension
is ever a path component (path-traversal defense, C17).
"""

from __future__ import annotations

import hashlib

import pytest

from brain import file_store


def test_save_file_bytes_stores_under_sha_only(tmp_path):
    data = b"hello world, a non-image file"
    sha = hashlib.sha256(data).hexdigest()
    rec = file_store.save_file_bytes(tmp_path, data)
    assert rec.sha == sha
    assert rec.size_bytes == len(data)
    stored = tmp_path / "files" / sha
    assert stored.exists()
    assert stored.read_bytes() == data
    # The only path component under files/ is the sha.
    assert [p.name for p in (tmp_path / "files").iterdir()] == [sha]


def test_save_file_bytes_dedupes_same_content(tmp_path):
    data = b"same bytes"
    r1 = file_store.save_file_bytes(tmp_path, data)
    r2 = file_store.save_file_bytes(tmp_path, data)
    assert r1.sha == r2.sha
    assert len(list((tmp_path / "files").iterdir())) == 1


def test_file_path_resolves_under_files_dir(tmp_path):
    sha = "a" * 64
    p = file_store.file_path(tmp_path, sha)
    assert p == tmp_path / "files" / sha
    assert p.parent == tmp_path / "files"


def test_file_path_rejects_non_sha(tmp_path):
    """A traversal-shaped 'sha' is rejected before any path is built (C17)."""
    for bad in ("../../etc/passwd", "a/b", "..", "g" * 64, "a" * 63):
        with pytest.raises(ValueError):
            file_store.file_path(tmp_path, bad)


def test_upload_max_bytes_is_at_least_20mb(tmp_path):
    assert file_store.upload_max_bytes() >= 20 * 1024 * 1024
