"""Tests for brain.memory.embeddings._materialize_symlinked_files /
_materialize_fastembed_model_dir -- the onnxruntime external-data
symlink workaround (#259 F1 model-swap increment).

These build a REAL symlink fixture mirroring HuggingFace's cache layout
(a blobs/ dir of real file content, a snapshot dir of symlinks into it) and
exercise the materialize step directly against the filesystem -- no model
download, no network, no fastembed import needed beyond the tiny stub shape
`_materialize_fastembed_model_dir` reaches into (`.model._model_dir`).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from brain.memory.embeddings import (
    _materialize_fastembed_model_dir,
    _materialize_symlinked_files,
)


def _make_snapshot_fixture(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """Build a scratch dir mirroring the HF cache layout: a blobs/ dir
    holding real file content, and a snapshot dir where model.onnx,
    model.onnx_data, and a nested tokenizer/tokenizer.json are SYMLINKS
    into two (for the onnx pair) or more DIFFERENT blobs, plus one
    already-real file (config.json) that must pass through untouched."""
    blobs = tmp_path / "cache_root" / "blobs"
    blobs.mkdir(parents=True)
    onnx_blob = blobs / "blob-onnx"
    onnx_blob.write_bytes(b"ONNX-MODEL-BYTES-AAAA")
    data_blob = blobs / "blob-onnx-data"
    data_blob.write_bytes(b"ONNX-EXTERNAL-DATA-SHARD-BBBB")
    tok_blob = blobs / "blob-tokenizer"
    tok_blob.write_bytes(b'{"tokenizer": true}')

    snapshot = tmp_path / "cache_root" / "snapshots" / "main"
    (snapshot / "tokenizer").mkdir(parents=True)
    (snapshot / "model.onnx").symlink_to(onnx_blob)
    (snapshot / "model.onnx_data").symlink_to(data_blob)
    (snapshot / "tokenizer" / "tokenizer.json").symlink_to(tok_blob)
    (snapshot / "config.json").write_bytes(b'{"real": "file"}')

    contents = {
        "model.onnx": onnx_blob.read_bytes(),
        "model.onnx_data": data_blob.read_bytes(),
        "tokenizer/tokenizer.json": tok_blob.read_bytes(),
        "config.json": b'{"real": "file"}',
    }
    return snapshot, contents


def test_materialize_symlinked_files_resolves_real_hf_cache_symlinks(tmp_path: Path) -> None:
    """Every symlink in the fixture becomes a real, non-symlinked file with
    the correct bytes; the pre-existing real file (config.json) passes
    through untouched; model.onnx and model.onnx_data end up as real
    SIBLINGS in the same directory (the whole point: onnxruntime resolves
    the external-data reference relative to the model file's own dir)."""
    snapshot, contents = _make_snapshot_fixture(tmp_path)

    materialized = _materialize_symlinked_files(snapshot)

    assert materialized != snapshot
    for rel, expected in contents.items():
        dest = materialized / rel
        assert dest.exists(), f"{rel} missing from materialized dir"
        assert not dest.is_symlink(), f"{rel} is still a symlink after materialize"
        assert dest.read_bytes() == expected

    assert (materialized / "model.onnx").parent == (materialized / "model.onnx_data").parent


def test_materialize_symlinked_files_second_call_is_idempotent_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second materialize call over the same snapshot must skip every
    already-materialized file (Finding A: the skip check compares the
    dest's inode/samefile identity against the real blob, not just size) --
    proven here by making os.link/shutil.copy2 raise if called AT ALL on
    the second pass, since a genuine skip never reaches either."""
    snapshot, contents = _make_snapshot_fixture(tmp_path)

    first = _materialize_symlinked_files(snapshot)
    assert first.exists()

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("materialize should have skipped every file on the second call")

    monkeypatch.setattr(os, "link", _boom)
    monkeypatch.setattr(shutil, "copy2", _boom)

    second = _materialize_symlinked_files(snapshot)

    assert second == first
    for rel, expected in contents.items():
        assert (second / rel).read_bytes() == expected


def test_materialize_symlinked_files_refreshes_a_stale_same_size_dest(tmp_path: Path) -> None:
    """Finding A: a stale REAL dest file of the SAME size as the real blob
    but with DIFFERENT bytes must be refreshed, not kept. A size-only skip
    check (the pre-fix behavior) would wrongly treat this as "already
    materialized" and leave the stale bytes in place; the inode/samefile
    check must reject it and re-materialize from the real blob."""
    snapshot, contents = _make_snapshot_fixture(tmp_path)
    onnx_bytes = contents["model.onnx"]

    materialized = snapshot.parent / f"{snapshot.name}.materialized"
    materialized.mkdir(parents=True)
    stale = materialized / "model.onnx"
    stale_bytes = bytes((b + 1) % 256 for b in onnx_bytes)
    assert len(stale_bytes) == len(onnx_bytes)
    assert stale_bytes != onnx_bytes
    stale.write_bytes(stale_bytes)

    result = _materialize_symlinked_files(snapshot)

    assert result == materialized
    assert (result / "model.onnx").read_bytes() == onnx_bytes, (
        "a stale same-size dest with different bytes must be refreshed, not kept"
    )


def test_materialize_fastembed_model_dir_repoints_internal_model_dir(tmp_path: Path) -> None:
    """`_materialize_fastembed_model_dir` reaches into a
    TextEmbedding-shaped stub's `.model._model_dir` and repoints it at the
    materialized (real-file) directory when the original held symlinks."""
    snapshot, contents = _make_snapshot_fixture(tmp_path)

    class _Inner:
        pass

    class _Stub:
        pass

    inner = _Inner()
    inner._model_dir = str(snapshot)  # noqa: SLF001 -- mirrors fastembed's real internal shape
    stub = _Stub()
    stub.model = inner

    _materialize_fastembed_model_dir(stub)

    new_dir = Path(inner._model_dir)  # noqa: SLF001
    assert new_dir != snapshot
    assert new_dir == snapshot.parent / f"{snapshot.name}.materialized"
    for rel, expected in contents.items():
        assert (new_dir / rel).read_bytes() == expected
