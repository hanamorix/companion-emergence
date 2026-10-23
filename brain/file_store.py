"""General (non-image) content-addressable file store.

Shares :func:`brain.utils.atomic_write.save_blob_once` with :mod:`brain.images`
(idempotent atomic write + dedup), but for arbitrary (non-image) uploads that a user hands the
companion so she can read them via the ``read_file`` MCP tool.

Content-addressable layout: ``<persona_dir>/files/<sha256>``.

Security — path-traversal (P0 red-team Finding 2):

* The on-disk name is the **validated 64-hex sha ONLY**. No
  client-supplied filename or extension is ever a filesystem path
  component, so a crafted upload filename (``../../etc/x``, ``a/b``)
  cannot escape ``<persona_dir>/files/``.
* The original filename, when known, is display metadata only — it is
  surfaced to the model in the shared-file path line but never used to
  build the path.

Because the on-disk key is pure content sha, dedup is by content alone.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from brain import tunables
from brain.utils.atomic_write import save_blob_once

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

# General upload cap. Defaults to >= the 20 MB /upload image cap so the
# two caps are coherent (no "upload succeeds but store refuses" dead zone).
_UPLOAD_MAX_BYTES = tunables.register(
    "files.upload_max_bytes", 20 * 1024 * 1024,
    description="Largest file you can upload, in bytes.",
)


def upload_max_bytes() -> int:
    """Current general-file upload byte cap (a tunable)."""
    return tunables.get_tunable("files.upload_max_bytes", _UPLOAD_MAX_BYTES)


@dataclass(frozen=True)
class FileRecord:
    """Result of a successful non-image file save.

    Attributes
    ----------
    sha:
        64-character lowercase hex sha256 of the file bytes.
    size_bytes:
        Length of the saved file's bytes (== len(input)).
    """

    sha: str
    size_bytes: int


def compute_sha(data: bytes) -> str:
    """Compute a 64-char lowercase hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _validate_sha(sha: str) -> None:
    """Raise ``ValueError`` unless ``sha`` is 64 lowercase hex chars."""
    if not _SHA256_HEX.fullmatch(sha):
        raise ValueError(f"file sha must be 64 lowercase hex chars, got {sha!r}")


def file_path(persona_dir: Path, sha: str) -> Path:
    """Return the canonical on-disk path for ``sha`` under ``persona_dir``.

    The path is derived only from the validated sha — never from a
    client-supplied filename — so it always resolves under
    ``<persona_dir>/files/`` and nowhere else. Does not check existence.
    """
    _validate_sha(sha)
    return persona_dir / "files" / sha


def save_file_bytes(persona_dir: Path, data: bytes) -> FileRecord:
    """Save ``data`` content-addressably under ``persona_dir/files/``.

    Atomic through :func:`brain.utils.atomic_write.save_blob_once`. If the
    target already exists with matching sha (deduplication), no write is
    performed and the existing record is returned.
    """
    sha = compute_sha(data)
    target = file_path(persona_dir, sha)
    # The shared content-addressed blob writer (the same one brain.images uses): dedup by name, a unique temp per
    # writer, fsync, a retried atomic replace, and success when a concurrent writer of the same content wins.
    save_blob_once(target, data)
    return FileRecord(sha=sha, size_bytes=len(data))
