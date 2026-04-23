"""OG NellBrain data readers — JSON + .npy files, SHA-256 manifest, live-lock pre-flight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_LIVE_LOCK_THRESHOLD_SECONDS = 5 * 60


class LiveLockDetected(Exception):  # noqa: N818
    """Raised when OG's bridge appears to be actively writing (recent lock file)."""


@dataclass(frozen=True)
class FileManifest:
    """Audit-trail entry for a single OG file consumed by the migrator."""

    relative_path: str
    size_bytes: int
    sha256: str
    mtime_utc: str


class OGReader:
    """Read-only access to an OG NellBrain `data/` directory.

    Records a FileManifest for every file actually consumed, to provide a
    cryptographic audit trail that no writes occurred. Manifest entries are
    computed lazily on first read; call `.manifest()` after all reads to
    retrieve the full list.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self._dir = Path(data_dir)
        self._manifests: dict[str, FileManifest] = {}

    def check_preflight(self) -> None:
        """Detect a live OG bridge and refuse to proceed.

        Raises LiveLockDetected if `memories_v2.json.lock` exists and its
        mtime is within the last 5 minutes (bridge is actively writing).
        Stale locks are tolerated.
        """
        lock = self._dir / "memories_v2.json.lock"
        if not lock.exists():
            return
        age_s = datetime.now(UTC).timestamp() - lock.stat().st_mtime
        if age_s < _LIVE_LOCK_THRESHOLD_SECONDS:
            raise LiveLockDetected(
                f"Recent lock file at {lock} (age {age_s:.0f}s < "
                f"{_LIVE_LOCK_THRESHOLD_SECONDS}s). Stop the OG bridge before migrating."
            )

    def read_memories(self) -> list[dict[str, Any]]:
        """Return the OG memories list as parsed JSON dicts."""
        path = self._dir / "memories_v2.json"
        data = self._read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a JSON list")
        return data

    def read_hebbian(self) -> tuple[list[str], np.ndarray]:
        """Return (ids, matrix) for the OG connection matrix.

        ids: list of memory ids, index-aligned with rows/cols of matrix.
        matrix: 2-D numpy array, typically float32.
        """
        ids_path = self._dir / "connection_matrix_ids.json"
        matrix_path = self._dir / "connection_matrix.npy"
        ids = self._read_json(ids_path)
        if not isinstance(ids, list):
            raise ValueError(f"{ids_path} is not a JSON list")

        matrix = np.load(matrix_path)
        self._record_manifest(matrix_path)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{matrix_path}: expected square 2-D matrix, got shape {matrix.shape}")
        if len(ids) != matrix.shape[0]:
            raise ValueError(f"id count {len(ids)} does not match matrix dim {matrix.shape[0]}")

        # Incidentally touch hebbian_state.json so it lands in the manifest,
        # even though its contents are not currently used.
        state_path = self._dir / "hebbian_state.json"
        if state_path.exists():
            self._read_json(state_path)

        return ids, matrix

    def iter_nonzero_upper_edges(self) -> Iterator[tuple[str, str, float]]:
        """Yield (id_a, id_b, weight) for upper-triangular (i<j) nonzero entries.

        i<j avoids double-counting undirected edges (matrix may or may not be
        symmetric in OG; upper-triangle is the canonical reading).
        """
        ids, matrix = self.read_hebbian()
        rows, cols = np.nonzero(matrix)
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            if i >= j:
                continue
            w = float(matrix[i, j])
            if w > 0.0:
                yield ids[i], ids[j], w

    def manifest(self) -> list[FileManifest]:
        """Return FileManifest entries for every file this reader has read so far."""
        return list(self._manifests.values())

    # --- internals ---

    def _read_json(self, path: Path) -> Any:
        with path.open("rb") as f:
            raw = f.read()
        self._record_manifest(path, raw)
        return json.loads(raw.decode("utf-8"))

    def _record_manifest(self, path: Path, raw: bytes | None = None) -> None:
        if str(path) in self._manifests:
            return
        if raw is None:
            raw = path.read_bytes()
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        self._manifests[str(path)] = FileManifest(
            relative_path=path.relative_to(self._dir).as_posix(),
            size_bytes=stat.st_size,
            sha256=hashlib.sha256(raw).hexdigest(),
            mtime_utc=mtime.isoformat().replace("+00:00", "Z"),
        )
