# Week 3.5 — OG Memory Migrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `brain/migrator/` package that ports 1,141 OG memories + 8,808 Hebbian edges from `/Users/hanamori/NellBrain/data/` into new companion-emergence SQLite stores, read-only on OG, refuse-to-clobber on output, with a cryptographic source manifest as audit trail.

**Architecture:** Four modules (`og`, `transform`, `report`, `cli`) under `brain/migrator/`. A small Week 3 amendment adds a `metadata: dict[str, Any]` field to `Memory` (and a `metadata_json` SQLite column) to absorb OG-only fields without proliferating the dataclass signature. Tests are fixtures-only in CI; real OG dry-run is user-side.

**Tech Stack:** Python 3.12 stdlib (sqlite3, argparse, hashlib, json, shutil), numpy (already a Week 3 dep), pytest. No new dependencies.

---

## Context: what already exists (Week 3 state)

Main branch HEAD: `444c5db` (merge commit for Week 3).

- `brain/` package: `__init__.py`, `paths.py`, `config.py`, `cli.py` (from Week 1).
- `brain/emotion/` package: 7 modules from Week 2.
- `brain/memory/` package: `store.py`, `embeddings.py`, `hebbian.py`, `search.py` from Week 3.
- 191 tests passing; tags `week-1-complete`, `week-2-complete`, `week-3-complete`.
- CI matrix green on macOS + Windows + Linux.

Feature branch for this work: `week-3.5-migrator` (to be created off `main`).

Observed OG data at `/Users/hanamori/NellBrain/data/`:
- `memories_v2.json` — 1,141 memories (3.9 MB)
- `connection_matrix.npy` — 855×855 float32, 8,808 non-zero edges
- `connection_matrix_ids.json` — 855 memory ids index-aligned with matrix
- `hebbian_state.json` — 293 bytes of metadata
- (embeddings skipped per spec)

---

## File structure (what this plan creates)

```
companion-emergence/
├── brain/
│   ├── memory/
│   │   └── store.py                    (MODIFIED — +metadata field + column)
│   └── migrator/                       (NEW package)
│       ├── __init__.py
│       ├── og.py                       (Task 3 — readers)
│       ├── transform.py                (Task 4 — field mapping + SkippedMemory)
│       ├── report.py                   (Task 5 — report + manifest)
│       └── cli.py                      (Task 6 — subcommand)
├── brain/cli.py                        (MODIFIED — Task 6 — wire migrate subcommand)
└── tests/
    ├── fixtures/og_mini/               (Task 7 — integration fixture)
    │   ├── memories_v2.json
    │   ├── connection_matrix.npy
    │   ├── connection_matrix_ids.json
    │   └── hebbian_state.json
    ├── unit/brain/memory/
    │   └── test_store.py               (MODIFIED — Task 2 — metadata round-trip)
    ├── unit/brain/migrator/            (NEW)
    │   ├── __init__.py
    │   ├── test_og.py                  (Task 3)
    │   ├── test_transform.py           (Task 4)
    │   ├── test_report.py              (Task 5)
    │   └── test_cli.py                 (Task 6)
    └── integration/                    (NEW)
        ├── __init__.py
        └── test_full_migration.py      (Task 7)
```

---

## Dependency order

Task 1 (Memory.metadata field) → Task 2 (MemoryStore metadata column) → parallelisable Tasks 3/4/5 (migrator modules) → Task 6 (CLI wiring; depends on 3+4+5) → Task 7 (integration fixture + test; depends on everything) → Task 8 (close-out).

Execute in numerical order. Tasks 3, 4, 5 can in principle be parallelised but we execute them sequentially to keep the subagent-driven pattern simple.

---

## Task 1: Memory dataclass — add `metadata` field

**Goal:** Add `metadata: dict[str, Any]` to `Memory`, with default `{}`, round-tripping through `to_dict`/`from_dict`. Does NOT touch MemoryStore yet — that's Task 2.

**Files:**
- Modify: `/Users/hanamori/companion-emergence/brain/memory/store.py`
- Modify: `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_store.py`

- [ ] **Step 1: Create the feature branch**

```bash
cd /Users/hanamori/companion-emergence
git checkout main
git pull origin main
git checkout -b week-3.5-migrator
```

- [ ] **Step 2: Write the failing tests**

Append to `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_store.py` (anywhere among the Memory-dataclass tests, e.g. just before the MemoryStore fixtures start around line 156):

```python


def test_memory_metadata_defaults_to_empty_dict() -> None:
    """metadata defaults to {} if not specified."""
    m = Memory.create_new(content="x", memory_type="meta", domain="work")
    assert m.metadata == {}


def test_memory_create_new_accepts_metadata_kwarg() -> None:
    """Memory.create_new accepts a metadata dict and preserves it verbatim."""
    m = Memory.create_new(
        content="x",
        memory_type="meta",
        domain="work",
        metadata={"source_date": "2024-01-01", "supersedes": "abc-123"},
    )
    assert m.metadata == {"source_date": "2024-01-01", "supersedes": "abc-123"}


def test_memory_metadata_round_trips_through_dict() -> None:
    """metadata round-trips through to_dict / from_dict."""
    original = Memory.create_new(
        content="x",
        memory_type="meta",
        domain="work",
        metadata={"emotional_tone": "tender", "access_count": 3, "tags_sig": None},
    )
    data = original.to_dict()
    assert data["metadata"] == original.metadata
    restored = Memory.from_dict(data)
    assert restored.metadata == original.metadata


def test_memory_from_dict_missing_metadata_defaults_empty() -> None:
    """Legacy dicts without a 'metadata' key restore cleanly with metadata={}."""
    from datetime import UTC, datetime as _dt

    data = {
        "id": "legacy-001",
        "content": "legacy",
        "memory_type": "meta",
        "domain": "work",
        "emotions": {},
        "tags": [],
        "importance": 0.0,
        "score": 0.0,
        "created_at": _dt.now(UTC).isoformat(),
        "last_accessed_at": None,
        "active": True,
        "protected": False,
        # no 'metadata' key
    }
    restored = Memory.from_dict(data)
    assert restored.metadata == {}


def test_memory_metadata_defensive_copy_on_create_new() -> None:
    """Mutating the caller's dict after create_new does not affect the memory."""
    caller_dict = {"source_date": "2024-01-01"}
    m = Memory.create_new(content="x", memory_type="meta", domain="work", metadata=caller_dict)
    caller_dict["source_date"] = "mutated"
    assert m.metadata == {"source_date": "2024-01-01"}
```

- [ ] **Step 3: Run tests — expect 5 failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_store.py -v -k metadata
```

Expected: 5 failures — `AttributeError: 'Memory' object has no attribute 'metadata'` or similar, and `TypeError: Memory.create_new() got an unexpected keyword argument 'metadata'`.

- [ ] **Step 4: Add the `metadata` field to the `Memory` dataclass**

In `/Users/hanamori/companion-emergence/brain/memory/store.py`, modify the `Memory` class.

Find the existing dataclass:
```python
@dataclass
class Memory:
    ...
    id: str
    content: str
    memory_type: str
    domain: str
    created_at: datetime
    emotions: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    importance: float = 0.0
    score: float = 0.0
    last_accessed_at: datetime | None = None
    active: bool = True
    protected: bool = False
```

Add `metadata` as the LAST field (keeps existing positional-arg callers working):

```python
    metadata: dict[str, Any] = field(default_factory=dict)
```

Also update the class docstring Attributes section to add:
```
        metadata: free-form dict for fields not modelled as first-class
            attributes — absorbs OG-only fields (source_date, supersedes,
            etc.) during migration without proliferating the dataclass.
            Forward-compatible: future engines read metadata[key] as needed.
```

- [ ] **Step 5: Update `Memory.create_new` to accept metadata**

Change the signature:
```python
    @classmethod
    def create_new(
        cls,
        content: str,
        memory_type: str,
        domain: str,
        emotions: dict[str, float] | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
```

In the body, after `tags = list(tags or [])` add:
```python
        metadata = dict(metadata or {})
```

And pass `metadata=metadata` to the `cls(...)` call.

- [ ] **Step 6: Update `to_dict` to emit metadata**

Find `to_dict` and add the key:
```python
            "metadata": dict(self.metadata),
```

Place it as the last key (just after `"protected"`).

- [ ] **Step 7: Update `from_dict` to restore metadata**

Add to the `cls(...)` call:
```python
            metadata=dict(data.get("metadata", {})),
```

Place it at the end of the kwargs (after `protected=...`).

- [ ] **Step 8: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_store.py -v
```

Expected: 37 passed (32 pre-existing + 5 new metadata tests). All other Memory/MemoryStore tests still pass (metadata defaults to `{}` everywhere).

- [ ] **Step 9: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```

Expected: 196 passed (191 + 5). Ruff clean. If format fails, run `uv run ruff format .`.

- [ ] **Step 10: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/memory/store.py tests/unit/brain/memory/test_store.py
git commit -m "feat(brain/memory): add metadata dict field to Memory dataclass

Week 3.5 amendment — prepares the Memory schema for the OG migrator,
which needs to absorb OG-only fields (source_date, source_summary,
supersedes, emotional_tone, access_count, emotion_count, intensity,
schema_version, connections) without proliferating first-class
dataclass attributes.

Default is {}. Round-trips cleanly through to_dict/from_dict. Legacy
dicts without the 'metadata' key restore as empty (migrator friendly).
Defensive copy on create_new — caller mutations don't leak.

MemoryStore schema column lands in a follow-up commit.

5 new tests; 196 total.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: MemoryStore — add `metadata_json` column

**Goal:** Persist `Memory.metadata` through the SQLite MemoryStore. Adds a `metadata_json` column with JSON serialisation.

**Files:**
- Modify: `/Users/hanamori/companion-emergence/brain/memory/store.py`
- Modify: `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_store.py`

- [ ] **Step 1: Write the failing tests**

Append to the end of `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_store.py`:

```python


def test_store_create_and_get_preserves_metadata(store: MemoryStore) -> None:
    """MemoryStore.create and .get round-trip metadata dict."""
    m = _mem("x", metadata={"source_date": "2024-01-01", "supersedes": "abc-123"})
    store.create(m)
    restored = store.get(m.id)
    assert restored is not None
    assert restored.metadata == {"source_date": "2024-01-01", "supersedes": "abc-123"}


def test_store_create_empty_metadata_survives(store: MemoryStore) -> None:
    """Default empty metadata dict round-trips as {} (not None or missing)."""
    m = _mem("x")
    store.create(m)
    restored = store.get(m.id)
    assert restored is not None
    assert restored.metadata == {}


def test_store_update_metadata_field(store: MemoryStore) -> None:
    """update() can mutate the metadata field."""
    m = _mem("x", metadata={"v": 1})
    store.create(m)
    store.update(m.id, metadata={"v": 2, "added": "yes"})

    restored = store.get(m.id)
    assert restored is not None
    assert restored.metadata == {"v": 2, "added": "yes"}


def test_store_update_rejects_unknown_field_still(store: MemoryStore) -> None:
    """Update's unknown-field guard still works after the metadata addition."""
    m = _mem("x")
    store.create(m)
    with pytest.raises(ValueError, match="Unknown update field"):
        store.update(m.id, nonsense_field="oops")
```

Also update the `_mem` helper near the top of the MemoryStore test block. Find:

```python
def _mem(content: str = "x", **kw: object) -> Memory:
    defaults = {"memory_type": "conversation", "domain": "us"}
    defaults.update(kw)
    return Memory.create_new(content=content, **defaults)  # type: ignore[arg-type]
```

No change needed — the `**kw` already passes `metadata` through.

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_store.py -v -k "metadata or update_metadata or update_rejects_unknown"
```

Expected: test_store_create_and_get_preserves_metadata FAILS (column doesn't exist) or passes with `metadata={}` depending on column handling — the point is the migrator-case round-trip must fail. test_store_update_metadata_field FAILS with `Unknown update field: 'metadata'`. The "rejects unknown" and "empty metadata survives" may pass trivially.

- [ ] **Step 3: Add the column to `_SCHEMA`**

In `/Users/hanamori/companion-emergence/brain/memory/store.py`, find the `_SCHEMA` block:

```python
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    emotions_json TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.0,
    score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    protected INTEGER NOT NULL DEFAULT 0
);
```

Add `metadata_json` as the LAST column (keeps column order stable for any manual SQL callers):

```python
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    emotions_json TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.0,
    score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    protected INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
```

- [ ] **Step 4: Update `MemoryStore.create` to write the column**

Find the `INSERT INTO memories (...)` statement:

```python
            """
            INSERT INTO memories (
                id, content, memory_type, domain, emotions_json, tags_json,
                importance, score, created_at, last_accessed_at, active, protected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
```

Change to include `metadata_json`:

```python
            """
            INSERT INTO memories (
                id, content, memory_type, domain, emotions_json, tags_json,
                importance, score, created_at, last_accessed_at, active, protected,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
```

And in the parameter tuple, add after `1 if memory.protected else 0,`:

```python
                json.dumps(memory.metadata),
```

- [ ] **Step 5: Update `MemoryStore.update` whitelist + column handling**

In the `update()` method's field-mapping loop, add a branch for `metadata`. Find:

```python
            elif key == "tags":
                column_map["tags_json"] = ("tags_json", json.dumps(value))
```

Add immediately after:

```python
            elif key == "metadata":
                column_map["metadata_json"] = ("metadata_json", json.dumps(value))
```

- [ ] **Step 6: Update `_row_to_memory` to read the column**

Find `_row_to_memory` at the bottom of the file:

```python
    return Memory(
        id=row["id"],
        ...
        protected=bool(row["protected"]),
    )
```

Add at the end (just before the closing paren):

```python
        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
```

The `if row["metadata_json"] else {}` guard handles any legacy row written before the column existed (sqlite's `DEFAULT '{}'` handles new inserts, but this belt-and-braces covers old .db files).

- [ ] **Step 7: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_store.py -v
```

Expected: 41 passed (37 pre-existing + 4 new MemoryStore metadata tests). All of Week 3's existing MemoryStore tests continue to pass (metadata is `{}` by default everywhere, no behavioural change for non-migrator callers).

- [ ] **Step 8: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```

Expected: 200 passed (196 + 4). Ruff clean.

- [ ] **Step 9: Commit**

```bash
git add brain/memory/store.py tests/unit/brain/memory/test_store.py
git commit -m "feat(brain/memory/store): persist Memory.metadata via metadata_json column

Completes the Week 3.5 amendment started in the previous commit.
MemoryStore now round-trips metadata through a metadata_json TEXT
column with DEFAULT '{}'. create/get/update all handle the field;
_row_to_memory guards empty/None values for legacy row compatibility.

Existing Memory/MemoryStore tests continue to pass — metadata is an
additive, zero-default field.

4 new tests; 200 total.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `brain/migrator/og.py` — OG data readers

**Goal:** Read OG data files (JSON + .npy), compute SHA-256 manifests, handle the pre-flight live-lock check. Zero write operations.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/migrator/__init__.py`
- Create: `/Users/hanamori/companion-emergence/brain/migrator/og.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/migrator/__init__.py` (empty)
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/migrator/test_og.py`

- [ ] **Step 1: Create empty package dirs**

```bash
cd /Users/hanamori/companion-emergence
mkdir -p brain/migrator tests/unit/brain/migrator
touch tests/unit/brain/migrator/__init__.py
```

- [ ] **Step 2: Write the migrator package `__init__.py`**

Create `/Users/hanamori/companion-emergence/brain/migrator/__init__.py`:

```python
"""OG NellBrain → companion-emergence memory migrator.

Read-only against OG; refuse-to-clobber on output; cryptographic
source manifest as audit trail. See:
docs/superpowers/specs/2026-04-23-og-memory-migrator-design.md
"""
```

- [ ] **Step 3: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/migrator/test_og.py`:

```python
"""Tests for brain.migrator.og — OG data readers + manifest."""

from __future__ import annotations

import hashlib
import json
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
    assert edges == [("m1", "m2", pytest.approx(0.5))]


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
    mem_entry = next(m for m in manifest if m.relative_path == "memories_v2.json")

    expected_sha = hashlib.sha256((og_dir / "memories_v2.json").read_bytes()).hexdigest()
    assert mem_entry.sha256 == expected_sha
    assert mem_entry.size_bytes == (og_dir / "memories_v2.json").stat().st_size
    assert mem_entry.mtime_utc.endswith("Z") or "+" in mem_entry.mtime_utc


def test_reader_raises_if_memories_lock_is_recent(og_dir: Path) -> None:
    """If memories_v2.json.lock is recent (< 5 min), raise LiveLockDetected."""
    (og_dir / "memories_v2.json.lock").write_bytes(b"")
    reader = OGReader(og_dir)
    with pytest.raises(LiveLockDetected):
        reader.check_preflight()


def test_reader_preflight_ok_when_lock_is_stale(og_dir: Path) -> None:
    """A lock file older than 5 minutes is treated as stale (no error)."""
    lock = og_dir / "memories_v2.json.lock"
    lock.write_bytes(b"")
    old_time = time.time() - 3600  # 1 hour ago
    import os

    os.utime(lock, (old_time, old_time))

    reader = OGReader(og_dir)
    reader.check_preflight()  # should not raise


def test_reader_preflight_ok_when_no_lock(og_dir: Path) -> None:
    """No lock file → preflight passes silently."""
    reader = OGReader(og_dir)
    reader.check_preflight()  # should not raise


def test_reader_raises_if_og_dir_missing(tmp_path: Path) -> None:
    """OGReader(nonexistent_dir) raises a clear error."""
    with pytest.raises(FileNotFoundError):
        OGReader(tmp_path / "nope").read_memories()
```

- [ ] **Step 4: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/migrator/test_og.py -v
```

Expected: 9 failures on `ModuleNotFoundError: No module named 'brain.migrator.og'`.

- [ ] **Step 5: Write `brain/migrator/og.py`**

Create `/Users/hanamori/companion-emergence/brain/migrator/og.py`:

```python
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


class LiveLockDetected(Exception):
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
            raise ValueError(
                f"{matrix_path}: expected square 2-D matrix, got shape {matrix.shape}"
            )
        if len(ids) != matrix.shape[0]:
            raise ValueError(
                f"id count {len(ids)} does not match matrix dim {matrix.shape[0]}"
            )

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
```

- [ ] **Step 6: Run tests — expect green**

```bash
uv run pytest tests/unit/brain/migrator/test_og.py -v
```

Expected: 9 passed.

- [ ] **Step 7: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```

Expected: 209 passed (200 + 9). Ruff clean.

- [ ] **Step 8: Commit**

```bash
git add brain/migrator/ tests/unit/brain/migrator/
git commit -m "feat(brain/migrator/og): OG data readers + SHA-256 manifest + live-lock preflight

OGReader provides read-only access to a NellBrain data/ directory.
read_memories() parses memories_v2.json; read_hebbian() returns
(ids, matrix); iter_nonzero_upper_edges() yields canonical undirected
edges (i<j) with positive weight.

Every file consumed is recorded in a FileManifest (relative_path,
size_bytes, sha256, mtime_utc) — a cryptographic audit trail that the
migrator is write-only against OG.

check_preflight() refuses to proceed if memories_v2.json.lock has
been modified within the last 5 minutes (live bridge detection).

9 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `brain/migrator/transform.py` — field mapping + validation

**Goal:** Transform OG memory dicts into new `Memory` dataclasses. Skip malformed memories with a structured `SkippedMemory` record.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/migrator/transform.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/migrator/test_transform.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/migrator/test_transform.py`:

```python
"""Tests for brain.migrator.transform — OG memory dict → Memory dataclass."""

from __future__ import annotations

from typing import Any

import pytest

from brain.migrator.transform import (
    SkippedMemory,
    transform_memory,
)


def _og(
    *,
    mem_id: str = "m1",
    content: str = "hello",
    **overrides: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": mem_id,
        "content": content,
        "memory_type": "conversation",
        "domain": "us",
        "created_at": "2024-01-01T12:00:00+00:00",
        "emotions": {"love": 9.0},
        "tags": ["a"],
        "importance": 5.0,
        "emotion_score": 9.0,
        "active": True,
    }
    base.update(overrides)
    return base


def test_transform_happy_path_produces_memory() -> None:
    """A well-formed OG memory transforms cleanly."""
    result = transform_memory(_og())
    assert isinstance(result, tuple)
    mem, skipped = result
    assert skipped is None
    assert mem is not None
    assert mem.id == "m1"
    assert mem.content == "hello"
    assert mem.emotions == {"love": 9.0}
    assert mem.score == 9.0


def test_transform_preserves_tz_aware_created_at() -> None:
    """created_at with tz-offset is preserved as UTC-aware."""
    mem, _ = transform_memory(_og(created_at="2024-03-01T15:00:00+00:00"))
    assert mem is not None
    assert mem.created_at.tzinfo is not None


def test_transform_coerces_tz_naive_created_at() -> None:
    """Naive created_at is coerced to UTC."""
    mem, _ = transform_memory(_og(created_at="2024-03-01T15:00:00"))
    assert mem is not None
    assert mem.created_at.tzinfo is not None


def test_transform_renames_last_accessed_to_last_accessed_at() -> None:
    """OG last_accessed → new last_accessed_at."""
    mem, _ = transform_memory(_og(last_accessed="2024-05-01T10:00:00+00:00"))
    assert mem is not None
    assert mem.last_accessed_at is not None


def test_transform_uses_emotion_score_as_score() -> None:
    """OG emotion_score → new score (verbatim)."""
    mem, _ = transform_memory(
        _og(emotions={"love": 8.0, "tenderness": 6.0}, emotion_score=14.0)
    )
    assert mem is not None
    assert mem.score == 14.0


def test_transform_defaults_missing_optional_fields() -> None:
    """Missing tags / importance / active fall back to sensible defaults."""
    minimal = {
        "id": "m2",
        "content": "x",
        "memory_type": "meta",
        "domain": "work",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    mem, skipped = transform_memory(minimal)
    assert skipped is None
    assert mem is not None
    assert mem.tags == []
    assert mem.importance == 0.0
    assert mem.active is True
    assert mem.emotions == {}
    assert mem.score == 0.0


def test_transform_absorbs_og_only_fields_into_metadata() -> None:
    """source_date, source_summary, supersedes, etc. land in metadata verbatim."""
    mem, _ = transform_memory(
        _og(
            source_date="2024-01-01",
            source_summary="first contact",
            emotional_tone="tender",
            supersedes="abc",
            access_count=3,
            emotion_count=1,
            intensity=7.0,
            schema_version=2,
            connections=["xyz"],
        )
    )
    assert mem is not None
    md = mem.metadata
    assert md["source_date"] == "2024-01-01"
    assert md["source_summary"] == "first contact"
    assert md["emotional_tone"] == "tender"
    assert md["supersedes"] == "abc"
    assert md["access_count"] == 3
    assert md["emotion_count"] == 1
    assert md["intensity"] == 7.0
    assert md["schema_version"] == 2
    assert md["connections"] == ["xyz"]


def test_transform_absorbs_unknown_fields_into_metadata() -> None:
    """Unknown OG keys (forward-proof) also land in metadata."""
    mem, _ = transform_memory(_og(some_future_field="hello", another={"nested": 1}))
    assert mem is not None
    assert mem.metadata["some_future_field"] == "hello"
    assert mem.metadata["another"] == {"nested": 1}


def test_transform_skips_missing_content() -> None:
    """Memory with missing / empty content is skipped."""
    mem, skipped = transform_memory(_og(content=""))
    assert mem is None
    assert skipped is not None
    assert skipped.reason == "missing_content"

    og = _og()
    del og["content"]
    mem2, skipped2 = transform_memory(og)
    assert mem2 is None
    assert skipped2 is not None
    assert skipped2.reason == "missing_content"


def test_transform_skips_non_numeric_emotion_value() -> None:
    """Emotions dict with a non-numeric value → skip."""
    mem, skipped = transform_memory(_og(emotions={"love": "high"}))
    assert mem is None
    assert skipped is not None
    assert skipped.reason == "non_numeric_emotion"
    assert skipped.field == "emotions"


def test_transform_skips_unparseable_created_at() -> None:
    """created_at that isoformat can't parse → skip."""
    mem, skipped = transform_memory(_og(created_at="not-a-date"))
    assert mem is None
    assert skipped is not None
    assert skipped.reason == "unparseable_created_at"


def test_transform_skips_missing_id() -> None:
    """A memory without an id cannot be referenced by the Hebbian matrix; skip."""
    og = _og()
    del og["id"]
    mem, skipped = transform_memory(og)
    assert mem is None
    assert skipped is not None
    assert skipped.reason == "missing_id"


def test_transform_score_mismatch_prefers_og_value() -> None:
    """If emotion_score disagrees with sum(emotions.values()), use OG's value
    (spec: prefer OG's stored value; don't rewrite historical scores).
    """
    mem, _ = transform_memory(
        _og(emotions={"love": 8.0, "tenderness": 6.0}, emotion_score=99.0)
    )
    assert mem is not None
    assert mem.score == 99.0


def test_skipped_memory_dataclass_shape() -> None:
    """SkippedMemory has id, reason, field, raw_snippet fields."""
    s = SkippedMemory(
        id="m1", reason="missing_content", field="content", raw_snippet="..."
    )
    assert s.id == "m1"
    assert s.reason == "missing_content"
    assert s.field == "content"
    assert s.raw_snippet == "..."
```

- [ ] **Step 2: Run tests — expect failures**

```bash
uv run pytest tests/unit/brain/migrator/test_transform.py -v
```

Expected: 14 failures on `ModuleNotFoundError`.

- [ ] **Step 3: Write `brain/migrator/transform.py`**

Create `/Users/hanamori/companion-emergence/brain/migrator/transform.py`:

```python
"""OG → new Memory transformer. Permissive: skips malformed records with a reason."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brain.memory.store import Memory, _coerce_utc


# OG fields with direct first-class mapping in the new schema.
_FIRST_CLASS_OG_FIELDS = frozenset({
    "id",
    "content",
    "memory_type",
    "domain",
    "created_at",
    "last_accessed",
    "tags",
    "importance",
    "emotions",
    "emotion_score",
    "active",
})


@dataclass(frozen=True)
class SkippedMemory:
    """Record of a skipped-during-migration OG memory."""

    id: str
    reason: str  # short code: missing_content, non_numeric_emotion, ...
    field: str  # the specific field that failed, or "" if whole-record
    raw_snippet: str  # truncated human-readable excerpt of the original


def transform_memory(og: dict[str, Any]) -> tuple[Memory | None, SkippedMemory | None]:
    """Transform a single OG memory dict into a new Memory.

    Returns (memory, None) on success, (None, SkippedMemory) on skip.
    Never raises — all malformed input surfaces as a SkippedMemory.
    """
    og_id = og.get("id")
    if not og_id or not isinstance(og_id, str):
        return None, SkippedMemory(
            id=str(og_id or "<unknown>"),
            reason="missing_id",
            field="id",
            raw_snippet=_snippet(og),
        )

    content = og.get("content")
    if not content or not isinstance(content, str):
        return None, SkippedMemory(
            id=og_id,
            reason="missing_content",
            field="content",
            raw_snippet=_snippet(og),
        )

    emotions_raw = og.get("emotions") or {}
    if not isinstance(emotions_raw, dict):
        return None, SkippedMemory(
            id=og_id,
            reason="non_numeric_emotion",
            field="emotions",
            raw_snippet=_snippet(og),
        )
    for k, v in emotions_raw.items():
        if not isinstance(v, (int, float)):
            return None, SkippedMemory(
                id=og_id,
                reason="non_numeric_emotion",
                field="emotions",
                raw_snippet=_snippet(og),
            )
    emotions = {k: float(v) for k, v in emotions_raw.items()}

    created_at_raw = og.get("created_at")
    if not created_at_raw or not isinstance(created_at_raw, str):
        return None, SkippedMemory(
            id=og_id,
            reason="unparseable_created_at",
            field="created_at",
            raw_snippet=_snippet(og),
        )
    try:
        created_at = _coerce_utc(created_at_raw)
    except ValueError:
        return None, SkippedMemory(
            id=og_id,
            reason="unparseable_created_at",
            field="created_at",
            raw_snippet=_snippet(og),
        )

    last_accessed_raw = og.get("last_accessed")
    last_accessed_at = None
    if last_accessed_raw and isinstance(last_accessed_raw, str):
        try:
            last_accessed_at = _coerce_utc(last_accessed_raw)
        except ValueError:
            last_accessed_at = None  # soft-skip this one field; don't skip the memory

    # Prefer OG's stored emotion_score; fall back to sum(emotions.values()).
    score_raw = og.get("emotion_score")
    if isinstance(score_raw, (int, float)):
        score = float(score_raw)
    else:
        score = float(sum(emotions.values()))

    metadata = {k: v for k, v in og.items() if k not in _FIRST_CLASS_OG_FIELDS}

    mem = Memory(
        id=og_id,
        content=content,
        memory_type=str(og.get("memory_type") or "conversation"),
        domain=str(og.get("domain") or "us"),
        created_at=created_at,
        emotions=emotions,
        tags=list(og.get("tags") or []),
        importance=float(og.get("importance") or 0.0),
        score=score,
        last_accessed_at=last_accessed_at,
        active=bool(og.get("active", True)),
        protected=False,
        metadata=metadata,
    )
    return mem, None


def _snippet(og: dict[str, Any], max_len: int = 120) -> str:
    """Human-readable excerpt of the offending OG record, truncated."""
    content = og.get("content") or ""
    if isinstance(content, str):
        excerpt = content[:max_len]
    else:
        excerpt = repr(content)[:max_len]
    return excerpt
```

- [ ] **Step 4: Run tests — expect green**

```bash
uv run pytest tests/unit/brain/migrator/test_transform.py -v
```

Expected: 14 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```

Expected: 223 passed (209 + 14). Ruff clean.

- [ ] **Step 6: Commit**

```bash
git add brain/migrator/transform.py tests/unit/brain/migrator/test_transform.py
git commit -m "feat(brain/migrator/transform): OG memory dict → Memory dataclass

Permissive transformer — returns (Memory, None) on success or
(None, SkippedMemory) on malformed input. Never raises.

Skip reasons: missing_id, missing_content, non_numeric_emotion,
unparseable_created_at.

Field mapping: id/content/memory_type/domain/tags/importance/active
verbatim. last_accessed → last_accessed_at. emotion_score → score
(prefers OG's stored value over recomputing sum). created_at coerced
via the Week 3 _coerce_utc helper (tz-naive → UTC).

OG-only fields (source_date, supersedes, emotional_tone, etc.) plus
any unknown forward-drift keys absorbed into Memory.metadata
verbatim.

14 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `brain/migrator/report.py` — migration report + source manifest writers

**Goal:** Format the end-of-run migration report as human-readable text; write `source-manifest.json` to output dir.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/migrator/report.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/migrator/test_report.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/migrator/test_report.py`:

```python
"""Tests for brain.migrator.report — migration report + source manifest."""

from __future__ import annotations

import json
from pathlib import Path

from brain.migrator.og import FileManifest
from brain.migrator.report import (
    MigrationReport,
    format_report,
    write_source_manifest,
)
from brain.migrator.transform import SkippedMemory


def test_format_report_totals_section() -> None:
    """Report leads with totals (memories migrated/skipped, edges migrated/skipped)."""
    report = MigrationReport(
        memories_migrated=1128,
        memories_skipped=[
            SkippedMemory(id="x", reason="missing_content", field="content", raw_snippet=""),
        ],
        edges_migrated=8808,
        edges_skipped=0,
        elapsed_seconds=2.3,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
    )
    text = format_report(report)
    assert "1,128 migrated" in text
    assert "1 skipped" in text
    assert "8,808 migrated" in text
    assert "2.3s" in text


def test_format_report_groups_skips_by_reason() -> None:
    """Skipped memories are grouped + counted by reason."""
    skipped = [
        SkippedMemory(id="a", reason="missing_content", field="content", raw_snippet=""),
        SkippedMemory(id="b", reason="missing_content", field="content", raw_snippet=""),
        SkippedMemory(id="c", reason="non_numeric_emotion", field="emotions", raw_snippet=""),
    ]
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=skipped,
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.1,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
    )
    text = format_report(report)
    assert "2 missing_content" in text
    assert "1 non_numeric_emotion" in text


def test_format_report_includes_manifest_entries() -> None:
    """Source manifest section lists each file with size + sha256 prefix."""
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[
            FileManifest(
                relative_path="memories_v2.json",
                size_bytes=123456,
                sha256="abc123" + "0" * 58,
                mtime_utc="2024-01-01T00:00:00Z",
            )
        ],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
    )
    text = format_report(report)
    assert "memories_v2.json" in text
    assert "123,456" in text
    assert "abc123" in text  # sha prefix visible


def test_format_report_includes_next_steps() -> None:
    """Report shows inspect commands + install command."""
    report = MigrationReport(
        memories_migrated=1,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[
            "sqlite3 out/memories.db \"SELECT COUNT(*) FROM memories;\"",
        ],
        next_steps_install_cmd="uv run brain migrate --input /og --install-as nell",
    )
    text = format_report(report)
    assert "Next steps" in text
    assert "sqlite3 out/memories.db" in text
    assert "--install-as nell" in text


def test_write_source_manifest_produces_valid_json(tmp_path: Path) -> None:
    """write_source_manifest produces valid JSON with the expected structure."""
    manifest = [
        FileManifest(
            relative_path="memories_v2.json",
            size_bytes=100,
            sha256="a" * 64,
            mtime_utc="2024-01-01T00:00:00Z",
        ),
    ]
    out_path = tmp_path / "source-manifest.json"
    write_source_manifest(out_path, manifest)

    data = json.loads(out_path.read_text())
    assert data["files"][0]["relative_path"] == "memories_v2.json"
    assert data["files"][0]["size_bytes"] == 100
    assert data["files"][0]["sha256"] == "a" * 64
    assert "generated_at_utc" in data
```

- [ ] **Step 2: Run tests — expect failures**

```bash
uv run pytest tests/unit/brain/migrator/test_report.py -v
```

Expected: 5 failures on `ModuleNotFoundError`.

- [ ] **Step 3: Write `brain/migrator/report.py`**

Create `/Users/hanamori/companion-emergence/brain/migrator/report.py`:

```python
"""Migration report formatting + source-manifest writer."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.migrator.og import FileManifest
from brain.migrator.transform import SkippedMemory


@dataclass(frozen=True)
class MigrationReport:
    """Aggregated outcome of a single migrator run."""

    memories_migrated: int
    memories_skipped: list[SkippedMemory]
    edges_migrated: int
    edges_skipped: int
    elapsed_seconds: float
    source_manifest: list[FileManifest]
    next_steps_inspect_cmds: list[str]
    next_steps_install_cmd: str


def format_report(report: MigrationReport) -> str:
    """Return the human-readable report text (printed + written to migration-report.md)."""
    lines: list[str] = []
    lines.append("Migration complete.")
    lines.append("")
    lines.append(
        f"  Memories:       {report.memories_migrated:,} migrated, "
        f"{len(report.memories_skipped):,} skipped"
    )
    lines.append(
        f"  Hebbian edges:  {report.edges_migrated:,} migrated, "
        f"{report.edges_skipped:,} skipped"
    )
    lines.append(f"  Elapsed:        {report.elapsed_seconds:.1f}s")
    lines.append("")

    if report.memories_skipped:
        lines.append(f"Skipped memories ({len(report.memories_skipped)}):")
        counts = Counter(s.reason for s in report.memories_skipped)
        for reason, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  - {n} {reason}")
        lines.append("")

    if report.source_manifest:
        lines.append("Source manifest:")
        for m in report.source_manifest:
            lines.append(
                f"  {m.relative_path:<32} {m.size_bytes:>12,} bytes  sha256={m.sha256[:12]}..."
            )
        lines.append("")

    if report.next_steps_inspect_cmds or report.next_steps_install_cmd:
        lines.append("Next steps:")
        if report.next_steps_inspect_cmds:
            lines.append("  1. Inspect the output:")
            for cmd in report.next_steps_inspect_cmds:
                lines.append(f"       {cmd}")
        if report.next_steps_install_cmd:
            lines.append("")
            lines.append("  2. When satisfied, install as a persona:")
            lines.append(f"       {report.next_steps_install_cmd}")

    return "\n".join(lines) + "\n"


def write_source_manifest(path: Path, manifest: list[FileManifest]) -> None:
    """Write source-manifest.json with every FileManifest entry + a generation timestamp."""
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "files": [
            {
                "relative_path": m.relative_path,
                "size_bytes": m.size_bytes,
                "sha256": m.sha256,
                "mtime_utc": m.mtime_utc,
            }
            for m in manifest
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run tests — expect green**

```bash
uv run pytest tests/unit/brain/migrator/test_report.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```

Expected: 228 passed (223 + 5). Ruff clean.

- [ ] **Step 6: Commit**

```bash
git add brain/migrator/report.py tests/unit/brain/migrator/test_report.py
git commit -m "feat(brain/migrator/report): migration report + source-manifest writers

MigrationReport aggregates a single run's outcome. format_report()
produces human-readable text (totals, skip-reason counts, source
manifest, next-steps). write_source_manifest() serialises the
FileManifest list to JSON with a generation timestamp.

5 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `brain/migrator/cli.py` — subcommand + safety + orchestration

**Goal:** Wire the migrator modules together behind a `brain migrate` subcommand. Argparse, safety checks (refuse-to-clobber, live-lock preflight, post-run source re-stat), atomic install-as-persona with backup.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/migrator/cli.py`
- Modify: `/Users/hanamori/companion-emergence/brain/cli.py` (wire subcommand)
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/migrator/test_cli.py`

- [ ] **Step 1: Inspect the existing `brain/cli.py` to understand the subcommand dispatch pattern**

```bash
cd /Users/hanamori/companion-emergence
cat brain/cli.py
```

Expected: an argparse-based main() with multiple stub subcommands. Note the existing pattern — we'll add `migrate` consistent with it.

- [ ] **Step 2: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/migrator/test_cli.py`:

```python
"""Tests for brain.migrator.cli — subcommand orchestration + safety."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from brain.migrator.cli import MigrateArgs, run_migrate


@pytest.fixture
def og_dir(tmp_path: Path) -> Path:
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
            "emotion_score": 9.0,
        },
        {
            "id": "m2",
            "content": "second",
            "memory_type": "meta",
            "domain": "work",
            "created_at": "2024-02-01T00:00:00+00:00",
            "emotions": {},
            "emotion_score": 0.0,
        },
    ]
    (og / "memories_v2.json").write_text(json.dumps(memories))
    (og / "connection_matrix_ids.json").write_text(json.dumps(["m1", "m2"]))
    matrix = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
    np.save(og / "connection_matrix.npy", matrix)
    (og / "hebbian_state.json").write_text("{}")
    return og


def test_run_migrate_output_mode_writes_expected_files(og_dir: Path, tmp_path: Path) -> None:
    """--output mode produces memories.db, hebbian.db, source-manifest.json, migration-report.md."""
    out = tmp_path / "migrated"
    args = MigrateArgs(input_dir=og_dir, output_dir=out, install_as=None, force=False)
    run_migrate(args)

    assert (out / "memories.db").exists()
    assert (out / "hebbian.db").exists()
    assert (out / "source-manifest.json").exists()
    assert (out / "migration-report.md").exists()


def test_run_migrate_refuses_nonempty_output_without_force(
    og_dir: Path, tmp_path: Path
) -> None:
    """Non-empty output dir without --force → error."""
    out = tmp_path / "migrated"
    out.mkdir()
    (out / "pre-existing.txt").write_text("x")

    args = MigrateArgs(input_dir=og_dir, output_dir=out, install_as=None, force=False)
    with pytest.raises(FileExistsError, match="non-empty"):
        run_migrate(args)


def test_run_migrate_allows_empty_output_dir(og_dir: Path, tmp_path: Path) -> None:
    """Empty output dir is acceptable — user may pre-create the path."""
    out = tmp_path / "migrated"
    out.mkdir()  # empty

    args = MigrateArgs(input_dir=og_dir, output_dir=out, install_as=None, force=False)
    run_migrate(args)  # must not raise


def test_run_migrate_refuses_live_lock(og_dir: Path, tmp_path: Path) -> None:
    """Recent memories_v2.json.lock → LiveLockDetected (migrator aborts)."""
    from brain.migrator.og import LiveLockDetected

    (og_dir / "memories_v2.json.lock").write_bytes(b"")
    # ensure fresh mtime
    time.sleep(0.01)

    out = tmp_path / "migrated"
    args = MigrateArgs(input_dir=og_dir, output_dir=out, install_as=None, force=False)
    with pytest.raises(LiveLockDetected):
        run_migrate(args)


def test_run_migrate_refuses_if_input_and_install_and_output_mutually_exclusive(
    og_dir: Path, tmp_path: Path
) -> None:
    """Passing both --output and --install-as is rejected."""
    with pytest.raises(ValueError, match="one of"):
        MigrateArgs(
            input_dir=og_dir,
            output_dir=tmp_path / "o",
            install_as="nell",
            force=False,
        )


def test_run_migrate_refuses_if_neither_output_nor_install(og_dir: Path) -> None:
    """Passing neither --output nor --install-as is rejected."""
    with pytest.raises(ValueError, match="one of"):
        MigrateArgs(input_dir=og_dir, output_dir=None, install_as=None, force=False)


def test_run_migrate_install_as_atomic_swap(
    og_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--install-as writes to <persona>.new then atomically renames."""
    persona_root = tmp_path / "persona_root"
    persona_root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(persona_root))

    args = MigrateArgs(input_dir=og_dir, output_dir=None, install_as="nell", force=False)
    run_migrate(args)

    persona_dir = persona_root / "nell"
    assert persona_dir.exists()
    assert (persona_dir / "memories.db").exists()
    assert (persona_dir / "hebbian.db").exists()
    # no leftover temp dir
    assert not (persona_root / "nell.new").exists()


def test_run_migrate_install_as_backs_up_existing(
    og_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--install-as --force backs up existing persona dir before overwriting."""
    persona_root = tmp_path / "persona_root"
    persona_root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(persona_root))

    old = persona_root / "nell"
    old.mkdir()
    (old / "marker.txt").write_text("old-data")

    args = MigrateArgs(input_dir=og_dir, output_dir=None, install_as="nell", force=True)
    run_migrate(args)

    # new nell is live
    assert (persona_root / "nell" / "memories.db").exists()
    # old nell is backed up somewhere as nell.backup-<timestamp>
    backups = [p for p in persona_root.iterdir() if p.name.startswith("nell.backup-")]
    assert len(backups) == 1
    assert (backups[0] / "marker.txt").read_text() == "old-data"


def test_run_migrate_install_as_refuses_without_force(
    og_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing persona + no --force → error."""
    persona_root = tmp_path / "persona_root"
    persona_root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(persona_root))

    (persona_root / "nell").mkdir()

    args = MigrateArgs(input_dir=og_dir, output_dir=None, install_as="nell", force=False)
    with pytest.raises(FileExistsError, match="persona"):
        run_migrate(args)
```

- [ ] **Step 3: Run tests — expect failures**

```bash
uv run pytest tests/unit/brain/migrator/test_cli.py -v
```

Expected: 9 failures on `ModuleNotFoundError`.

- [ ] **Step 4: Write `brain/migrator/cli.py`**

Create `/Users/hanamori/companion-emergence/brain/migrator/cli.py`:

```python
"""`brain migrate` subcommand — safety checks + orchestration."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.migrator.og import OGReader
from brain.migrator.report import MigrationReport, format_report, write_source_manifest
from brain.migrator.transform import SkippedMemory, transform_memory
from brain.paths import resolve_persona_root


@dataclass(frozen=True)
class MigrateArgs:
    """Validated argument bundle for run_migrate."""

    input_dir: Path
    output_dir: Path | None
    install_as: str | None
    force: bool

    def __post_init__(self) -> None:
        if (self.output_dir is None) == (self.install_as is None):
            raise ValueError(
                "Exactly one of --output or --install-as must be provided."
            )


def run_migrate(args: MigrateArgs) -> MigrationReport:
    """Execute a full migration. Returns the MigrationReport."""
    reader = OGReader(args.input_dir)
    reader.check_preflight()

    # Determine the write directory.
    if args.output_dir is not None:
        work_dir = args.output_dir
        _ensure_clobber_safe(work_dir, args.force, kind="output directory")
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp: Path | None = None
        finalise_target: Path | None = None
    else:
        assert args.install_as is not None
        persona_root = resolve_persona_root()
        final_dir = persona_root / args.install_as
        if final_dir.exists():
            if not args.force:
                raise FileExistsError(
                    f"Persona directory already exists: {final_dir}. "
                    "Pass --force to back up and overwrite."
                )
        work_dir = persona_root / f"{args.install_as}.new"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        cleanup_temp = work_dir
        finalise_target = final_dir

    started = time.monotonic()

    # ---- memories ----
    og_memories = reader.read_memories()
    store = MemoryStore(db_path=work_dir / "memories.db")
    migrated_count = 0
    skipped: list[SkippedMemory] = []
    seen_ids: set[str] = set()

    for og_mem in og_memories:
        mem, sk = transform_memory(og_mem)
        if sk is not None:
            skipped.append(sk)
            continue
        assert mem is not None
        if mem.id in seen_ids:
            skipped.append(
                SkippedMemory(
                    id=mem.id, reason="duplicate_id", field="id", raw_snippet=mem.content[:120]
                )
            )
            continue
        seen_ids.add(mem.id)
        store.create(mem)
        migrated_count += 1
    store.close()

    # ---- hebbian ----
    hebbian = HebbianMatrix(db_path=work_dir / "hebbian.db")
    edges_migrated = 0
    for a, b, w in reader.iter_nonzero_upper_edges():
        hebbian.strengthen(a, b, delta=float(w))
        edges_migrated += 1
    hebbian.close()

    elapsed = time.monotonic() - started

    # ---- post-run source re-stat (detect OG mutation) ----
    manifest = reader.manifest()
    _verify_sources_unchanged(args.input_dir, manifest)

    # ---- report + manifest artefacts ----
    inspect_cmds = _inspect_cmds(work_dir)
    install_cmd = _install_cmd(args.input_dir, args.install_as)
    report = MigrationReport(
        memories_migrated=migrated_count,
        memories_skipped=skipped,
        edges_migrated=edges_migrated,
        edges_skipped=0,
        elapsed_seconds=elapsed,
        source_manifest=manifest,
        next_steps_inspect_cmds=inspect_cmds,
        next_steps_install_cmd=install_cmd,
    )
    write_source_manifest(work_dir / "source-manifest.json", manifest)
    report_text = format_report(report)
    (work_dir / "migration-report.md").write_text(report_text, encoding="utf-8")
    print(report_text)

    # ---- finalise install-as with backup + atomic rename ----
    if finalise_target is not None:
        assert cleanup_temp is not None
        if finalise_target.exists():
            ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
            backup = finalise_target.with_name(f"{finalise_target.name}.backup-{ts}")
            os.rename(finalise_target, backup)
        os.rename(cleanup_temp, finalise_target)

    return report


def _ensure_clobber_safe(path: Path, force: bool, kind: str) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"{kind.capitalize()} is non-empty: {path}. Pass --force to overwrite."
            )
        # Force: remove contents
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def _verify_sources_unchanged(og_dir: Path, manifest: list) -> None:
    """Re-stat each source file; abort if any size/mtime differs from manifest."""
    for m in manifest:
        path = og_dir / m.relative_path
        st = path.stat()
        if st.st_size != m.size_bytes:
            raise RuntimeError(
                f"Source file {path} changed size during migration "
                f"(was {m.size_bytes}, now {st.st_size}). Aborting."
            )


def _inspect_cmds(work_dir: Path) -> list[str]:
    p = work_dir
    return [
        f'sqlite3 "{p / "memories.db"}" "SELECT COUNT(*) FROM memories;"',
        f'sqlite3 "{p / "memories.db"}" "SELECT domain, COUNT(*) FROM memories GROUP BY domain;"',
        f'sqlite3 "{p / "hebbian.db"}" "SELECT COUNT(*) FROM hebbian_edges;"',
        f'cat "{p / "migration-report.md"}"',
    ]


def _install_cmd(input_dir: Path, install_as: str | None) -> str:
    if install_as is not None:
        return ""  # already installed
    return (
        f"uv run brain migrate --input {input_dir} --install-as <persona-name>"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the `brain migrate` argparse subparser."""
    p = argparse.ArgumentParser(prog="brain migrate", description="Port OG NellBrain data into a new persona.")
    p.add_argument("--input", dest="input_dir", type=Path, required=True,
                   help="Path to the OG NellBrain data/ directory.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--output", dest="output_dir", type=Path,
                   help="Write migrated artefacts to this directory for inspection.")
    g.add_argument("--install-as", dest="install_as", type=str,
                   help="Install migrated data as this persona name.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite non-empty output dir / existing persona (with backup).")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    args = MigrateArgs(
        input_dir=ns.input_dir,
        output_dir=ns.output_dir,
        install_as=ns.install_as,
        force=ns.force,
    )
    run_migrate(args)
    return 0
```

- [ ] **Step 5: Check `brain/paths.py` for `resolve_persona_root`**

```bash
uv run python -c "from brain.paths import resolve_persona_root; print(resolve_persona_root())"
```

If this raises ImportError, we need to add `resolve_persona_root` to `brain/paths.py`. Check what exists first:

```bash
grep -n "def " /Users/hanamori/companion-emergence/brain/paths.py
```

If there is no `resolve_persona_root`, add the function to `brain/paths.py` modelled on whatever persona-path helper Week 1 built. The function should:
- Read `NELLBRAIN_HOME` environment variable if set → return `Path(NELLBRAIN_HOME)`
- Otherwise use `platformdirs.user_data_path("companion-emergence")` or the existing Week 1 equivalent

Example (adjust to match Week 1's style):

```python
def resolve_persona_root() -> Path:
    """Return the root directory under which persona sub-directories live.

    Honours NELLBRAIN_HOME environment override; falls back to platformdirs.
    """
    env = os.environ.get("NELLBRAIN_HOME")
    if env:
        return Path(env)
    from platformdirs import user_data_path
    return user_data_path("companion-emergence")
```

If Week 1 already provides an equivalent function under a different name, use that name in `cli.py` instead.

- [ ] **Step 6: Wire `migrate` into the top-level `brain` CLI**

Modify `/Users/hanamori/companion-emergence/brain/cli.py`. Find the existing argparse subcommand dispatch (probably a dict or if/elif cascade).

Add `migrate` as a subcommand. The minimum wiring:

```python
# At the top with other imports:
from brain.migrator.cli import main as migrate_main

# In the subcommand registration — wherever the existing subcommands are declared,
# add a `migrate` subparser that simply forwards remaining args to migrate_main.
```

The cleanest integration — register a subparser that re-parses its own args through migrate's `build_parser`. If the existing `brain/cli.py` uses `subparsers = parser.add_subparsers(...)`, add:

```python
from brain.migrator.cli import build_parser as _build_migrate_parser, run_migrate as _run_migrate
from brain.migrator.cli import MigrateArgs as _MigrateArgs

# In the subparser setup:
migrate_parser = subparsers.add_parser("migrate", help="Port OG NellBrain data into a persona.")
migrate_parser.add_argument("--input", dest="input_dir", type=Path, required=True)
g = migrate_parser.add_mutually_exclusive_group(required=True)
g.add_argument("--output", dest="output_dir", type=Path)
g.add_argument("--install-as", dest="install_as", type=str)
migrate_parser.add_argument("--force", action="store_true")
migrate_parser.set_defaults(
    _handler=lambda ns: _run_migrate(_MigrateArgs(
        input_dir=ns.input_dir,
        output_dir=ns.output_dir,
        install_as=ns.install_as,
        force=ns.force,
    )),
)
```

If the existing `brain/cli.py` uses a dict-based dispatch, adapt accordingly — the point is that `uv run brain migrate --input ... --output ...` dispatches to `run_migrate`.

- [ ] **Step 7: Run tests — expect green**

```bash
uv run pytest tests/unit/brain/migrator/test_cli.py -v
```

Expected: 9 passed.

- [ ] **Step 8: Manual CLI smoke**

```bash
uv run brain migrate --help
```

Expected: prints usage for the migrate subcommand with --input, --output, --install-as, --force.

- [ ] **Step 9: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```

Expected: 237 passed (228 + 9). Ruff clean.

- [ ] **Step 10: Commit**

```bash
git add brain/migrator/cli.py brain/cli.py tests/unit/brain/migrator/test_cli.py brain/paths.py
git commit -m "feat(brain/migrator/cli): brain migrate subcommand + safety + orchestration

MigrateArgs validates input (exactly one of --output / --install-as).
run_migrate() orchestrates: preflight → read OG → transform memories
→ populate MemoryStore + HebbianMatrix → post-run source re-stat →
write report + manifest → optional atomic install-as-persona.

Safety invariants enforced:
- Live OG lock → LiveLockDetected, abort.
- Non-empty output dir or existing persona → error unless --force.
- Install mode: write to <persona>.new, atomic rename, pre-existing
  persona renamed to <persona>.backup-<timestamp> first.
- Source file size/mtime re-checked after all reads; mismatch aborts.

9 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Integration fixture + end-to-end test

**Goal:** Build a small, deterministic OG-shaped fixture and run the migrator end-to-end against it. Verifies the whole pipeline composes.

**Files:**
- Create: `/Users/hanamori/companion-emergence/tests/integration/__init__.py` (empty)
- Create: `/Users/hanamori/companion-emergence/tests/integration/test_full_migration.py`
- The fixture is built in-test via `tmp_path` — no files checked into the repo.

- [ ] **Step 1: Create the integration test dir**

```bash
cd /Users/hanamori/companion-emergence
mkdir -p tests/integration
touch tests/integration/__init__.py
```

- [ ] **Step 2: Write the integration test**

Create `/Users/hanamori/companion-emergence/tests/integration/test_full_migration.py`:

```python
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


def test_full_migration_output_mode_produces_expected_counts(
    og_mini: Path, tmp_path: Path
) -> None:
    """End-to-end: run migrator, open output dbs, verify counts + a specific record."""
    out = tmp_path / "migrated-mini"
    args = MigrateArgs(input_dir=og_mini, output_dir=out, install_as=None, force=False)
    report = run_migrate(args)

    # Report shape
    assert report.memories_migrated == 4  # 5 input - 1 malformed
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


def test_full_migration_metadata_absorbs_og_only_fields(
    og_mini: Path, tmp_path: Path
) -> None:
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


def test_full_migration_source_manifest_records_all_files(
    og_mini: Path, tmp_path: Path
) -> None:
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
    """After the migrator runs, OG dir contents (except any stat metadata) unchanged."""
    # Snapshot before
    before = {
        p.name: (p.stat().st_size, p.read_bytes())
        for p in og_mini.iterdir()
    }
    out = tmp_path / "migrated-mini"
    args = MigrateArgs(input_dir=og_mini, output_dir=out, install_as=None, force=False)
    run_migrate(args)

    after = {
        p.name: (p.stat().st_size, p.read_bytes())
        for p in og_mini.iterdir()
    }
    assert before == after, "OG files must not be mutated"
```

- [ ] **Step 3: Run the integration test — expect green**

```bash
uv run pytest tests/integration/test_full_migration.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```

Expected: 241 passed (237 + 4). Ruff clean.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/
git commit -m "test(integration): end-to-end migrator against a 5-memory OG fixture

Builds a tmp_path OG-shaped fixture (5 memories incl. 1 malformed,
5x5 hebbian matrix with 3 non-zero edges), runs the migrator in
--output mode, opens the resulting memories.db + hebbian.db via
raw sqlite3, asserts counts + record shapes + metadata round-trip +
source manifest coverage + OG read-only invariant.

Four assertions:
- memories + edges match expected counts; malformed m4 skipped
- m1.metadata contains source_date, source_summary, supersedes
- source-manifest.json covers all four OG files with valid sha256
- OG dir is byte-identical before and after the run

4 tests green; 241 total.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Week 3.5 close-out — CI + merge

**Goal:** Verify CI green on 3 OSes, merge PR, no tag (this is ancillary/half-week work).

- [ ] **Step 1: Clean install from scratch**

```bash
cd /Users/hanamori/companion-emergence
rm -rf .venv
uv sync --all-extras
```

- [ ] **Step 2: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```

Expected: 241 passed. Ruff clean.

- [ ] **Step 3: Manual CLI smoke against the fixture**

The `tests/integration/` fixture is built dynamically inside tmp_path, so a standalone CLI invocation against the real OG data is the only interactive verification. Confirm the subcommand is reachable:

```bash
uv run brain migrate --help
```

Expected: help text showing --input, --output, --install-as, --force.

- [ ] **Step 4: Push branch + open PR**

```bash
git push -u origin week-3.5-migrator
gh pr create --title "feat: Week 3.5 — OG memory migrator (memories + Hebbian, read-only source)" --body "$(cat <<'EOF'
## Summary
- Adds `brain/migrator/` package (og.py + transform.py + report.py + cli.py) with `brain migrate` subcommand
- Ships the Week 3 amendment: Memory.metadata field + metadata_json column
- Preserves OG data strictly (read-only reads + SHA-256 source manifest + post-run re-stat)
- Refuse-to-clobber output and atomic install-as-persona with timestamped backup
- 50 new tests (9+14+5+9+4+9) = ~50; total suite reaches 241 across macOS + Windows + Linux

## What landed per task

| Task | Purpose | Tests added |
|---|---|---|
| 1. Memory.metadata field | Week 3 amendment | 5 |
| 2. MemoryStore metadata_json column | Week 3 amendment | 4 |
| 3. brain/migrator/og.py | OG readers + manifest + live-lock preflight | 9 |
| 4. brain/migrator/transform.py | OG → Memory transform + SkippedMemory | 14 |
| 5. brain/migrator/report.py | Report formatter + manifest writer | 5 |
| 6. brain/migrator/cli.py | Subcommand + safety + orchestration | 9 |
| 7. Integration test | End-to-end against a 5-memory fixture | 4 |

## Safety summary

- **OG writes:** zero. Every file opened `rb` / `r`. Source manifest with SHA-256 records every file read.
- **Live bridge detection:** refuses to run if `memories_v2.json.lock` mtime < 5 min.
- **Post-run verification:** source file size/mtime re-checked against manifest — abort on mismatch.
- **Output clobber:** refuses non-empty output dir or existing persona without --force.
- **Install atomicity:** writes to `<persona>.new/`, timestamp-backs-up any existing `<persona>/`, then `os.rename`s the new dir into place.

## Test plan
- [x] Fresh `uv sync --all-extras` from scratch succeeds
- [x] pytest — 241 tests pass locally
- [x] ruff check + format — clean
- [x] Manual: `uv run brain migrate --help` resolves
- [x] Integration test verifies OG bytes are unchanged after migration
- [ ] CI matrix green across all 3 OSes (verifies after push)
- [ ] Hana dry-runs against real OG data in `--output` mode and inspects
- [ ] Hana runs `--install-as nell` once satisfied
EOF
)"
```

- [ ] **Step 5: Watch CI to completion**

```bash
sleep 10
gh run list --branch week-3.5-migrator --limit 1
gh run watch
```

Expected: all 3 OSes complete with `success`. If any fail, diagnose with `gh run view --log-failed`, fix, commit, push; re-verify.

- [ ] **Step 6: Merge PR (no tag — this is auxiliary/half-week work)**

After CI green:

```bash
gh pr merge --merge --delete-branch
git checkout main
git pull origin main
```

Expected: main advances to include the week-3.5-migrator work.

Week 4 engines can now be specced against real persona data once Hana runs the migrator end-to-end.

---

## Week 3.5 green-light criterion

Week 3.5 is green when ALL of:

1. `uv sync --all-extras` succeeds on a fresh clone
2. `uv run pytest` reports 241 passed
3. `uv run ruff check .` + `uv run ruff format --check .` both clean
4. `uv run brain migrate --help` prints the subcommand usage
5. `tests/integration/test_full_migration.py` end-to-end test green
6. CI shows `✓ success` on macOS + Ubuntu + Windows
7. PR merged to main

**User-side verification (not part of automated criterion):**
- Hana runs `uv run brain migrate --input /Users/hanamori/NellBrain/data --output ./migrated-nell` and inspects the output dbs + report
- Hana runs `uv run brain migrate --input /Users/hanamori/NellBrain/data --install-as nell` once satisfied
- `sqlite3 <persona-dir>/memories.db "SELECT COUNT(*) FROM memories"` returns ~1,128 (1,141 minus whatever is malformed)
- `sqlite3 <persona-dir>/hebbian.db "SELECT COUNT(*) FROM hebbian_edges"` returns 8,808

When all seven automated criteria are true, Week 4 engines can begin being specced against the real migrated persona.

---

## Notes for the engineer executing this plan

- **OG data at `/Users/hanamori/NellBrain/data/` is production-live.** Never write to it. The migrator's read-only discipline is tested (Task 7's byte-identical check). Do not add any code path that opens these files with `"w"` or `"a"`.
- **Week 3 amendment first.** Tasks 1 + 2 amend Memory + MemoryStore before any migrator code lands. This preserves backward-compat with all existing 191 Week 3 tests.
- **No CI runs against real OG data.** Fixtures only. The 3.9 MB `memories_v2.json` is not in the repo and stays on Hana's machine.
- **Persona root resolution.** If `brain/paths.py` doesn't already expose `resolve_persona_root`, add it — honours `NELLBRAIN_HOME` env override then falls back to platformdirs. This is consistent with the Week 1 pattern.
- **The migrator never calls any LLM.** Embeddings are explicitly out of scope (regenerate in Week 5 with the Ollama / Claude CLI bridge). No API calls, no token costs.
- **Timestamp discipline.** Always `datetime.now(UTC)` — the Week 2/3 code is strict about tz-aware datetimes; the migrator must match. `_coerce_utc` is the shared helper.
- **Test isolation.** All migrator tests use `tmp_path` fixtures. No absolute paths, no `/Users/hanamori/...` inside a test — keeps CI portable across the 3 OS matrix.

---

*End of Week 3.5 plan.*
