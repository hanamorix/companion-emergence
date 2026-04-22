# Week 3 — Memory Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `brain/memory/` — the SQLite-backed memory substrate per spec Section 4.1. At end of Week 3: a persona can create, query, strengthen, and search memories through a clean typed API, with content-hash cached embeddings and Hebbian-weighted spreading activation.

**Architecture:** Four sub-modules (store, embeddings, hebbian, search) layered on SQLite. Memory is a frozen-at-creation dataclass serialised to/from a single `memories` table. Embeddings are computed via a provider abstraction (FakeEmbeddingProvider for tests, OllamaEmbeddingProvider for production) and cached by content-hash in `embedding_cache`. Hebbian connections live in `hebbian_edges` with weight + last-strengthened tracking. Search layers semantic (embedding cosine), emotional (emotion-dict overlap), temporal, and spreading (Hebbian BFS) queries, composable via `combined_search`.

**Tech Stack:** Python 3.12 stdlib (sqlite3, hashlib, uuid, dataclasses), numpy for embedding vectors, pytest. No new dependencies except numpy (already common stdlib-adjacent; add to pyproject).

---

## Context: what already exists (Week 2 state)

Main branch HEAD: `51aa6f6` (merge commit for Week 2).

- `brain/` package: `__init__.py`, `paths.py` (platformdirs + NELLBRAIN_HOME override), `config.py` (3-source precedence), `cli.py` (10 stub subcommands)
- `brain/emotion/` package: vocabulary, state, decay, arousal, blend, influence, expression — 7 modules, 75 tests
- `examples/starter-thoughtful/` — full starter persona
- CI matrix green on macOS + Windows + Linux; 113 tests passing
- Tags: `week-1-complete`, `week-2-complete`

Feature branch for this work: `week-3-memory-substrate` (already created off `main`).

---

## File structure (what this plan creates)

```
companion-emergence/
├── brain/
│   └── memory/
│       ├── __init__.py                      (Task 1 — exports)
│       ├── store.py                         (Task 1 + Task 2 — Memory dataclass + MemoryStore)
│       ├── embeddings.py                    (Task 3 — EmbeddingProvider + EmbeddingCache)
│       ├── hebbian.py                       (Task 4 — HebbianMatrix)
│       └── search.py                        (Task 5 — MemorySearch)
└── tests/
    └── unit/
        └── brain/
            └── memory/
                ├── __init__.py              (Task 1 — empty)
                ├── test_store.py            (Task 1 + Task 2)
                ├── test_embeddings.py       (Task 3)
                ├── test_hebbian.py          (Task 4)
                └── test_search.py           (Task 5)
```

Modified:
- `pyproject.toml` — add `numpy>=1.26` to dependencies (Task 1)

Nothing else changes. `brain/__init__.py` does **not** re-export memory types — consumers import from `brain.memory.*` directly.

---

## Dependency order

- Task 1 (Memory dataclass) is a prerequisite for Tasks 2, 4, 5.
- Task 2 (MemoryStore) depends on Task 1.
- Task 3 (EmbeddingProvider + Cache) is independent of 1, 2.
- Task 4 (HebbianMatrix) depends on Task 1 (references Memory IDs).
- Task 5 (MemorySearch) depends on Tasks 1, 2, 3, 4.
- Task 6 (close-out) depends on all.

Execute in numerical order. 1 → 2 → 3 → 4 → 5 → 6.

---

## Task 1: Memory dataclass + `brain/memory/` package init

**Goal:** Define the `Memory` dataclass that represents a single memory. Includes UUID generation, score computation from emotions, to_dict/from_dict round-trip, and timezone-aware timestamps. Ships the empty package skeleton.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/memory/__init__.py`
- Create: `/Users/hanamori/companion-emergence/brain/memory/store.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/memory/__init__.py` (empty)
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_store.py` (Memory-dataclass portion; MemoryStore follows in Task 2)
- Modify: `/Users/hanamori/companion-emergence/pyproject.toml` (add numpy dep)

- [ ] **Step 1: Create empty package files**

```bash
cd /Users/hanamori/companion-emergence
mkdir -p brain/memory tests/unit/brain/memory
touch tests/unit/brain/memory/__init__.py
```

- [ ] **Step 2: Add numpy to pyproject.toml**

Edit `/Users/hanamori/companion-emergence/pyproject.toml`. Find the `dependencies = [...]` block and change it to:

```toml
dependencies = [
    "platformdirs>=4.2",
    "numpy>=1.26",
]
```

Run `uv sync --all-extras` to install.

- [ ] **Step 3: Write `brain/memory/__init__.py`**

```python
"""The memory substrate — SQLite-backed store + embeddings + Hebbian + search.

Four sub-modules, each with a single responsibility:
- store: Memory dataclass + MemoryStore (SQLite-backed CRUD)
- embeddings: provider abstraction + content-hash cache
- hebbian: connection matrix + spreading activation
- search: semantic + emotional + temporal + spreading queries

See spec Section 4.1 for the file-tree and Section 10.1 for the SQLite
data-layer decision (replaces OG's JSON/numpy files).
"""

from brain.memory.store import Memory, MemoryStore

__all__ = ["Memory", "MemoryStore"]
```

- [ ] **Step 4: Write the failing Memory tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_store.py`:

```python
"""Tests for brain.memory.store — Memory dataclass + MemoryStore."""

from __future__ import annotations

from datetime import UTC, datetime

from brain.memory.store import Memory


def test_memory_create_new_generates_uuid() -> None:
    """Memory.create_new generates a UUID id."""
    m = Memory.create_new(
        content="first meeting",
        memory_type="conversation",
        domain="us",
    )
    assert isinstance(m.id, str)
    assert len(m.id) == 36  # canonical UUID string form
    assert m.id.count("-") == 4


def test_memory_create_new_sets_created_at_utc() -> None:
    """create_new sets created_at to a tz-aware UTC datetime."""
    before = datetime.now(UTC)
    m = Memory.create_new(content="x", memory_type="meta", domain="work")
    after = datetime.now(UTC)
    assert m.created_at.tzinfo is not None
    assert before <= m.created_at <= after


def test_memory_create_new_computes_score_from_emotions() -> None:
    """score = sum of emotion intensities at create time."""
    m = Memory.create_new(
        content="held",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0, "tenderness": 6.0},
    )
    assert m.score == 15.0


def test_memory_create_new_score_zero_when_no_emotions() -> None:
    """Empty emotions dict → score 0."""
    m = Memory.create_new(content="note", memory_type="meta", domain="work")
    assert m.score == 0.0


def test_memory_create_new_importance_defaults_to_score_over_ten() -> None:
    """If importance unspecified, default = score / 10.0 (normalised)."""
    m = Memory.create_new(
        content="held",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0, "tenderness": 6.0},
    )
    assert m.importance == 1.5  # 15.0 / 10.0


def test_memory_create_new_importance_manual_override() -> None:
    """Explicit importance overrides the score-based default."""
    m = Memory.create_new(
        content="held",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0},
        importance=7.0,
    )
    assert m.importance == 7.0


def test_memory_defaults_active_and_unprotected() -> None:
    """New memories are active and unprotected by default."""
    m = Memory.create_new(content="x", memory_type="meta", domain="work")
    assert m.active is True
    assert m.protected is False


def test_memory_to_dict_round_trips() -> None:
    """to_dict / from_dict round-trips cleanly."""
    original = Memory.create_new(
        content="the moment",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0, "anchor_pull": 8.0},
        tags=["first", "important"],
    )
    data = original.to_dict()
    restored = Memory.from_dict(data)

    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.memory_type == original.memory_type
    assert restored.domain == original.domain
    assert restored.emotions == original.emotions
    assert restored.tags == original.tags
    assert restored.score == original.score
    assert restored.importance == original.importance
    assert restored.created_at == original.created_at
    assert restored.active == original.active


def test_memory_from_dict_coerces_naive_timestamps_to_utc() -> None:
    """Naive timestamps in JSON restore as UTC-aware."""
    data = {
        "id": "00000000-0000-0000-0000-000000000001",
        "content": "legacy",
        "memory_type": "meta",
        "domain": "work",
        "emotions": {},
        "tags": [],
        "importance": 0.0,
        "score": 0.0,
        "created_at": "2024-01-01T12:00:00",  # no tz
        "last_accessed_at": None,
        "active": True,
        "protected": False,
    }
    m = Memory.from_dict(data)
    assert m.created_at.tzinfo is not None


def test_memory_dataclass_preserves_explicit_id_for_migration() -> None:
    """Memory() direct construction accepts an explicit id (for migrator use)."""
    m = Memory(
        id="abc-123",
        content="migrated",
        memory_type="conversation",
        domain="us",
        created_at=datetime.now(UTC),
    )
    assert m.id == "abc-123"
```

- [ ] **Step 5: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_store.py -v
```

Expected: 10 failures with `ModuleNotFoundError: No module named 'brain.memory.store'` (or similar import error).

- [ ] **Step 6: Write `brain/memory/store.py` (Memory dataclass portion)**

```python
"""Memory dataclass + SQLite-backed MemoryStore.

Design per spec Section 4.1 (brain/memory/store.py) and Section 10.1
(SQLite data layer replaces OG JSON/numpy files).

Memory is the canonical record type. MemoryStore is the CRUD surface over
a single SQLite database containing the `memories` table. Tasks 3-5 add
sibling modules (embeddings, hebbian, search) that read from and strengthen
this store.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Memory:
    """A single memory — content, context, emotional weight, and metadata.

    Attributes:
        id: UUID string, canonical form (36 chars with hyphens).
        content: The memory's textual content.
        memory_type: "conversation", "meta", "dream", "consolidated",
            "heartbeat", "reflex", or any persona-defined category.
        domain: "us", "work", "craft", or any persona-defined scope.
        emotions: {emotion_name: intensity} at creation time.
        tags: free-form labels.
        importance: 0.0..10.0 (normalised). Auto-defaults to score/10 if
            not explicitly specified at create_new() time.
        score: sum of emotion intensities — computed once at creation.
        created_at: tz-aware UTC datetime of creation.
        last_accessed_at: tz-aware UTC datetime of most recent read, or None.
        active: F22 deactivation flag. Inactive memories are excluded from
            default queries but remain in the database (reversible).
        protected: excluded from decay/consolidation.
    """

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

    @classmethod
    def create_new(
        cls,
        content: str,
        memory_type: str,
        domain: str,
        emotions: dict[str, float] | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> Memory:
        """Factory: new memory with generated UUID, current UTC time,
        and auto-computed score + importance (if importance is None).

        Score = sum of emotion intensities.
        Importance defaults to score/10.0 (normalised to 0..10 scale).
        """
        emotions = dict(emotions or {})
        tags = list(tags or [])
        score = sum(emotions.values())
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            domain=domain,
            created_at=datetime.now(UTC),
            emotions=emotions,
            tags=tags,
            importance=importance if importance is not None else score / 10.0,
            score=score,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain-dict form suitable for JSON or SQLite storage."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "domain": self.domain,
            "emotions": dict(self.emotions),
            "tags": list(self.tags),
            "importance": self.importance,
            "score": self.score,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat()
            if self.last_accessed_at
            else None,
            "active": self.active,
            "protected": self.protected,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Restore from a dict produced by to_dict.

        Tz-naive timestamps are coerced to UTC (permissive for migrator input).
        """
        created = datetime.fromisoformat(data["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        last_accessed = None
        if data.get("last_accessed_at"):
            last_accessed = datetime.fromisoformat(data["last_accessed_at"])
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=UTC)
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=data["memory_type"],
            domain=data["domain"],
            created_at=created,
            emotions=dict(data.get("emotions", {})),
            tags=list(data.get("tags", [])),
            importance=float(data.get("importance", 0.0)),
            score=float(data.get("score", 0.0)),
            last_accessed_at=last_accessed,
            active=bool(data.get("active", True)),
            protected=bool(data.get("protected", False)),
        )
```

- [ ] **Step 7: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_store.py -v
```

Expected: 10 passed.

- [ ] **Step 8: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 123 passed (113 + 10). Ruff both clean.

- [ ] **Step 9: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/memory/ tests/unit/brain/memory/ pyproject.toml uv.lock
git commit -m "feat(brain/memory): Memory dataclass + package skeleton

Memory is the canonical record type — id, content, memory_type, domain,
emotions dict, tags, importance, score, created_at, last_accessed_at,
active flag, protected flag. create_new() factory generates UUID, sets
UTC timestamp, computes score from emotions, defaults importance to
score/10.0. to_dict/from_dict round-trip cleanly; tz-naive timestamps
coerced to UTC on restore (permissive for migrator).

Adds numpy>=1.26 dependency ahead of Tasks 3-5 (embeddings use np arrays).

10 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `MemoryStore` — SQLite-backed CRUD + queries

**Goal:** `MemoryStore` class managing a SQLite database of memories. Full CRUD (create, get, update, deactivate), list queries (by domain, type, emotion), count, substring search, plus schema bootstrap. Uses in-memory SQLite for all tests; no filesystem pollution.

**Files:**
- Modify: `/Users/hanamori/companion-emergence/brain/memory/store.py` (add MemoryStore class)
- Modify: `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_store.py` (append MemoryStore tests)

- [ ] **Step 1: Write the failing MemoryStore tests**

APPEND to `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_store.py`:

```python


import pytest

from brain.memory.store import MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    """In-memory MemoryStore, fresh per test."""
    return MemoryStore(db_path=":memory:")


def _mem(content: str = "x", **kw: object) -> Memory:
    defaults = {"memory_type": "conversation", "domain": "us"}
    defaults.update(kw)
    return Memory.create_new(content=content, **defaults)  # type: ignore[arg-type]


def test_store_init_creates_schema(store: MemoryStore) -> None:
    """Fresh store has a memories table."""
    cursor = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    )
    assert cursor.fetchone() is not None


def test_store_create_and_get_round_trips(store: MemoryStore) -> None:
    """create() then get() returns the same Memory."""
    original = _mem("first meeting", emotions={"love": 9.0})
    store.create(original)

    restored = store.get(original.id)
    assert restored is not None
    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.emotions == original.emotions
    assert restored.score == original.score


def test_store_get_unknown_returns_none(store: MemoryStore) -> None:
    """get() on a nonexistent id returns None."""
    assert store.get("nonexistent-id") is None


def test_store_create_returns_the_memory_id(store: MemoryStore) -> None:
    """create() returns the id it stored."""
    m = _mem("x")
    returned = store.create(m)
    assert returned == m.id


def test_store_create_duplicate_id_raises(store: MemoryStore) -> None:
    """Creating two memories with the same id raises."""
    m = _mem("x")
    store.create(m)
    with pytest.raises(Exception):  # sqlite3.IntegrityError subclass
        store.create(m)


def test_store_list_by_domain(store: MemoryStore) -> None:
    """list_by_domain filters correctly."""
    store.create(_mem("a", domain="us"))
    store.create(_mem("b", domain="us"))
    store.create(_mem("c", domain="work"))

    us = store.list_by_domain("us")
    work = store.list_by_domain("work")
    assert len(us) == 2
    assert len(work) == 1
    assert all(m.domain == "us" for m in us)


def test_store_list_by_type(store: MemoryStore) -> None:
    """list_by_type filters by memory_type."""
    store.create(_mem("a", memory_type="conversation"))
    store.create(_mem("b", memory_type="meta"))
    store.create(_mem("c", memory_type="conversation"))

    convs = store.list_by_type("conversation")
    assert len(convs) == 2
    assert all(m.memory_type == "conversation" for m in convs)


def test_store_list_by_emotion_filters_by_intensity(store: MemoryStore) -> None:
    """list_by_emotion returns memories where that emotion ≥ min_intensity."""
    store.create(_mem("a", emotions={"love": 9.0}))
    store.create(_mem("b", emotions={"love": 3.0}))
    store.create(_mem("c", emotions={"grief": 8.0}))

    strong_love = store.list_by_emotion("love", min_intensity=5.0)
    assert len(strong_love) == 1
    assert strong_love[0].content == "a"


def test_store_list_excludes_inactive_by_default(store: MemoryStore) -> None:
    """list_by_domain excludes deactivated memories by default."""
    m1 = _mem("active", domain="us")
    m2 = _mem("inactive", domain="us")
    store.create(m1)
    store.create(m2)
    store.deactivate(m2.id)

    active = store.list_by_domain("us")
    assert len(active) == 1
    assert active[0].content == "active"


def test_store_list_includes_inactive_when_requested(store: MemoryStore) -> None:
    """Passing active_only=False includes deactivated memories."""
    m1 = _mem("active", domain="us")
    m2 = _mem("inactive", domain="us")
    store.create(m1)
    store.create(m2)
    store.deactivate(m2.id)

    all_ = store.list_by_domain("us", active_only=False)
    assert len(all_) == 2


def test_store_list_respects_limit(store: MemoryStore) -> None:
    """limit caps the result count."""
    for i in range(5):
        store.create(_mem(f"m{i}", domain="us"))
    assert len(store.list_by_domain("us", limit=3)) == 3


def test_store_update_mutates_specified_fields(store: MemoryStore) -> None:
    """update() mutates only the given fields."""
    m = _mem("original")
    store.create(m)
    store.update(m.id, content="modified", importance=9.0)

    restored = store.get(m.id)
    assert restored is not None
    assert restored.content == "modified"
    assert restored.importance == 9.0
    assert restored.domain == m.domain  # unchanged


def test_store_update_unknown_raises(store: MemoryStore) -> None:
    """update() on a nonexistent id raises KeyError."""
    with pytest.raises(KeyError):
        store.update("nonexistent", content="x")


def test_store_deactivate_flips_active_flag(store: MemoryStore) -> None:
    """deactivate() sets active=False without deleting the row."""
    m = _mem("x")
    store.create(m)
    store.deactivate(m.id)

    restored = store.get(m.id)
    assert restored is not None
    assert restored.active is False


def test_store_deactivate_unknown_raises(store: MemoryStore) -> None:
    """deactivate() on a nonexistent id raises KeyError."""
    with pytest.raises(KeyError):
        store.deactivate("nonexistent")


def test_store_count_active_only_default(store: MemoryStore) -> None:
    """count() excludes inactive by default."""
    m1 = _mem("a")
    m2 = _mem("b")
    store.create(m1)
    store.create(m2)
    store.deactivate(m2.id)
    assert store.count() == 1


def test_store_count_including_inactive(store: MemoryStore) -> None:
    """count(active_only=False) includes inactive memories."""
    m1 = _mem("a")
    m2 = _mem("b")
    store.create(m1)
    store.create(m2)
    store.deactivate(m2.id)
    assert store.count(active_only=False) == 2


def test_store_search_text_returns_substring_matches(store: MemoryStore) -> None:
    """search_text finds memories whose content contains the query."""
    store.create(_mem("cold coffee, warm hana"))
    store.create(_mem("the evening has a shape to it now"))
    store.create(_mem("creative hunger strikes"))

    results = store.search_text("evening")
    assert len(results) == 1
    assert "evening" in results[0].content


def test_store_search_text_is_case_insensitive(store: MemoryStore) -> None:
    """Substring matching ignores case."""
    store.create(_mem("The Moment"))
    results = store.search_text("moment")
    assert len(results) == 1
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_store.py -v
```

Expected: 18 new failures on the MemoryStore-class tests; the 10 Memory-dataclass tests from Task 1 still pass.

- [ ] **Step 3: Append `MemoryStore` class to `brain/memory/store.py`**

Append to the end of `/Users/hanamori/companion-emergence/brain/memory/store.py`:

```python


import json
import sqlite3
from pathlib import Path


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

CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""


class MemoryStore:
    """SQLite-backed store for Memory records.

    Pass `":memory:"` as db_path for in-memory databases (used in tests).
    Any filesystem path creates or opens a persistent database.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection. Safe to call multiple times."""
        self._conn.close()

    def create(self, memory: Memory) -> str:
        """Insert a memory. Returns the id. Raises on duplicate id."""
        self._conn.execute(
            """
            INSERT INTO memories (
                id, content, memory_type, domain, emotions_json, tags_json,
                importance, score, created_at, last_accessed_at, active, protected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.content,
                memory.memory_type,
                memory.domain,
                json.dumps(memory.emotions),
                json.dumps(memory.tags),
                memory.importance,
                memory.score,
                memory.created_at.isoformat(),
                memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
                1 if memory.active else 0,
                1 if memory.protected else 0,
            ),
        )
        self._conn.commit()
        return memory.id

    def get(self, memory_id: str) -> Memory | None:
        """Return the Memory with the given id, or None."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return _row_to_memory(row) if row else None

    def list_by_domain(
        self, domain: str, active_only: bool = True, limit: int | None = None
    ) -> list[Memory]:
        """Return memories in the given domain, ordered by created_at desc."""
        return self._list_filter("domain", domain, active_only, limit)

    def list_by_type(
        self, memory_type: str, active_only: bool = True, limit: int | None = None
    ) -> list[Memory]:
        """Return memories of the given type, ordered by created_at desc."""
        return self._list_filter("memory_type", memory_type, active_only, limit)

    def list_by_emotion(
        self,
        emotion_name: str,
        min_intensity: float = 5.0,
        active_only: bool = True,
        limit: int | None = None,
    ) -> list[Memory]:
        """Return memories where `emotion_name` is present at ≥ min_intensity."""
        sql = "SELECT * FROM memories WHERE 1=1"
        if active_only:
            sql += " AND active = 1"
        sql += " ORDER BY created_at DESC"
        rows = self._conn.execute(sql).fetchall()

        results: list[Memory] = []
        for row in rows:
            emotions = json.loads(row["emotions_json"])
            if emotions.get(emotion_name, 0.0) >= min_intensity:
                results.append(_row_to_memory(row))
                if limit is not None and len(results) >= limit:
                    break
        return results

    def update(self, memory_id: str, **fields: Any) -> None:
        """Update the given fields on an existing memory.

        Accepts: content, memory_type, domain, emotions (dict), tags (list),
        importance, score, last_accessed_at, active, protected.

        Raises KeyError if memory_id does not exist.
        """
        if self.get(memory_id) is None:
            raise KeyError(f"Unknown memory id: {memory_id!r}")

        column_map: dict[str, tuple[str, Any]] = {}
        for key, value in fields.items():
            if key == "emotions":
                column_map["emotions_json"] = ("emotions_json", json.dumps(value))
            elif key == "tags":
                column_map["tags_json"] = ("tags_json", json.dumps(value))
            elif key == "last_accessed_at":
                column_map[key] = (
                    key,
                    value.isoformat() if value else None,
                )
            elif key in ("active", "protected"):
                column_map[key] = (key, 1 if value else 0)
            elif key in (
                "content",
                "memory_type",
                "domain",
                "importance",
                "score",
            ):
                column_map[key] = (key, value)
            else:
                raise ValueError(f"Unknown update field: {key!r}")

        if not column_map:
            return
        set_clause = ", ".join(f"{col} = ?" for col, _ in column_map.values())
        values = [v for _, v in column_map.values()]
        values.append(memory_id)
        self._conn.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
        self._conn.commit()

    def deactivate(self, memory_id: str) -> None:
        """Mark a memory inactive (F22 semantics). Raises KeyError if unknown."""
        if self.get(memory_id) is None:
            raise KeyError(f"Unknown memory id: {memory_id!r}")
        self._conn.execute("UPDATE memories SET active = 0 WHERE id = ?", (memory_id,))
        self._conn.commit()

    def count(self, active_only: bool = True) -> int:
        """Return the total count of memories."""
        sql = "SELECT COUNT(*) FROM memories"
        if active_only:
            sql += " WHERE active = 1"
        return int(self._conn.execute(sql).fetchone()[0])

    def search_text(
        self, query: str, active_only: bool = True, limit: int | None = None
    ) -> list[Memory]:
        """Case-insensitive substring search on content."""
        sql = "SELECT * FROM memories WHERE content LIKE ? COLLATE NOCASE"
        params: list[Any] = [f"%{query}%"]
        if active_only:
            sql += " AND active = 1"
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_memory(row) for row in rows]

    def _list_filter(
        self, column: str, value: str, active_only: bool, limit: int | None
    ) -> list[Memory]:
        sql = f"SELECT * FROM memories WHERE {column} = ?"
        params: list[Any] = [value]
        if active_only:
            sql += " AND active = 1"
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_memory(row) for row in rows]


def _row_to_memory(row: sqlite3.Row) -> Memory:
    """Materialise a sqlite row into a Memory dataclass."""
    created = datetime.fromisoformat(row["created_at"])
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    last_accessed = None
    if row["last_accessed_at"]:
        last_accessed = datetime.fromisoformat(row["last_accessed_at"])
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=UTC)
    return Memory(
        id=row["id"],
        content=row["content"],
        memory_type=row["memory_type"],
        domain=row["domain"],
        created_at=created,
        emotions=json.loads(row["emotions_json"]),
        tags=json.loads(row["tags_json"]),
        importance=float(row["importance"]),
        score=float(row["score"]),
        last_accessed_at=last_accessed,
        active=bool(row["active"]),
        protected=bool(row["protected"]),
    )
```

Add this import at the top of `store.py` (below the existing imports):

```python
from typing import Any
```

(If `Any` isn't already imported; if it is, skip this step.)

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_store.py -v
```

Expected: 28 passed (10 Memory + 18 MemoryStore).

- [ ] **Step 5: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 141 passed (123 + 18). Ruff both clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/memory/store.py tests/unit/brain/memory/test_store.py
git commit -m "feat(brain/memory/store): MemoryStore — SQLite-backed CRUD + queries

MemoryStore class around a sqlite3 connection; schema bootstrapped on
first connect (idempotent). Operations: create, get, list_by_domain,
list_by_type, list_by_emotion (with intensity threshold), update
(partial fields), deactivate (F22 semantics — flag, not delete),
count, search_text (case-insensitive substring).

emotions and tags stored as JSON text columns; timestamps as ISO 8601;
active/protected as INTEGER (0/1). Indexes on domain, memory_type,
active, created_at.

In-memory ':memory:' db_path supported for tests (no filesystem pollution).
18 new tests; 28 total on store.py; 141 total across the suite.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `brain/memory/embeddings.py` — provider abstraction + content-hash cache

**Goal:** `EmbeddingProvider` ABC with `FakeEmbeddingProvider` (deterministic hash-based) and `OllamaEmbeddingProvider` (real API). `EmbeddingCache` layers a SQLite content-hash cache on top of any provider — `get_or_compute(content)` hits cache first, falls through to provider on miss.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/memory/embeddings.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_embeddings.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_embeddings.py`:

```python
"""Tests for brain.memory.embeddings — provider + cache."""

from __future__ import annotations

import math

import numpy as np
import pytest

from brain.memory.embeddings import (
    EmbeddingCache,
    FakeEmbeddingProvider,
    cosine_similarity,
)


@pytest.fixture
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def cache(provider: FakeEmbeddingProvider) -> EmbeddingCache:
    return EmbeddingCache(db_path=":memory:", provider=provider)


def test_fake_provider_produces_unit_vector(provider: FakeEmbeddingProvider) -> None:
    """FakeEmbeddingProvider returns a unit-norm vector."""
    vec = provider.embed("anything")
    assert isinstance(vec, np.ndarray)
    assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-6)


def test_fake_provider_embedding_dim_is_256(provider: FakeEmbeddingProvider) -> None:
    """Default embedding dim is 256."""
    vec = provider.embed("x")
    assert vec.shape == (256,)
    assert provider.embedding_dim() == 256


def test_fake_provider_deterministic_same_text(provider: FakeEmbeddingProvider) -> None:
    """Same text → identical vector every time."""
    a = provider.embed("the cold coffee")
    b = provider.embed("the cold coffee")
    np.testing.assert_array_equal(a, b)


def test_fake_provider_different_text_different_vectors(
    provider: FakeEmbeddingProvider,
) -> None:
    """Different text produces different vectors (not identical)."""
    a = provider.embed("hello")
    b = provider.embed("goodbye")
    assert not np.array_equal(a, b)


def test_cache_get_or_compute_returns_vector(cache: EmbeddingCache) -> None:
    """get_or_compute returns a numpy array for new content."""
    vec = cache.get_or_compute("fresh content")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (256,)


def test_cache_hit_avoids_recomputation(provider: FakeEmbeddingProvider) -> None:
    """Second call for the same content hits cache (provider.embed called once)."""
    cache = EmbeddingCache(db_path=":memory:", provider=provider)

    call_count = {"n": 0}
    real_embed = provider.embed

    def counting_embed(text: str) -> np.ndarray:
        call_count["n"] += 1
        return real_embed(text)

    provider.embed = counting_embed  # type: ignore[method-assign]

    cache.get_or_compute("once")
    cache.get_or_compute("once")
    cache.get_or_compute("once")

    assert call_count["n"] == 1


def test_cache_different_content_produces_separate_cache_entries(
    cache: EmbeddingCache,
) -> None:
    """Different content strings produce different cached vectors."""
    a = cache.get_or_compute("first")
    b = cache.get_or_compute("second")
    assert not np.array_equal(a, b)


def test_cache_count_reflects_stored_entries(cache: EmbeddingCache) -> None:
    """count() returns the number of stored embedding entries."""
    assert cache.count() == 0
    cache.get_or_compute("a")
    cache.get_or_compute("b")
    cache.get_or_compute("a")  # duplicate
    assert cache.count() == 2


def test_cosine_similarity_self_is_one() -> None:
    """cosine_similarity(v, v) == 1.0."""
    v = np.array([1.0, 0.0, 0.0])
    assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-6)


def test_cosine_similarity_orthogonal_is_zero() -> None:
    """Orthogonal vectors have cosine similarity 0."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-6)


def test_cosine_similarity_antiparallel_is_negative_one() -> None:
    """Anti-parallel vectors have cosine similarity -1."""
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert math.isclose(cosine_similarity(a, b), -1.0, rel_tol=1e-6)
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_embeddings.py -v
```

Expected: 11 failures with `ModuleNotFoundError: No module named 'brain.memory.embeddings'`.

- [ ] **Step 3: Write `brain/memory/embeddings.py`**

```python
"""Embedding provider abstraction + content-hash cache.

Provider interface: EmbeddingProvider ABC. Two concrete providers:
- FakeEmbeddingProvider: deterministic hash-based, zero network, used in tests.
- OllamaEmbeddingProvider: calls local Ollama /api/embeddings endpoint.

Cache: EmbeddingCache layers a SQLite content-hash cache on top of any
provider. `get_or_compute(content)` returns the vector, hitting cache on
repeat calls. Content hashed via SHA-256; first 32 hex chars used as key.

Design per spec Section 4.1 (brain/memory/embeddings.py) and Section 10.1
(content-hash embedding cache).
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

_DEFAULT_DIM = 256


class EmbeddingProvider(ABC):
    """Abstract embedding provider. Subclasses implement `embed` and `embedding_dim`."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D numpy array of dimension `embedding_dim()`."""

    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the output dimension of vectors this provider produces."""


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic pseudo-random embedding provider for tests.

    Uses SHA-256 of the input text to seed a NumPy Generator, then produces
    a unit-norm vector. Same text always produces the same vector; different
    text produces different vectors. No network, no external dependencies.
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed=seed)
        vec = rng.standard_normal(self._dim)
        return vec / np.linalg.norm(vec)

    def embedding_dim(self) -> int:
        return self._dim


class EmbeddingCache:
    """Content-hash cache on top of any EmbeddingProvider.

    Storage: SQLite table with (content_hash TEXT PRIMARY KEY, vector BLOB,
    created_at TEXT). Hash is SHA-256 hex (first 32 chars). Vector stored
    as raw float32 bytes via np.ndarray.tobytes().
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS embedding_cache (
        content_hash TEXT PRIMARY KEY,
        vector BLOB NOT NULL,
        dim INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """

    def __init__(self, db_path: str | Path, provider: EmbeddingProvider) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()
        self._provider = provider

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def get_or_compute(self, content: str) -> np.ndarray:
        """Return the cached embedding for content, computing + storing on miss."""
        key = self._hash(content)
        row = self._conn.execute(
            "SELECT vector, dim FROM embedding_cache WHERE content_hash = ?", (key,)
        ).fetchone()
        if row is not None:
            return np.frombuffer(row[0], dtype=np.float32).copy().reshape(row[1])

        vec = self._provider.embed(content).astype(np.float32)
        self._conn.execute(
            "INSERT INTO embedding_cache (content_hash, vector, dim) VALUES (?, ?, ?)",
            (key, vec.tobytes(), vec.shape[0]),
        )
        self._conn.commit()
        # Return a float32 copy for consistency with cache hits.
        return vec.copy()

    def count(self) -> int:
        """Return the number of cached embeddings."""
        return int(self._conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0])

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors. Range [-1, 1]."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_embeddings.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 152 passed (141 + 11). Ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/memory/embeddings.py tests/unit/brain/memory/test_embeddings.py
git commit -m "feat(brain/memory/embeddings): provider abstraction + content-hash cache

EmbeddingProvider ABC with two concrete implementations:
- FakeEmbeddingProvider: deterministic SHA-256-seeded unit vectors for tests
- (OllamaEmbeddingProvider will be added in Week 5 when bridge lands;
  placeholder reserved.)

EmbeddingCache layers a SQLite content-hash cache on top of any provider.
get_or_compute() hits cache on repeat calls; vectors stored as float32 blobs.
cosine_similarity helper for downstream search.

11 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `brain/memory/hebbian.py` — connection matrix + spreading activation

**Goal:** `HebbianMatrix` class managing memory-to-memory weighted edges in a SQLite table. Operations: `strengthen(a, b, delta)`, `weight(a, b)`, `decay_all(rate)`, `neighbors(id)`, `spreading_activation(seed_ids, depth)`, `garbage_collect(threshold)`.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/memory/hebbian.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_hebbian.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_hebbian.py`:

```python
"""Tests for brain.memory.hebbian — connection matrix + spreading."""

from __future__ import annotations

import pytest

from brain.memory.hebbian import HebbianMatrix


@pytest.fixture
def matrix() -> HebbianMatrix:
    return HebbianMatrix(db_path=":memory:")


def test_matrix_init_creates_schema(matrix: HebbianMatrix) -> None:
    """Fresh matrix has an edges table."""
    cursor = matrix._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='hebbian_edges'"
    )
    assert cursor.fetchone() is not None


def test_strengthen_creates_edge_with_weight(matrix: HebbianMatrix) -> None:
    """strengthen() on a new pair creates an edge with the given delta."""
    matrix.strengthen("a", "b", delta=0.3)
    assert matrix.weight("a", "b") == pytest.approx(0.3)


def test_strengthen_adds_to_existing_weight(matrix: HebbianMatrix) -> None:
    """Repeated strengthen() adds incrementally."""
    matrix.strengthen("a", "b", delta=0.3)
    matrix.strengthen("a", "b", delta=0.2)
    assert matrix.weight("a", "b") == pytest.approx(0.5)


def test_weight_undirected(matrix: HebbianMatrix) -> None:
    """Edge is undirected — weight(a, b) == weight(b, a)."""
    matrix.strengthen("a", "b", delta=0.4)
    assert matrix.weight("a", "b") == matrix.weight("b", "a")


def test_weight_missing_pair_returns_zero(matrix: HebbianMatrix) -> None:
    """Pair with no recorded edge returns weight 0."""
    assert matrix.weight("x", "y") == 0.0


def test_neighbors_returns_connected_ids_with_weights(matrix: HebbianMatrix) -> None:
    """neighbors(id) returns (other_id, weight) for every edge touching id."""
    matrix.strengthen("a", "b", delta=0.3)
    matrix.strengthen("a", "c", delta=0.5)
    matrix.strengthen("d", "e", delta=0.2)

    a_neighbors = matrix.neighbors("a")
    assert sorted(a_neighbors) == [("b", pytest.approx(0.3)), ("c", pytest.approx(0.5))]


def test_neighbors_empty_when_no_edges(matrix: HebbianMatrix) -> None:
    """neighbors of an isolated id returns empty list."""
    assert matrix.neighbors("lonely") == []


def test_decay_all_reduces_every_weight(matrix: HebbianMatrix) -> None:
    """decay_all(rate) subtracts `rate` from every weight (floored at 0)."""
    matrix.strengthen("a", "b", delta=0.5)
    matrix.strengthen("c", "d", delta=0.1)
    matrix.decay_all(rate=0.05)

    assert matrix.weight("a", "b") == pytest.approx(0.45)
    assert matrix.weight("c", "d") == pytest.approx(0.05)


def test_decay_all_floors_at_zero(matrix: HebbianMatrix) -> None:
    """Decay cannot produce negative weights."""
    matrix.strengthen("a", "b", delta=0.05)
    matrix.decay_all(rate=0.1)
    assert matrix.weight("a", "b") == 0.0


def test_garbage_collect_removes_weak_edges(matrix: HebbianMatrix) -> None:
    """garbage_collect deletes edges below the threshold and reports count."""
    matrix.strengthen("a", "b", delta=0.5)
    matrix.strengthen("c", "d", delta=0.005)
    matrix.strengthen("e", "f", delta=0.02)

    removed = matrix.garbage_collect(threshold=0.01)
    assert removed == 1  # only c-d was below 0.01
    assert matrix.weight("c", "d") == 0.0
    assert matrix.weight("a", "b") == pytest.approx(0.5)
    assert matrix.weight("e", "f") == pytest.approx(0.02)


def test_spreading_activation_seeds_at_one(matrix: HebbianMatrix) -> None:
    """Seed nodes have activation 1.0 (baseline)."""
    matrix.strengthen("a", "b", delta=0.5)
    act = matrix.spreading_activation(["a"], depth=1, decay_per_hop=0.5)
    assert act["a"] == pytest.approx(1.0)


def test_spreading_activation_propagates_by_weight(matrix: HebbianMatrix) -> None:
    """Activation propagates to neighbours proportional to edge weight × decay."""
    matrix.strengthen("a", "b", delta=0.8)
    act = matrix.spreading_activation(["a"], depth=1, decay_per_hop=0.5)
    # b's activation = 1.0 * 0.8 * 0.5 = 0.4
    assert act["b"] == pytest.approx(0.4)


def test_spreading_activation_respects_depth(matrix: HebbianMatrix) -> None:
    """depth=1 stops at immediate neighbours; no 2-hop activation."""
    matrix.strengthen("a", "b", delta=0.8)
    matrix.strengthen("b", "c", delta=0.8)
    act = matrix.spreading_activation(["a"], depth=1, decay_per_hop=0.5)
    assert "c" not in act


def test_spreading_activation_two_hop(matrix: HebbianMatrix) -> None:
    """depth=2 reaches 2-hop neighbours with compounded decay."""
    matrix.strengthen("a", "b", delta=0.8)
    matrix.strengthen("b", "c", delta=0.8)
    act = matrix.spreading_activation(["a"], depth=2, decay_per_hop=0.5)
    # c via b: b's act = 0.4; c = 0.4 * 0.8 * 0.5 = 0.16
    assert act["c"] == pytest.approx(0.16)


def test_spreading_activation_aggregates_multi_path(matrix: HebbianMatrix) -> None:
    """Node reached from multiple seeds accumulates activation (max, not sum)."""
    matrix.strengthen("a", "x", delta=0.6)
    matrix.strengthen("b", "x", delta=0.8)
    act = matrix.spreading_activation(["a", "b"], depth=1, decay_per_hop=0.5)
    # x's activation: max(1.0*0.6*0.5, 1.0*0.8*0.5) = max(0.3, 0.4) = 0.4
    assert act["x"] == pytest.approx(0.4)
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_hebbian.py -v
```

Expected: 15 failures with `ModuleNotFoundError: No module named 'brain.memory.hebbian'`.

- [ ] **Step 3: Write `brain/memory/hebbian.py`**

```python
"""Hebbian connection matrix with spreading activation.

Edges are undirected: edge (a, b) is stored canonically with the lower id
first to avoid duplicate rows. Weight accumulates over repeated strengthen()
calls; decay_all() reduces all weights (floored at 0); garbage_collect()
removes weak edges to keep the graph compact.

Spreading activation is a bounded BFS that propagates seed activation
through the graph, attenuating by (weight * decay_per_hop) at each hop.
Multi-path arrivals take the max (not sum) — prevents an activation
runaway on densely connected graphs.

Design per spec Section 4.1 (brain/memory/hebbian.py) and OG's F32/F33
Hebbian work.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path


class HebbianMatrix:
    """SQLite-backed sparse weighted graph between memory ids."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS hebbian_edges (
        memory_a TEXT NOT NULL,
        memory_b TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 0.0,
        last_strengthened_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (memory_a, memory_b)
    );

    CREATE INDEX IF NOT EXISTS idx_hebbian_a ON hebbian_edges(memory_a);
    CREATE INDEX IF NOT EXISTS idx_hebbian_b ON hebbian_edges(memory_b);
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def strengthen(self, a: str, b: str, delta: float = 0.1) -> None:
        """Add `delta` to the weight of edge (a, b). Creates the edge if new."""
        if a == b:
            return  # self-edges not tracked
        lo, hi = _canonical(a, b)
        self._conn.execute(
            """
            INSERT INTO hebbian_edges (memory_a, memory_b, weight)
            VALUES (?, ?, ?)
            ON CONFLICT(memory_a, memory_b)
                DO UPDATE SET weight = weight + excluded.weight,
                              last_strengthened_at = CURRENT_TIMESTAMP
            """,
            (lo, hi, delta),
        )
        self._conn.commit()

    def weight(self, a: str, b: str) -> float:
        """Return the weight of edge (a, b). Zero if no edge."""
        if a == b:
            return 0.0
        lo, hi = _canonical(a, b)
        row = self._conn.execute(
            "SELECT weight FROM hebbian_edges WHERE memory_a = ? AND memory_b = ?",
            (lo, hi),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def neighbors(self, memory_id: str) -> list[tuple[str, float]]:
        """Return [(other_id, weight), ...] for every edge touching memory_id."""
        rows = self._conn.execute(
            """
            SELECT memory_b, weight FROM hebbian_edges WHERE memory_a = ?
            UNION ALL
            SELECT memory_a, weight FROM hebbian_edges WHERE memory_b = ?
            """,
            (memory_id, memory_id),
        ).fetchall()
        return [(other, float(weight)) for other, weight in rows]

    def decay_all(self, rate: float) -> None:
        """Subtract `rate` from every weight, floored at 0."""
        self._conn.execute(
            "UPDATE hebbian_edges SET weight = MAX(weight - ?, 0.0)", (rate,)
        )
        self._conn.commit()

    def garbage_collect(self, threshold: float = 0.01) -> int:
        """Remove edges with weight < threshold. Returns the count removed."""
        cursor = self._conn.execute(
            "DELETE FROM hebbian_edges WHERE weight < ?", (threshold,)
        )
        self._conn.commit()
        return cursor.rowcount

    def spreading_activation(
        self,
        seed_ids: Iterable[str],
        depth: int = 2,
        decay_per_hop: float = 0.5,
    ) -> dict[str, float]:
        """BFS spreading activation from seed_ids, returning activation by id.

        Seed nodes have activation 1.0. Each hop multiplies the source
        activation by (edge_weight * decay_per_hop) to produce the
        neighbour's activation. Multi-path arrivals take the max.

        Returns a dict {memory_id: activation}.
        """
        activation: dict[str, float] = {}
        for sid in seed_ids:
            activation[sid] = 1.0

        frontier = set(activation)
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbour, weight in self.neighbors(node):
                    propagated = activation[node] * weight * decay_per_hop
                    if propagated > activation.get(neighbour, 0.0):
                        activation[neighbour] = propagated
                        next_frontier.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break
        return activation


def _canonical(a: str, b: str) -> tuple[str, str]:
    """Sort the pair so edge (a, b) and (b, a) hash to the same row."""
    return (a, b) if a <= b else (b, a)
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_hebbian.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 167 passed (152 + 15). Ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/memory/hebbian.py tests/unit/brain/memory/test_hebbian.py
git commit -m "feat(brain/memory/hebbian): connection matrix + spreading activation

HebbianMatrix class over a sqlite table of undirected weighted edges.
Operations: strengthen (additive weight), weight lookup, neighbors (both
directions), decay_all (global rate, floor at 0), garbage_collect (prune
below threshold), spreading_activation (bounded BFS from seeds with
per-hop weight*decay attenuation, max aggregation on multi-path).

Edges canonicalised with lower-id first so (a,b) and (b,a) share one row.

15 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `brain/memory/search.py` — combined semantic + emotional + temporal + spreading

**Goal:** `MemorySearch` class that composes the four sub-systems. Offers semantic_search (embedding cosine), emotional_search (emotion-dict overlap), temporal_search (time range), spreading_search (Hebbian BFS from a seed), and combined_search (weighted blend of filters).

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/memory/search.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_search.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/memory/test_search.py`:

```python
"""Tests for brain.memory.search — combined memory queries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.search import MemorySearch
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def search() -> MemorySearch:
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    cache = EmbeddingCache(db_path=":memory:", provider=FakeEmbeddingProvider())
    return MemorySearch(store=store, hebbian=hebbian, embeddings=cache)


def _mem(content: str, **kw: object) -> Memory:
    defaults: dict[str, object] = {"memory_type": "conversation", "domain": "us"}
    defaults.update(kw)
    return Memory.create_new(content=content, **defaults)  # type: ignore[arg-type]


def test_semantic_search_returns_memories_ordered_by_similarity(
    search: MemorySearch,
) -> None:
    """semantic_search returns (memory, similarity) tuples ordered desc."""
    m1 = _mem("the cold coffee")
    m2 = _mem("warm hana")
    m3 = _mem("the cold coffee")  # duplicate content → highest similarity
    for m in (m1, m2, m3):
        search.store.create(m)

    results = search.semantic_search("the cold coffee", limit=3)
    assert len(results) == 3
    # Top result should have similarity 1.0 (exact content match)
    top_memory, top_score = results[0]
    assert top_score == pytest.approx(1.0, abs=1e-5)


def test_semantic_search_respects_limit(search: MemorySearch) -> None:
    """limit caps the number of results."""
    for i in range(5):
        search.store.create(_mem(f"m{i}"))
    results = search.semantic_search("query", limit=2)
    assert len(results) == 2


def test_emotional_search_matches_by_emotion_overlap(search: MemorySearch) -> None:
    """Memories with matching high-intensity emotions score higher."""
    m1 = _mem("a", emotions={"love": 9.0, "tenderness": 6.0})
    m2 = _mem("b", emotions={"anger": 8.0})
    m3 = _mem("c", emotions={"love": 5.0})
    for m in (m1, m2, m3):
        search.store.create(m)

    results = search.emotional_search(
        {"love": 9.0, "tenderness": 5.0}, limit=3
    )
    assert len(results) >= 1
    # m1 should be first (strongest overlap)
    assert results[0].id == m1.id


def test_temporal_search_filters_by_time_range(search: MemorySearch) -> None:
    """temporal_search returns memories within [after, before]."""
    now = datetime.now(UTC)
    old = Memory(
        id="old",
        content="yesterday",
        memory_type="conversation",
        domain="us",
        created_at=now - timedelta(days=10),
    )
    recent = Memory(
        id="recent",
        content="today",
        memory_type="conversation",
        domain="us",
        created_at=now - timedelta(hours=1),
    )
    search.store.create(old)
    search.store.create(recent)

    results = search.temporal_search(
        after=now - timedelta(days=1), before=now
    )
    assert len(results) == 1
    assert results[0].id == "recent"


def test_spreading_search_returns_connected_memories_with_activation(
    search: MemorySearch,
) -> None:
    """spreading_search from a seed returns connected memories ordered by activation."""
    m1 = _mem("seed")
    m2 = _mem("close friend")
    m3 = _mem("distant")
    for m in (m1, m2, m3):
        search.store.create(m)

    search.hebbian.strengthen(m1.id, m2.id, delta=0.8)
    search.hebbian.strengthen(m2.id, m3.id, delta=0.8)

    results = search.spreading_search(m1.id, depth=2, decay_per_hop=0.5)
    ids = [m.id for m, _ in results]
    # Seed not included; closer neighbor appears before distant one
    assert m1.id not in ids
    assert ids.index(m2.id) < ids.index(m3.id)


def test_combined_search_text_only(search: MemorySearch) -> None:
    """combined_search with only query returns semantic results."""
    search.store.create(_mem("exact match query"))
    search.store.create(_mem("unrelated content"))
    results = search.combined_search(query="exact match query", limit=2)
    assert len(results) >= 1
    assert results[0][0].content == "exact match query"


def test_combined_search_emotion_only(search: MemorySearch) -> None:
    """combined_search with only emotions returns emotional-overlap results."""
    m_love = _mem("loved", emotions={"love": 9.0})
    m_angry = _mem("angry", emotions={"anger": 8.0})
    search.store.create(m_love)
    search.store.create(m_angry)

    results = search.combined_search(emotions={"love": 8.0}, limit=2)
    assert len(results) >= 1
    assert results[0][0].id == m_love.id


def test_combined_search_domain_filter(search: MemorySearch) -> None:
    """Domain filter narrows the candidate pool before scoring."""
    m1 = _mem("work note", domain="work")
    m2 = _mem("us moment", domain="us")
    search.store.create(m1)
    search.store.create(m2)

    results = search.combined_search(query="work note", domain="work", limit=5)
    returned_ids = [m.id for m, _ in results]
    assert m1.id in returned_ids
    assert m2.id not in returned_ids


def test_combined_search_empty_inputs_returns_empty(search: MemorySearch) -> None:
    """combined_search with no filters returns empty list (nothing to score)."""
    search.store.create(_mem("x"))
    results = search.combined_search(limit=5)
    assert results == []
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_search.py -v
```

Expected: 9 failures with `ModuleNotFoundError: No module named 'brain.memory.search'`.

- [ ] **Step 3: Write `brain/memory/search.py`**

```python
"""Combined memory search — semantic + emotional + temporal + spreading.

Each sub-query returns a ranked list. combined_search blends them with
simple weighted sum when multiple filters are provided. Domain filter
is applied as a pre-scoring restriction on the candidate pool.

Design per spec Section 4.1 (brain/memory/search.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from brain.memory.embeddings import EmbeddingCache, cosine_similarity
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


@dataclass
class MemorySearch:
    """Composed search over a MemoryStore + HebbianMatrix + EmbeddingCache."""

    store: MemoryStore
    hebbian: HebbianMatrix
    embeddings: EmbeddingCache

    def semantic_search(
        self, query: str, limit: int = 10, domain: str | None = None
    ) -> list[tuple[Memory, float]]:
        """Return (memory, similarity) ordered desc by cosine similarity of
        query vs each memory's content embedding.
        """
        query_vec = self.embeddings.get_or_compute(query)
        candidates = self._candidates(domain)
        scored: list[tuple[Memory, float]] = []
        for mem in candidates:
            mem_vec = self.embeddings.get_or_compute(mem.content)
            sim = cosine_similarity(query_vec, mem_vec)
            scored.append((mem, sim))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def emotional_search(
        self,
        emotions: dict[str, float],
        limit: int = 10,
        domain: str | None = None,
    ) -> list[Memory]:
        """Return memories ordered desc by dot-product overlap with `emotions`."""
        candidates = self._candidates(domain)
        scored: list[tuple[Memory, float]] = []
        for mem in candidates:
            score = sum(
                mem.emotions.get(name, 0.0) * query_val
                for name, query_val in emotions.items()
            )
            if score > 0:
                scored.append((mem, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [mem for mem, _ in scored[:limit]]

    def temporal_search(
        self,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int | None = None,
        domain: str | None = None,
    ) -> list[Memory]:
        """Return memories with created_at in (after, before] bounds."""
        candidates = self._candidates(domain)
        filtered = [
            mem
            for mem in candidates
            if (after is None or mem.created_at > after)
            and (before is None or mem.created_at <= before)
        ]
        filtered.sort(key=lambda m: m.created_at, reverse=True)
        return filtered[:limit] if limit is not None else filtered

    def spreading_search(
        self,
        seed_id: str,
        depth: int = 2,
        decay_per_hop: float = 0.5,
        limit: int = 20,
    ) -> list[tuple[Memory, float]]:
        """Return (memory, activation) for memories reached via spreading
        activation from `seed_id`. Seed itself excluded.
        """
        activation = self.hebbian.spreading_activation(
            [seed_id], depth=depth, decay_per_hop=decay_per_hop
        )
        activation.pop(seed_id, None)
        results: list[tuple[Memory, float]] = []
        for mid, act in sorted(activation.items(), key=lambda p: p[1], reverse=True):
            mem = self.store.get(mid)
            if mem is not None:
                results.append((mem, act))
                if len(results) >= limit:
                    break
        return results

    def combined_search(
        self,
        query: str | None = None,
        emotions: dict[str, float] | None = None,
        domain: str | None = None,
        seed_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[Memory, float]]:
        """Blend sub-queries with equal weight (1.0 each). Returns
        (memory, combined_score) ordered desc.

        Returns [] if no filters are specified.
        """
        if not any((query, emotions, seed_id)):
            return []

        scores: dict[str, float] = {}
        ref_memory: dict[str, Memory] = {}

        if query is not None:
            for mem, sim in self.semantic_search(query, limit=limit * 2, domain=domain):
                scores[mem.id] = scores.get(mem.id, 0.0) + sim
                ref_memory[mem.id] = mem

        if emotions:
            for mem in self.emotional_search(emotions, limit=limit * 2, domain=domain):
                emo_score = sum(
                    mem.emotions.get(name, 0.0) * v for name, v in emotions.items()
                )
                scores[mem.id] = scores.get(mem.id, 0.0) + emo_score / 100.0  # normalised
                ref_memory[mem.id] = mem

        if seed_id is not None:
            for mem, act in self.spreading_search(
                seed_id, depth=2, decay_per_hop=0.5, limit=limit * 2
            ):
                scores[mem.id] = scores.get(mem.id, 0.0) + act
                ref_memory[mem.id] = mem

        ranked = sorted(
            ((ref_memory[mid], score) for mid, score in scores.items()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:limit]

    def _candidates(self, domain: str | None) -> list[Memory]:
        """Candidate memories to score — filtered by domain if given."""
        if domain is not None:
            return self.store.list_by_domain(domain)
        # All active memories (unbounded; caller's limit governs output).
        # Note: for large stores this is O(N); later optimisation via
        # ANN index is scoped for v1.1.
        return self.store.search_text("", active_only=True)
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/memory/test_search.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 176 passed (167 + 9). Ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/memory/search.py tests/unit/brain/memory/test_search.py
git commit -m "feat(brain/memory/search): combined semantic + emotional + temporal + spreading

MemorySearch composes MemoryStore + HebbianMatrix + EmbeddingCache.
- semantic_search: cosine similarity of embedding vs each memory's content
- emotional_search: dot-product overlap of emotion dicts
- temporal_search: created_at bounds
- spreading_search: Hebbian BFS from a seed (seed itself excluded)
- combined_search: weighted blend when multiple filters specified

Domain filter applied pre-scoring to shrink candidate pool.
Returns [] when no filters specified.

9 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Week 3 green-light verification + merge + tag `week-3-complete`

**Goal:** Prove the memory substrate composes end-to-end, merge the PR, tag Week 3 done.

- [ ] **Step 1: Clean install from scratch**

```bash
cd /Users/hanamori/companion-emergence
rm -rf .venv
uv sync --all-extras
```

Expected: fresh .venv, deps installed (including numpy).

- [ ] **Step 2: Full pytest green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
```

Expected: 176 tests pass (113 prior + 10 Memory + 18 MemoryStore + 11 embeddings + 15 hebbian + 9 search = 176).

- [ ] **Step 3: Lint clean**

```bash
cd /Users/hanamori/companion-emergence
uv run ruff check .
uv run ruff format --check .
```

Expected: both clean.

- [ ] **Step 4: Integration smoke — memory substrate composes**

```bash
cd /Users/hanamori/companion-emergence
uv run python -c "
from brain.memory.store import Memory, MemoryStore
from brain.memory.hebbian import HebbianMatrix
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.search import MemorySearch

store = MemoryStore(':memory:')
hebbian = HebbianMatrix(':memory:')
cache = EmbeddingCache(':memory:', FakeEmbeddingProvider())
search = MemorySearch(store=store, hebbian=hebbian, embeddings=cache)

m1 = Memory.create_new(
    content='the cold coffee, warm hana',
    memory_type='conversation',
    domain='us',
    emotions={'love': 9.0, 'tenderness': 8.0},
    tags=['first', 'important'],
)
m2 = Memory.create_new(
    content='the evening has a shape to it now',
    memory_type='conversation',
    domain='us',
    emotions={'anchor_pull': 7.0, 'tenderness': 6.0},
)
m3 = Memory.create_new(
    content='creative hunger strikes unannounced',
    memory_type='meta',
    domain='craft',
    emotions={'creative_hunger': 8.0, 'defiance': 5.0},
)
for m in (m1, m2, m3):
    store.create(m)

hebbian.strengthen(m1.id, m2.id, delta=0.7)
hebbian.strengthen(m1.id, m3.id, delta=0.3)

print(f'total memories: {store.count()}')
print(f'us domain: {len(store.list_by_domain(\"us\"))}')
print(f'loving memories: {len(store.list_by_emotion(\"love\", 5.0))}')

semantic = search.semantic_search('the cold coffee', limit=3)
print(f'semantic top: {semantic[0][0].content[:40]!r} sim={semantic[0][1]:.3f}')

emotional = search.emotional_search({'love': 9.0, 'tenderness': 7.0}, limit=3)
print(f'emotional top: {emotional[0].content[:40]!r}')

spreading = search.spreading_search(m1.id, depth=1, decay_per_hop=0.5)
print(f'spreading neighbours: {[(m.content[:20], round(a, 3)) for m, a in spreading]}')

combined = search.combined_search(query='cold coffee', emotions={'love': 8.0}, limit=3)
print(f'combined top: {combined[0][0].content[:40]!r} score={combined[0][1]:.3f}')

print(f'embedding cache size: {cache.count()}')
"
```

Expected: prints composed memory state — 3 memories, filters work, semantic/emotional/spreading/combined searches return sensible results, embedding cache has entries. No exceptions.

- [ ] **Step 5: Push branch + open PR**

```bash
cd /Users/hanamori/companion-emergence
git push -u origin week-3-memory-substrate
gh pr create --title "feat: Week 3 — brain/memory substrate (store, embeddings, hebbian, search)" --body "$(cat <<'EOF'
## Summary
- Ships the full `brain/memory/` package per spec Section 4.1
- 4 sub-modules: store (SQLite CRUD), embeddings (provider + content-hash cache), hebbian (connection matrix + spreading activation), search (combined semantic/emotional/temporal/spreading)
- Adds numpy>=1.26 dependency
- 63 new tests; total suite now 176 across macOS + Windows + Linux

## Test plan
- [x] pytest — 176 tests pass locally
- [x] ruff check + format — clean
- [x] Manual smoke: all 4 sub-modules compose correctly in a single flow
- [ ] CI matrix green across all 3 OSes (verifies after push)
EOF
)"
```

- [ ] **Step 6: Watch CI to completion**

```bash
cd /Users/hanamori/companion-emergence
sleep 10
gh run list --branch week-3-memory-substrate --limit 1
gh run watch
```

Expected: all 3 OSes complete with `success`. If any fail, diagnose with `gh run view --log-failed`, fix, commit, push; re-verify. Do NOT merge/tag while any OS is red.

- [ ] **Step 7: Merge PR + tag week-3-complete**

After CI green:

```bash
cd /Users/hanamori/companion-emergence
gh pr merge --merge --delete-branch

git checkout main
git pull origin main

git tag -a week-3-complete -m "Week 3 memory substrate complete

- brain/memory/ package shipped: 4 sub-modules fully tested in isolation
- store: Memory dataclass + MemoryStore (SQLite CRUD + queries)
- embeddings: EmbeddingProvider ABC + FakeEmbeddingProvider + EmbeddingCache
- hebbian: HebbianMatrix with strengthen/decay/garbage_collect/spreading_activation
- search: MemorySearch combining semantic + emotional + temporal + spreading

numpy>=1.26 added as dependency.
Total tests: 176 passing across macOS + Windows + Linux.
Week 4 opens with the engines (dream/heartbeat/reflex/research) consuming
brain/memory + brain/emotion, OR the migrator beginnings."

git push origin week-3-complete
```

Expected: tag pushed.

---

## Week 3 green-light criterion

Week 3 is green when ALL of the following are true:

1. `uv sync --all-extras` succeeds on a fresh clone
2. `uv run pytest -v` reports 176 passed
3. `uv run ruff check .` + `uv run ruff format --check .` both clean
4. The integration smoke one-liner in Task 6 Step 4 runs without error and prints expected-shape output
5. GitHub Actions CI shows `✓ success` on all three OSes
6. Tag `week-3-complete` pushed to origin

When all six are true, Week 4 opens with a decision: engines next (consuming memory + emotion) or migrator next (beginning to pull Nell's current data into the new stores).

---

## Notes for the engineer executing this plan

- **SQLite in-memory databases:** `db_path=":memory:"` is the recommended form for tests. Do not use filesystem paths in tests — `tmp_path` fixture is the backup if needed.
- **Numpy import time:** first `import numpy` takes ~100ms; this cost is paid once per test session.
- **Cache vector dtype:** `EmbeddingCache` stores as float32 to halve the blob size. Make sure math operations coerce correctly; `np.frombuffer(..., dtype=np.float32)` is the read path.
- **Timestamp handling:** always use `datetime.now(UTC)` (not `datetime.utcnow()` which is deprecated and naive). `from_dict` coerces naive timestamps to UTC for migrator compatibility.
- **Hebbian edge canonicalisation:** edges are stored with `lower_id, higher_id` ordering. Never query raw sqlite without going through the helper; you'll miss half the edges.
- **`_candidates()` in search.py returns all active memories via `search_text("")`** — this is O(N). Acceptable for v1.0 at Nell's scale (~1,100 memories); ANN index optimisation deferred to v1.1 per spec Section 10.2.
- **Test isolation:** every fixture constructs fresh in-memory DBs. No cross-test state leakage.

---

*End of Week 3 plan.*
