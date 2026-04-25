"""Memory dataclass + SQLite-backed MemoryStore.

Design per spec Section 4.1 (brain/memory/store.py) and Section 10.1
(SQLite data layer replaces OG JSON/numpy files).

Memory is the canonical record type. MemoryStore is the CRUD surface over
a single SQLite database containing the `memories` table. Tasks 3-5 add
sibling modules (embeddings, hebbian, search) that read from and strengthen
this store.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _coerce_utc(ts: str) -> datetime:
    """Parse ISO-8601 timestamp; coerce tz-naive values to UTC.

    Shared by Memory.from_dict and MemoryStore._row_to_memory so both paths
    from storage apply identical naive→UTC handling — no drift risk when a
    third reader (e.g. migrator) is added.
    """
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _safe_load_metadata(raw: str | None) -> dict[str, Any]:
    """Decode a metadata_json column value into a dict, defending against
    manual DB edits or legacy writers that stored the string "null",
    malformed JSON, or non-dict top-level values.
    """
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
        score: sum of emotion intensities — snapshot at construction.
            Not updated if `emotions` is mutated in place after creation;
            consumers that want a live sum should recompute themselves.
        created_at: tz-aware UTC datetime of creation.
        last_accessed_at: tz-aware UTC datetime of most recent read, or None.
        active: F22 deactivation flag. Inactive memories are excluded from
            default queries but remain in the database (reversible).
        protected: excluded from decay/consolidation.
        metadata: free-form dict for fields not modelled as first-class
            attributes — absorbs OG-only fields (source_date, supersedes,
            etc.) during migration without proliferating the dataclass.
            Forward-compatible: future engines read metadata[key] as needed.
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
    metadata: dict[str, Any] = field(default_factory=dict)

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
        """Factory: new memory with generated UUID, current UTC time,
        and auto-computed score + importance (if importance is None).

        Score = sum of emotion intensities.
        Importance defaults to score/10.0 (normalised to 0..10 scale).
        """
        emotions = dict(emotions or {})
        tags = list(tags or [])
        score = float(sum(emotions.values()))
        metadata = dict(metadata or {})
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
            metadata=metadata,
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
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Restore from a dict produced by to_dict.

        Tz-naive timestamps are coerced to UTC (permissive for migrator input).
        """
        created = _coerce_utc(data["created_at"])
        last_accessed = (
            _coerce_utc(data["last_accessed_at"]) if data.get("last_accessed_at") else None
        )
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
            metadata=dict(data.get("metadata") or {}),
        )


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

CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""


_ALLOWED_FILTER_COLUMNS = frozenset({"domain", "memory_type"})


class MemoryStore:
    """SQLite-backed store for Memory records.

    Pass `":memory:"` as db_path for in-memory databases (used in tests).
    Any filesystem path creates or opens a persistent database.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        # Run integrity check BEFORE setting row_factory so result rows are
        # plain tuples — the comparison [("ok",)] is unambiguous.
        try:
            result = self._conn.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.DatabaseError as exc:
            self._conn.close()
            from brain.health.anomaly import BrainIntegrityError

            raise BrainIntegrityError(str(db_path), str(exc)) from exc
        if result != [("ok",)]:
            detail = "; ".join(str(row[0]) for row in result)
            self._conn.close()
            from brain.health.anomaly import BrainIntegrityError

            raise BrainIntegrityError(str(db_path), detail)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection. Safe to call multiple times."""
        self._conn.close()

    def create(self, memory: Memory) -> str:
        """Insert a memory. Returns the id. Raises on duplicate id."""
        try:
            metadata_json = json.dumps(memory.metadata)
        except TypeError as exc:
            raise TypeError(
                f"Memory.metadata for id={memory.id!r} contains non-JSON-serialisable values: {exc}"
            ) from exc
        self._conn.execute(
            """
            INSERT INTO memories (
                id, content, memory_type, domain, emotions_json, tags_json,
                importance, score, created_at, last_accessed_at, active, protected,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                metadata_json,
            ),
        )
        self._conn.commit()
        return memory.id

    def get(self, memory_id: str) -> Memory | None:
        """Return the Memory with the given id, or None."""
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
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
        """Return memories where `emotion_name` is present at >= min_intensity."""
        sql = "SELECT * FROM memories WHERE 1=1"
        if active_only:
            sql += " AND active = 1"
        sql += " ORDER BY created_at DESC"
        rows = self._conn.execute(sql).fetchall()

        results: list[Memory] = []
        for row in rows:
            emotions = json.loads(row["emotions_json"])
            intensity = emotions.get(emotion_name, 0.0)
            if isinstance(intensity, (int, float)) and intensity >= min_intensity:
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
            elif key == "metadata":
                column_map["metadata_json"] = ("metadata_json", json.dumps(value))
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

        # Empty `fields` kwargs — existence already verified above, nothing to write.
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
        """Case-insensitive substring search on content.

        `%` and `_` in `query` are escaped so a caller passing `"%"` does
        not match every row.
        """
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql = "SELECT * FROM memories WHERE content LIKE ? ESCAPE '\\' COLLATE NOCASE"
        params: list[Any] = [f"%{escaped}%"]
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
        if column not in _ALLOWED_FILTER_COLUMNS:
            raise ValueError(f"Invalid filter column: {column!r}")
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
    created = _coerce_utc(row["created_at"])
    last_accessed = _coerce_utc(row["last_accessed_at"]) if row["last_accessed_at"] else None
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
        metadata=_safe_load_metadata(row["metadata_json"]),
    )
