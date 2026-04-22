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
