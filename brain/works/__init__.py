"""brain.works — brain-authored creative artifact portfolio.

Public API:
    Work          — dataclass for one work
    WORK_TYPES    — frozenset of valid type strings
    make_work_id  — content-hash ID generator (SHA-256 prefix)

Storage and indexing live in submodules:
    brain.works.storage — markdown file I/O at persona/<name>/data/works/<id>.md
    brain.works.store   — SQLite index at persona/<name>/data/works.db

See docs/superpowers/specs/2026-05-04-nell-works-design.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

WORK_TYPES: frozenset[str] = frozenset(
    {"story", "code", "planning", "idea", "role_play", "letter", "other"}
)


@dataclass
class Work:
    """A single brain-authored work — story, code, planning doc, idea,
    role-play scene, letter, or other.

    Attributes:
        id: 12-char hex SHA-256 prefix of content (content-addressed).
        title: Nell-supplied, free-form, max 200 chars (validated at save).
        type: One of WORK_TYPES.
        created_at: tz-aware UTC datetime.
        session_id: Bridge session_id when available, None otherwise.
        word_count: Pre-computed for sorting/filtering without reading file.
        summary: Nell-supplied one-liner, optional, max 500 chars (validated at save).
    """

    id: str
    title: str
    type: str
    created_at: datetime
    session_id: str | None
    word_count: int
    summary: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (e.g., for SQLite row, JSON response, frontmatter)."""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "created_at": self.created_at.isoformat(),
            "session_id": self.session_id,
            "word_count": self.word_count,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Work:
        """Deserialize from dict. created_at must be ISO-8601 string."""
        created_at = datetime.fromisoformat(data["created_at"])
        return cls(
            id=data["id"],
            title=data["title"],
            type=data["type"],
            created_at=created_at,
            session_id=data.get("session_id"),
            word_count=int(data["word_count"]),
            summary=data.get("summary"),
        )


def make_work_id(content: str) -> str:
    """Generate a stable 12-char hex id from content via SHA-256 prefix.

    Same content → same id. Used to dedupe accidental double-saves and
    to address the on-disk markdown file (persona/<name>/data/works/<id>.md).
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return digest[:12]


__all__ = ["Work", "WORK_TYPES", "make_work_id"]
