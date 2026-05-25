"""Read-only readers for a source persona dir — never mutate the source.

MemoryStore.__init__ ALTERs the table to add columns, so the pristine source
is read with a raw read-only sqlite connection instead.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from brain.memory.store import Memory, _coerce_utc


def _ro_connect(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def read_source_memories(source_dir: Path) -> dict[str, Memory]:
    """Map id -> Memory for every row in the source memories.db.

    Tolerates older schemas missing state/content_snapshot/recall_count by
    defaulting those fields. Skips rows missing id/content/created_at.
    """
    db = source_dir / "memories.db"
    if not db.is_file():
        return {}
    conn = _ro_connect(db)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
        rows = conn.execute("SELECT * FROM memories").fetchall()
    finally:
        conn.close()

    out: dict[str, Memory] = {}
    for row in rows:
        d = {k: row[k] for k in cols}
        mid, content, created = d.get("id"), d.get("content"), d.get("created_at")
        if not mid or not content or not created:
            continue
        try:
            created_at = _coerce_utc(created)
        except (ValueError, TypeError):
            continue
        last_accessed = None
        la = d.get("last_accessed_at")
        if isinstance(la, str) and la:
            try:
                last_accessed = _coerce_utc(la)
            except ValueError:
                last_accessed = None
        out[mid] = Memory(
            id=mid,
            content=content,
            memory_type=d.get("memory_type") or "conversation",
            domain=d.get("domain") or "us",
            created_at=created_at,
            emotions=_loads_dict(d.get("emotions_json")),
            tags=_loads_list(d.get("tags_json")),
            importance=float(d.get("importance") or 0.0),
            score=float(d.get("score") or 0.0),
            last_accessed_at=last_accessed,
            active=bool(d.get("active", 1)),
            protected=bool(d.get("protected", 0)),
            metadata=_loads_dict(d.get("metadata_json")),
            state=d.get("state") or "active",
            content_snapshot=d.get("content_snapshot"),
            recall_count=int(d.get("recall_count") or 0),
        )
    return out


def read_source_edges(source_dir: Path) -> list[tuple[str, str, float]]:
    """Return (a, b, weight) for every edge in the source hebbian.db."""
    db = source_dir / "hebbian.db"
    if not db.is_file():
        return []
    conn = _ro_connect(db)
    try:
        rows = conn.execute(
            "SELECT memory_a, memory_b, weight FROM hebbian_edges"
        ).fetchall()
    finally:
        conn.close()
    return [(r["memory_a"], r["memory_b"], float(r["weight"])) for r in rows]


def _loads_dict(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def _loads_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return []
