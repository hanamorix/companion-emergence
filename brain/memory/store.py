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
import logging
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    state: str = "active"
    content_snapshot: str | None = None
    recall_count: float = 0.0
    peak_emotion_intensity: float = 0.0

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
            peak_emotion_intensity=max(
                (float(v) for v in emotions.values()), default=0.0
            ),
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
    metadata_json TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'active',
    content_snapshot TEXT,
    recall_count REAL NOT NULL DEFAULT 0,
    peak_emotion_intensity REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

-- External-content FTS5 shadow index (P2 relevance overhaul). `memories` is a
-- rowid table (id TEXT PRIMARY KEY → implicit integer rowid), so external
-- content with content_rowid='rowid' indexes only `content` (no duplication).
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, content='memories', content_rowid='rowid'
);

-- Tier-1 passive-recall salience: zero-dependency corpus IDF straight from
-- SQLite's own fts5vocab shadow of memories_fts ('row' mode → one row per
-- term with its document frequency in column `doc`). Virtual/read-only (no
-- storage, no write path); must be created AFTER memories_fts since it
-- indexes that table's content.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vocab USING fts5vocab('memories_fts', 'row');

-- Schema-level sync triggers: they live in the DB and fire on WHICHEVER
-- connection performs the write, so every one of the ~15 MemoryStore
-- connections maintains the index atomically within its own transaction.
CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


_ALLOWED_FILTER_COLUMNS = frozenset({"domain", "memory_type"})

# Minimum token length fed into the FTS5 MATCH sanitizer. Default 3 keeps short
# proper-ish terms while still dropping the shortest stopword-shaped fragments;
# every term is double-quoted anyway, so a token colliding with an FTS keyword
# (AND/OR/NOT/NEAR) is treated as a literal string, not an operator. Sibling to
# the recall/tool tokenizers' min_len=4.
_FTS_TOKEN_MIN_LEN = 3


def _to_fts_match(query: str) -> str:
    """Build an FTS5 MATCH expression from a raw query string.

    Tokenizes (split on ``[^A-Za-z0-9]+``, drop tokens shorter than
    ``_FTS_TOKEN_MIN_LEN``, dedup case-insensitively) and **OR-joins each term
    wrapped in double-quotes** — e.g. ``'"henryk" OR "preferences"'``.

    The OR is mandatory, not cosmetic: FTS5's default bare-term MATCH is an
    implicit AND, so a disjoint multi-term query would return ZERO rows. The
    double-quoting makes each term a literal FTS string, immune to a token that
    collides with an FTS keyword (AND/OR/NOT/NEAR) or a special char.

    Returns ``""`` when the query yields no usable tokens (caller returns no
    matches rather than issuing a MATCH).
    """
    seen: set[str] = set()
    terms: list[str] = []
    for piece in re.split(r"[^A-Za-z0-9]+", query):
        if len(piece) < _FTS_TOKEN_MIN_LEN:
            continue
        low = piece.lower()
        if low in seen:
            continue
        seen.add(low)
        terms.append(f'"{piece}"')
    return " OR ".join(terms)


def _bump_amount(bump: bool | float) -> float | None:
    """Normalize a ``bump`` kwarg (bool | float) to a concrete increment.

    ``True`` -> 1.0 (the genuine full-read amount); ``False`` -> ``None`` (no
    bump); a positive float -> that amount (the passive-recall fractional bump).
    A zero amount (``0``, ``0.0``, or ``False``) -> ``None``: it is a no-op, not
    a real bump. Returning 0.0 would still fire a ``recall_count + 0.0`` UPDATE
    (and, on the ``get()`` path, reset ``last_accessed_at``) for no
    reinforcement. Shared by every ``recall_count`` writer so the bool/float
    widening stays in one place.
    """
    if isinstance(bump, bool):
        return 1.0 if bump else None
    amount = float(bump)
    return amount if amount != 0.0 else None


class MemoryStore:
    """SQLite-backed store for Memory records.

    Pass `":memory:"` as db_path for in-memory databases (used in tests).
    Any filesystem path creates or opens a persistent database.
    """

    def __init__(self, db_path: str | Path, *, integrity_check: bool = True) -> None:
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(db_path))
        # Run integrity check BEFORE setting row_factory so result rows are
        # plain tuples — the comparison [("ok",)] is unambiguous. Hot request
        # paths may pass integrity_check=False and leave deep checks to health.
        if integrity_check:
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
        # WAL + 5s busy_timeout — the bridge runs concurrent writers
        # (chat tool calls, supervisor stale-close sweep, heartbeat,
        # growth). Without WAL these can surface as `database is
        # locked` under realistic desktop timing. In-memory dbs reject
        # WAL; the fallback keeps tests using `:memory:` working. We
        # set these AFTER the integrity check so a corrupt-file probe
        # still surfaces BrainIntegrityError instead of a pragma crash.
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # Idempotent column migration for upgraded personas — _SCHEMA's
        # CREATE TABLE IF NOT EXISTS leaves pre-existing tables alone, so
        # we check for missing columns + add them with their defaults.
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "state" not in existing:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN state TEXT NOT NULL DEFAULT 'active'"
            )
        if "content_snapshot" not in existing:
            self._conn.execute("ALTER TABLE memories ADD COLUMN content_snapshot TEXT")
        if "recall_count" not in existing:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN recall_count REAL NOT NULL DEFAULT 0"
            )
        if "peak_emotion_intensity" not in existing:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN peak_emotion_intensity "
                "REAL NOT NULL DEFAULT 0.0"
            )
            # One-time seed from current intensities (column-existence is the
            # marker; v0.0.33 Track 3). Intensities already erased by the
            # pre-v0.0.32 stub decay are unrecoverable — those rows keep an
            # honest 0.0 (spec D4). Fail-soft wholesale: old schemas (e.g.
            # v0.0.12) use ``emotions`` not ``emotions_json`` — skip seeding
            # entirely if the column is absent; rows keep DEFAULT 0.0.
            if "emotions_json" in existing:
                try:
                    for row in self._conn.execute(
                        "SELECT id, emotions_json FROM memories"
                    ).fetchall():
                        try:
                            emotions = json.loads(row["emotions_json"]) or {}
                            vals = [
                                float(v)
                                for v in emotions.values()
                                if isinstance(v, (int, float))
                            ]
                        except (ValueError, TypeError):
                            vals = []
                        if vals:
                            self._conn.execute(
                                "UPDATE memories SET peak_emotion_intensity = ? WHERE id = ?",
                                (max(vals), row["id"]),
                            )
                except sqlite3.OperationalError as exc:
                    logger.warning("peak seeding skipped: %s", exc)
        # Index on state — used by forgetting pass to find fading rows fast.
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state)")
        self._conn.commit()
        self._boot_fts_backstop()

    def _boot_fts_backstop(self) -> None:
        """FTS5 boot integrity-check → rebuild backstop.

        Runs on EVERY constructor, including ``integrity_check=False`` ones
        (the hot/feed paths): this FTS check is separate from the ``memories.db``
        ``PRAGMA integrity_check`` gated by that flag — it is a cheap
        FTS-internal consistency probe, not a full-DB scan, and must run so a
        feed-path writer never operates against a stale/absent FTS index.

        Rebuilds when the shadow index is corrupt (integrity-check raises) OR
        empty against a pre-populated ``memories`` table — which covers both a
        persona whose ``memories.db`` predates FTS (the table was just created
        against existing rows) and a cleared/corrupted index (C2 self-heal).
        Fail-soft: a rebuild failure logs and leaves the LIKE fallback usable.
        """
        try:
            self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')")
            healthy = True
        except sqlite3.DatabaseError:
            healthy = False
        needs_rebuild = not healthy
        if healthy:
            try:
                mem_rows = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                # COUNT(*) on the FTS table itself reads the external CONTENT
                # table, not the index — use the `_docsize` shadow table for the
                # true count of INDEXED rows so an empty index against a
                # populated `memories` (predates-FTS, or a cleared index) is
                # detected.
                fts_rows = self._conn.execute(
                    "SELECT COUNT(*) FROM memories_fts_docsize"
                ).fetchone()[0]
                # mem_rows != fts_rows catches BOTH the empty-index case
                # (predates-FTS / cleared → fts_rows==0) AND partial staleness
                # (0 < fts_rows < mem_rows) at no extra cost (stage-6 minor).
                needs_rebuild = mem_rows != fts_rows
            except sqlite3.DatabaseError:
                needs_rebuild = True
        if needs_rebuild:
            try:
                self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            except sqlite3.DatabaseError as exc:
                logger.warning("memories_fts rebuild failed; LIKE fallback in use: %s", exc)
        # The 'integrity-check'/'rebuild' commands are INSERT statements, so
        # sqlite3 opens an implicit write transaction. Commit it unconditionally
        # (even the read-only integrity-check path) so no boot leaves a lock held
        # against the other ~15 concurrent MemoryStore connections to this file.
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection. Safe to call multiple times."""
        self._conn.close()

    @property
    def db_path(self) -> Path:
        """Filesystem path of the backing SQLite database (or ':memory:')."""
        return self._db_path

    @property
    def persona_dir(self) -> Path:
        """Persona directory containing this store — the parent of db_path.

        Function-style writers use this to locate the sibling pending-candidate
        queue (``pending_candidates.jsonl``) without threading persona_dir
        through every call. Meaningless for an in-memory store (parent of
        ':memory:'); such stores are test-only and never enqueue.
        """
        return self._db_path.parent

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
                metadata_json, peak_emotion_intensity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                max(
                    [memory.peak_emotion_intensity]
                    + [float(v) for v in memory.emotions.values()]
                ),
            ),
        )
        self._conn.commit()
        return memory.id

    def get(self, memory_id: str, *, bump: bool | float = True) -> Memory | None:
        """Return the Memory with the given id, or None. Bumps
        last_accessed_at + recall_count on hit so salience scoring sees
        the access (Forgetting integration — spec v0.0.14-alpha.3).

        bump: when True (default), a hit updates recall_count (+1.0) and
        last_accessed_at — the correct behaviour for a genuine full-read.
        False skips the bump entirely — internal existence-checks (e.g.
        update()/deactivate()'s pre-write lookup) pass ``bump=False`` so a
        maintenance write doesn't count as engagement. A float bumps
        recall_count by that amount instead of 1.0 (also touching
        last_accessed_at — this is a genuine-access path, unlike
        ``bump_recall``).
        """
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        amount = _bump_amount(bump)
        if amount is not None:
            now_iso = datetime.now(UTC).isoformat()
            self._conn.execute(
                "UPDATE memories SET recall_count = recall_count + ?, last_accessed_at = ? "
                "WHERE id = ?",
                (amount, now_iso, memory_id),
            )
            self._conn.commit()
        return _row_to_memory(row)

    def bump_recall(self, memory_id: str, amount: float) -> None:
        """Fractional ``recall_count`` bump for the passive-recall surface path.

        A single atomic ``UPDATE`` incrementing ``recall_count`` ONLY — this
        deliberately does NOT touch ``last_accessed_at``. A per-turn ambient
        surface is not a genuine "access event" the way a full read is;
        resetting the freshness anchor at full strength for every surfaced
        memory (including the rank-bottom ~0.1 one) would give an
        un-discounted freshness boost, defeating the rank-discount rationale
        (``salience.py``'s freshness arm anchors on ``last_accessed_at``).
        Callers already hold the ``Memory`` objects (``_build_recall_block``),
        so this does not re-``get()`` each surfaced memory.
        """
        self._conn.execute(
            "UPDATE memories SET recall_count = recall_count + ? WHERE id = ?",
            (amount, memory_id),
        )
        self._conn.commit()

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
        if self.get(memory_id, bump=False) is None:
            raise KeyError(f"Unknown memory id: {memory_id!r}")

        column_map: dict[str, tuple[str, Any]] = {}
        extra_clauses: list[str] = []
        extra_values: list[Any] = []
        for key, value in fields.items():
            if key == "emotions":
                column_map["emotions_json"] = ("emotions_json", json.dumps(value))
                new_max = max((float(v) for v in (value or {}).values()), default=0.0)
                extra_clauses.append(
                    "peak_emotion_intensity = MAX(peak_emotion_intensity, ?)"
                )
                extra_values.append(new_max)
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
        if not column_map and not extra_clauses:
            return
        set_clause = ", ".join(
            [f"{col} = ?" for col, _ in column_map.values()] + extra_clauses
        )
        values = [v for _, v in column_map.values()] + extra_values
        values.append(memory_id)
        self._conn.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
        self._conn.commit()

    def deactivate(self, memory_id: str) -> None:
        """Mark a memory inactive (F22 semantics). Raises KeyError if unknown."""
        if self.get(memory_id, bump=False) is None:
            raise KeyError(f"Unknown memory id: {memory_id!r}")
        self._conn.execute("UPDATE memories SET active = 0 WHERE id = ?", (memory_id,))
        self._conn.commit()

    def fade(self, memory_id: str, *, summary: str) -> None:
        """Fade a memory: snapshot content into content_snapshot, replace
        content with summary, set state='fading'. Raises KeyError if unknown.
        """
        row = self._conn.execute("SELECT state FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown memory id: {memory_id!r}")
        if row["state"] == "fading":
            logger.warning(
                "fade called on memory id=%s already in fading state — noop to preserve content_snapshot",
                memory_id,
            )
            return
        self._conn.execute(
            "UPDATE memories SET content_snapshot = content, content = ?, state = 'fading' WHERE id = ?",
            (summary, memory_id),
        )
        self._conn.commit()

    def unfade(self, memory_id: str) -> None:
        """Unfade a memory: restore content from content_snapshot, clear
        snapshot, set state='active'. NULL snapshot logs a warning and
        returns without mutating (defensive). Raises KeyError if unknown.
        """
        if not self._conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone():
            raise KeyError(f"Unknown memory id: {memory_id!r}")
        row = self._conn.execute(
            "SELECT content_snapshot FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row["content_snapshot"] is None:
            logger.warning(
                "unfade called on memory id=%s with NULL content_snapshot — noop",
                memory_id,
            )
            return
        self._conn.execute(
            "UPDATE memories SET content = content_snapshot, content_snapshot = NULL, state = 'active' WHERE id = ?",
            (memory_id,),
        )
        self._conn.commit()

    def hard_delete(self, memory_id: str) -> None:
        """Drop the row. Caller MUST write the graveyard entry first.
        Raises KeyError if unknown.
        """
        if not self._conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone():
            raise KeyError(f"Unknown memory id: {memory_id!r}")
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def count(self, active_only: bool = True) -> int:
        """Return the total count of memories."""
        sql = "SELECT COUNT(*) FROM memories"
        if active_only:
            sql += " WHERE active = 1"
        return int(self._conn.execute(sql).fetchone()[0])

    def term_stats(self, terms: list[str]) -> dict[str, tuple[int, float]]:
        """Return per-term document frequency + corpus IDF (Tier-1 recall salience).

        Looks up ``doc`` (document frequency) for each (lowercased) term from
        ``memories_vocab`` — the ``fts5vocab('memories_fts', 'row')`` shadow
        table — and computes ``idf = log(N / (1 + df))`` where ``N`` is the
        **all-indexed-docs** count (``SELECT count(*) FROM memories``). This
        deliberately does NOT reuse :meth:`count`, whose ``active_only=True``
        default would risk ``N < df`` and a negative IDF.

        Returns ``{lowered_term: (df, idf)}`` for every requested term — a
        term absent from the vocabulary gets ``df=0`` (present in the store's
        vocabulary is exactly what "absent" means here; the caller treats
        ``df=0`` as both lowest-recall-value and, in ``_build_recall_block``,
        "not recognised"). Fail-soft: returns ``{}`` when the store is empty
        (``N == 0``, so any IDF would be undefined) or on any sqlite error —
        callers fall back to non-IDF, non-``in_store`` salience. Read-only:
        no write, no recall_count bump.
        """
        if not terms:
            return {}
        lowered = [t.lower() for t in terms]
        try:
            n = int(self._conn.execute("SELECT count(*) FROM memories").fetchone()[0])
            if n == 0:
                return {}
            placeholders = ",".join("?" * len(lowered))
            rows = self._conn.execute(
                f"SELECT term, doc FROM memories_vocab WHERE term IN ({placeholders})",
                lowered,
            ).fetchall()
            df_by_term = {row["term"]: int(row["doc"]) for row in rows}
        except sqlite3.Error:
            return {}
        stats: dict[str, tuple[int, float]] = {}
        for term in lowered:
            df = df_by_term.get(term, 0)
            stats[term] = (df, math.log(n / (1 + df)))
        return stats

    def search_text(
        self,
        query: str,
        active_only: bool = True,
        limit: int | None = None,
        include_fading: bool = True,
        *,
        bump: bool | float = True,
    ) -> list[Memory]:
        """Case-insensitive substring search on content.

        `%` and `_` in `query` are escaped so a caller passing `"%"` does
        not match every row. Empty queries are rejected; use list_active()
        when the caller intentionally wants a bounded/all-memory scan.

        include_fading: when True (default), faded memories appear in
        results with their summary as content and state='fading'.
        When False, only active memories are returned. Set False on
        callers that need pre-Forgetting search semantics.

        bump: when True (default), matched rows have recall_count (+1.0) and
        last_accessed_at updated — the correct behaviour for a genuine
        recall. False skips the bump — the consolidation gate's internal
        dedup/context scans pass ``bump=False`` so an evaluation read does
        not inflate committed-row salience (which would perturb the
        forgetting pass). A float bumps recall_count by that amount instead
        of 1.0.
        """
        if query == "":
            raise ValueError("empty query passed to search_text; use list_active() instead")
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql = "SELECT * FROM memories WHERE content LIKE ? ESCAPE '\\' COLLATE NOCASE"
        params: list[Any] = [f"%{escaped}%"]
        if active_only:
            sql += " AND active = 1"
        if not include_fading:
            sql += " AND state != 'fading'"
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        amount = _bump_amount(bump)
        if rows and amount is not None:
            now_iso = datetime.now(UTC).isoformat()
            ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"UPDATE memories SET recall_count = recall_count + ?, last_accessed_at = ? "
                f"WHERE id IN ({placeholders})",
                (amount, now_iso, *ids),
            )
            self._conn.commit()
        return [_row_to_memory(row) for row in rows]

    def search_fts_scored(
        self,
        query: str,
        *,
        active_only: bool = True,
        include_fading: bool = True,
        limit: int | None = None,
        bump: bool | float = False,
    ) -> list[tuple[Memory, float]]:
        """FTS5/BM25 text-match search, best-match first.

        Runs the sanitized query (``_to_fts_match``) as an FTS5 ``MATCH``,
        joins back to ``memories``, and returns each Memory **with its raw
        bm25 score** ordered by ``bm25(memories_fts)`` ascending (lower is a
        better match). The ranker (:mod:`brain.memory.relevance`) inverts and
        normalizes the score; it is surfaced here because the ranker needs it.

        Applies the same ``active``/``state`` filters ``search_text`` does, so
        an ``active_only`` caller gets no fading rows. ``bump`` defaults
        **False** (surfacing must not inflate ``recall_count``); when True the
        matched rows' ``recall_count`` (+1.0) / ``last_accessed_at`` are
        bumped; a float bumps ``recall_count`` by that amount instead.

        An empty/all-tokens-dropped query returns ``[]`` (no MATCH is issued).
        """
        match = _to_fts_match(query)
        if not match:
            return []
        sql = (
            "SELECT m.*, bm25(memories_fts) AS _bm25 "
            "FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
            "WHERE memories_fts MATCH ?"
        )
        params: list[Any] = [match]
        if active_only:
            sql += " AND m.active = 1"
        if not include_fading:
            sql += " AND m.state != 'fading'"
        sql += " ORDER BY _bm25 ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        amount = _bump_amount(bump)
        if rows and amount is not None:
            now_iso = datetime.now(UTC).isoformat()
            ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"UPDATE memories SET recall_count = recall_count + ?, "
                f"last_accessed_at = ? WHERE id IN ({placeholders})",
                (amount, now_iso, *ids),
            )
            self._conn.commit()
        return [(_row_to_memory(row), float(row["_bm25"])) for row in rows]

    def list_active(self, limit: int | None = None) -> list[Memory]:
        """Return active memories ordered by created_at desc."""
        sql = "SELECT * FROM memories WHERE active = 1 ORDER BY created_at DESC"
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_memory(row) for row in rows]

    def exists_recent_grief_touch(self, referent_id: str, *, hours: float) -> bool:
        """Return True if a grief_event memory with grief_referent_id == referent_id
        exists in the memories table created within the last `hours`.

        Implementation: SQL filter on (memory_type='grief_event', created_at >= cutoff),
        then Python-side filter on metadata.grief_referent_id. Recent grief volume is
        sparse — linear filter cost is negligible. Avoids depending on SQLite
        json_extract. Spec §4 + §6.
        """
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        cursor = self._conn.execute(
            "SELECT metadata_json FROM memories "
            "WHERE memory_type = 'grief_event' AND created_at >= ?",
            (cutoff,),
        )
        for row in cursor:
            meta = _safe_load_metadata(row["metadata_json"])
            if meta.get("grief_referent_id") == referent_id:
                return True
        return False

    def list_since_iso(self, opened_at_iso: str, *, include_fading: bool = True) -> list[Memory]:
        """Return memories with created_at > opened_at_iso, ordered ascending.

        Used by narrative_memory ArcUpdatePass to draw the candidate pool —
        memories born after the arc opened (or after the last pass ts).

        Includes ``state='fading'`` rows by default — their content is a
        deterministic summary, still eligible for arc membership. Lost
        rows are gone (hard-deleted) so they never appear here.
        """
        sql = "SELECT * FROM memories WHERE created_at > ?"
        if not include_fading:
            sql += " AND state = 'active'"
        sql += " ORDER BY created_at ASC"
        rows = self._conn.execute(sql, (opened_at_iso,)).fetchall()
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
    row_keys = row.keys()
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
        state=row["state"] if "state" in row_keys else "active",
        content_snapshot=row["content_snapshot"] if "content_snapshot" in row_keys else None,
        recall_count=row["recall_count"] if "recall_count" in row_keys else 0,
        peak_emotion_intensity=(
            float(row["peak_emotion_intensity"])
            if "peak_emotion_intensity" in row_keys
            else 0.0
        ),
    )
