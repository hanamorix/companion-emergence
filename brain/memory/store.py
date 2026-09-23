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

import numpy as np

from brain import tunables
from brain.memory.floor_calibration import RETENTION_WINDOW_DAYS_DEFAULT

logger = logging.getLogger(__name__)

# F2a (#250 inc5, FINALIZED inc7, SHRUNK pre-flip revision Change 1):
# calibration_log retention window (spec Section 5 / acceptance 5b). inc5
# shipped this as a PROVISIONAL flat 14.0 placeholder; inc7 replaced it with
# a derived max(sample-drawable-days, drift-responsiveness) formula sized to
# cover the (then-pooled) multi-day fit. Change 1 re-points the fit to read
# only the MOST RECENTLY COMPLETED DAY (`MemoryStore.labeled_calibration_
# pairs`, day-scoped), so retention no longer needs to cover a multi-day
# accumulation window — only today's in-progress bucket, yesterday's (the
# day actually read), and a small safety buffer. See
# `floor_calibration.derive_retention_window_days` for the current
# derivation, imported here as `RETENTION_WINDOW_DAYS_DEFAULT`. The
# derivation's one home stays in floor_calibration.py; this module keeps
# owning the tunable KEY (`calibration.retention_window_days`) and `prune_
# calibration_log`'s contract, unchanged from inc5. An operator override via
# tunables.json is unaffected by this swap from a flat default to a derived
# one.
CALIBRATION_LOG_RETENTION_WINDOW_DAYS: float = tunables.register(
    "calibration.retention_window_days", RETENTION_WINDOW_DAYS_DEFAULT
)

# F2b (#276 §5): the score scale every `calibration_log` row `log_
# calibration_sample` writes FROM HERE ON is stamped with — a single named
# constant (never a bare string literal at either the write site or the
# fit-sample filter site) so the two stay in lockstep by construction. Once
# F2b ships, `reranker_scores` is always the per-query anchor-normalized
# value (`brain.memory.reranker.normalize_against_anchors`), never the raw
# cross-encoder score — see the `calibration_log` schema comment above for
# why the two scales must never be mixed into one floor fit.
CALIBRATION_SCORE_SCALE = "normalized"


def _coerce_utc(ts: str) -> datetime:
    """Parse ISO-8601 timestamp; coerce tz-naive values to UTC.

    Shared by Memory.from_dict and MemoryStore._row_to_memory so both paths
    from storage apply identical naive→UTC handling — no drift risk when a
    third reader (e.g. migrator) is added.
    """
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def clamp_importance(x: float) -> float:
    """Clamp a raw importance value into the store's [0.0, 10.0] scale.

    Shared boundary guard (P3 retention rework, Change 1): every producer's
    importance, however derived, passes through here before it can reach a
    Memory row. Applied inside Memory.create_new (covers an unclamped
    emotion-sum default, e.g. engines/dream.py's saturated aggregate) AND
    inside migrator/transform.py's bespoke Memory(...) construction (which
    bypasses create_new). One helper, both boundaries: no producer can ever
    emit importance > 10.0, and a legitimate value already in range is
    never altered.
    """
    return min(10.0, max(0.0, float(x)))


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
        raw_importance = importance if importance is not None else score / 10.0
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            domain=domain,
            created_at=datetime.now(UTC),
            emotions=emotions,
            tags=tags,
            importance=clamp_importance(raw_importance),
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
    peak_emotion_intensity REAL NOT NULL DEFAULT 0.0,
    embedding BLOB,
    embedding_model_id TEXT,
    cluster_id INTEGER,
    cluster_model_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

-- F1 (#259): per-cluster centroid vectors, relocated into memories.db so
-- every per-persona vector artifact lives in the one DB (I1) instead of the
-- old content-hash-keyed side file (embeddings.db's MemoryClusterStore).
-- Keyed by (model_id, cluster_id) — centroids are per-cluster, not
-- per-memory, so they have no row on `memories` to live on. Shape mirrors
-- `MemoryClusterStore.memory_cluster_centroids`
-- (brain/memory/clustering.py) column-for-column so the later read/write
-- swap-over is a straight port, not a reshape.
CREATE TABLE IF NOT EXISTS cluster_centroids (
    model_id TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    centroid BLOB NOT NULL,
    dim INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, cluster_id)
);

-- F2a (#250 inc4): real-query calibration log — one row per recall turn,
-- logging the REAL (never synthetic) query, the retrieved candidate ids, and
-- the reranker scores ALREADY computed that turn, so a later idle-gated
-- daily tick (spec Section 5/7, not yet built) can judge-label a SAMPLE of
-- these rows and re-derive RERANK_FLOOR against the actual corpus and the
-- actual reranker model in use. Lives in memories.db per I1 — same posture
-- as `cluster_centroids` above (a per-persona artifact, not per-memory; no
-- separate .db file). `reranker_model_id` is logged per row so a later floor
-- derivation can filter to one model's score scale and never mixes scores
-- across a reranker swap (F2a's own jina swap, or any future one).
-- `day_bucket` mirrors `logged_at`'s date so the daily tick can select one
-- day's rows cheaply without parsing timestamps. `local_judge_label` /
-- `haiku_label` are nullable and start empty on day one — populated by the
-- offline judge pass (Section 6, not yet built) and read by F2c's later
-- fine-tuning (Out-of-scope) — logged from day one per spec so F2c has data
-- to start from once it exists.
--
-- Retention/pruning (F2a #250 inc5, spec Section 5 / acceptance 5b): the
-- daily calibration tick (`_run_calibration_tick` in brain/bridge/
-- supervisor.py) calls `MemoryStore.prune_calibration_log` every firing, so
-- this table stays bounded to a rolling `day_bucket` window instead of
-- growing unbounded — see CALIBRATION_LOG_RETENTION_WINDOW_DAYS above for
-- the window's FINALIZED (inc7) value and derivation.
-- `score_scale` (F2b, #276 §5): marks whether `reranker_scores` on this row
-- is on the PRE-F2b raw reranker-score scale or the POST-F2b per-query
-- anchor-normalized scale (`brain.memory.reranker.normalize_against_
-- anchors`) — the two are not comparable (a constant per-query offset
-- separates them) and must never be mixed into one floor fit. DEFAULT
-- 'raw' so every row written before this column existed (and every legacy
-- row picked up by the migration below) reads as raw without a backfill;
-- `log_calibration_sample` (F2b) always stamps the CURRENT scale
-- (`CALIBRATION_SCORE_SCALE` = 'normalized') explicitly on every row it
-- writes from here on, never relying on the column default. Read by
-- `labeled_calibration_pairs`'s fit-sample filter (F2b §5) so a post-deploy
-- floor fit draws only from normalized-scale rows. This same marker is a
-- candidate for F2c/inc3's deploy-time one-time-recalibration trigger
-- (spec §6) — "has a normalized-scale row ever been logged" is exactly a
-- deploy-detection signal, though wiring that trigger is out of scope here.
-- `local_judge_raw_score` (F2c inc1, data foundation only — spec §3
-- Addition A): the bge judge's RAW per-candidate score/logit, JSON-encoded
-- and positionally aligned with `candidate_ids` (same convention as
-- `reranker_scores`/`local_judge_label`). Today `relevance_judge.
-- label_calibration_sample` computes this score (`judge.score(query,
-- mem.content)`) and discards it after deriving the label — F2c's
-- knob-refit (later increment) needs the raw score itself to fit a
-- threshold/Platt mapping, not just the derived label. Nullable: NULL on
-- every legacy row and on any position this judge pass never scored
-- (`"unknown"`/`"error"` sentinels — see `write_calibration_labels`).
-- `candidate_docs` (F2c inc1, data foundation only — spec §3 Addition B):
-- a JSON list of candidate DOC-TEXT strings, positionally aligned with
-- `candidate_ids`, snapshotted at RECALL time (`log_calibration_sample`'s
-- caller, `semantic_recall.run_semantic_recall`, already holds this text
-- as `real_documents` that turn) rather than re-fetched from `memories` at
-- label time — a memory can be edited/pruned/forgotten in the days between
-- being logged and being labeled, so a label-time re-fetch would drift
-- from what the judge/reranker actually scored. F2c's later LoRA/full-FT
-- tiers need `(query, doc, label)` triples built from this exact snapshot.
-- Nullable: NULL on every legacy row (I9 — legacy import stays working).
CREATE TABLE IF NOT EXISTS calibration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    day_bucket TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d', 'now')),
    query TEXT NOT NULL,
    candidate_ids TEXT NOT NULL,
    reranker_scores TEXT NOT NULL,
    reranker_model_id TEXT NOT NULL,
    local_judge_label TEXT,
    haiku_label TEXT,
    score_scale TEXT NOT NULL DEFAULT 'raw',
    local_judge_raw_score TEXT,
    candidate_docs TEXT
);
CREATE INDEX IF NOT EXISTS idx_calibration_log_day_bucket ON calibration_log(day_bucket);

-- F2a (#250 inc7): the DB-adaptive abstention floor itself (spec Section 7)
-- — the per-persona replacement for the hardcoded
-- `semantic_recall.RERANK_FLOOR` constant. Lives in memories.db per I1, same
-- posture as `cluster_centroids`/`calibration_log` above (a per-persona
-- artifact, never a side file). Keyed by `reranker_model_id`, mirroring
-- `cluster_centroids`' model_id keying, so a reranker precision flip
-- (fp32<->fp16, #250 §2) or any future model_tier swap never mixes a floor
-- fit against one score scale with scores from another — each model_id
-- tracks its own independently-derived, independently-EMA-smoothed floor.
-- `floor` is the value actually in effect (EMA-smoothed once past cold
-- start); `raw_fit_floor` is that cycle's pre-EMA fit, kept for
-- diagnostics/tests. `is_cold_start` distinguishes a bootstrap-pairs fit
-- (not yet enough real labeled data — spec's outcome-based cold-start exit)
-- from a real corpus-derived fit. WRITTEN by
-- `brain.memory.floor_calibration.derive_and_persist_floor` (inc7, called
-- from the daily calibration tick); READ live by `select_standouts`
-- (`brain/memory/semantic_recall.py`) and the reranker precision self-check
-- (`brain/memory/reranker.py`) as of inc8's cutover — the bare
-- `RERANK_FLOOR` constant this table replaces no longer exists. A model_id
-- with NO row here yet is a fresh-install/early-days state: `MemoryStore.
-- get_reranker_floor` serves a derived, transient BOOTSTRAP floor instead of
-- None in that case (F2a inc8, #250 §7 UPDATED, Roy 2026-09-18) — never
-- written to this table; the first accepted daily-tick write always
-- supersedes it.
-- `score_scale` (F2b, #276 §6): mirrors `calibration_log.score_scale`
-- (same rationale, same DEFAULT 'raw' — see that column's comment above)
-- but marks the SCALE OF THE PERSISTED FLOOR ITSELF rather than a logged
-- score row. A pre-F2b row (written before this column existed) reads as
-- 'raw' via the column default without a backfill; every write FROM HERE
-- ON (`MemoryStore.write_reranker_floor`, called only by
-- `floor_calibration.derive_and_persist_floor`) stamps the CURRENT scale
-- (`CALIBRATION_SCORE_SCALE` = 'normalized', post-§5b true for BOTH the
-- real-fit and cold-start branches) explicitly, never relying on the
-- default. Read by `MemoryStore.reranker_floor_is_stale` — the deploy-time
-- one-time-recalibration trigger (spec §6, #276 inc3,
-- `brain.bridge.supervisor._run_deploy_recalibration_check`): an ABSENT
-- row or a row still reading 'raw' means the persisted floor predates
-- F2b's normalized-scale gate and needs one out-of-cycle recalibration
-- pass, rather than waiting for the next scheduled daily tick.
CREATE TABLE IF NOT EXISTS reranker_floor_calibration (
    reranker_model_id TEXT PRIMARY KEY,
    floor REAL NOT NULL,
    raw_fit_floor REAL NOT NULL,
    sample_pairs INTEGER NOT NULL,
    is_cold_start INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    score_scale TEXT NOT NULL DEFAULT 'raw'
);

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
        # F1 (#259): embedding/cluster columns. Nullable, no default — existing
        # rows land NULL and are picked up by the embedding backfill (a later
        # increment); this migration step only guarantees the columns exist.
        if "embedding" not in existing:
            self._conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
        if "embedding_model_id" not in existing:
            self._conn.execute("ALTER TABLE memories ADD COLUMN embedding_model_id TEXT")
        if "cluster_id" not in existing:
            self._conn.execute("ALTER TABLE memories ADD COLUMN cluster_id INTEGER")
        if "cluster_model_id" not in existing:
            self._conn.execute("ALTER TABLE memories ADD COLUMN cluster_model_id TEXT")
        # F2b (#276 §5): score_scale migration for a `calibration_log` table
        # that pre-dates this column (I9 — legacy DBs keep working; never
        # drop/rewrite). Same idempotent existing-columns-check pattern as
        # the `memories` migrations above, scoped to `calibration_log`.
        # DEFAULT 'raw' matches the CREATE TABLE default: every row that
        # existed before this migration ran was logged on the pre-F2b raw
        # scale, so backfilling them as 'raw' (rather than leaving them
        # NULL) is the honest label, not a guess.
        existing_calibration_log = {
            row[1] for row in self._conn.execute("PRAGMA table_info(calibration_log)").fetchall()
        }
        if "score_scale" not in existing_calibration_log:
            self._conn.execute(
                "ALTER TABLE calibration_log ADD COLUMN score_scale TEXT NOT NULL DEFAULT 'raw'"
            )
        # F2c inc1 (data foundation only, spec §3): same idempotent
        # existing-columns-check pattern, scoped to `calibration_log`'s two
        # new additive columns. Both are nullable with NO default (unlike
        # `score_scale`'s 'raw' default) — there is no honest backfill
        # value for a pre-F2c row's raw judge score or doc-text snapshot
        # (unlike `score_scale`, where "every prior row was on the raw
        # scale" is a true fact); NULL correctly means "not captured for
        # this legacy row" (I9 — legacy rows keep reading, just without
        # this data).
        if "local_judge_raw_score" not in existing_calibration_log:
            self._conn.execute(
                "ALTER TABLE calibration_log ADD COLUMN local_judge_raw_score TEXT"
            )
        if "candidate_docs" not in existing_calibration_log:
            self._conn.execute("ALTER TABLE calibration_log ADD COLUMN candidate_docs TEXT")
        # F2b (#276 §6): same migration shape, scoped to
        # `reranker_floor_calibration` — a legacy DB's persisted floor row
        # predates the scale marker and must read as 'raw' (never a guess)
        # so `reranker_floor_is_stale` correctly flags it for the deploy-time
        # one-time recalibration (§6) rather than silently trusting a
        # raw-scale floor under the normalized gate.
        existing_floor_calibration = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(reranker_floor_calibration)"
            ).fetchall()
        }
        if "score_scale" not in existing_floor_calibration:
            self._conn.execute(
                "ALTER TABLE reranker_floor_calibration ADD COLUMN score_scale "
                "TEXT NOT NULL DEFAULT 'raw'"
            )
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
        """Insert a memory. Returns the id. Raises on duplicate id.

        Deliberately does NOT itself touch embeddings — MemoryStore has no
        embeddings dependency baked into `create()` (~11 call sites use it
        directly: brain/tools/impls/add_memory.py, crystallize_soul.py,
        brain/memory/pending.py, etc.). Embedding a memory on write,
        everywhere, would mean either threading an embeddings dependency
        through every one of those call sites, or this method spawning
        off-thread work itself (its own new sqlite connection per call, on a
        hot synchronous path some of those sites sit on) — both more
        invasive and riskier than the alternative that was chosen instead:
        the idle-chipped embedding backfill (brain/memory/embedding_backfill.py),
        which scans `memories` every supervisor tick and catches whatever a
        `create()` call didn't embed, regardless of which call site wrote it.
        New memories are lexically recallable immediately either way; an
        un-embedded one becomes semantically recallable within one backfill
        tick at the latest.

        ONE caller is the deliberate exception: `brain.engines.consolidation.
        _dispatch`'s promote branch calls this method, then immediately
        calls `embed_row(cand.id, cand.content)` on the freshly-inserted id
        (F1 #259 step 4, embed-on-write at pending-queue -> committed-memory
        promotion) — the steady-state path a promoted memory takes, so it is
        recallable semantically on the very next turn rather than waiting
        for backfill. This method itself stays embedding-agnostic; the two
        calls are simply sequenced at that one call site. See that module's
        docstring and hunts/semantic-retrieval/plan.md Part A #7 for the
        full backfill reasoning (Stage 2 of the local semantic-retrieval
        build).
        """
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

    def embed_row(self, memory_id: str, content: str) -> None:
        """Compute + persist the retrieval embedding for one existing row
        (F1 #259 step 4: embed-on-write).

        Embeds `content` via the process-cached production provider
        (`brain.memory.embeddings.build_embedding_provider()`, looked up
        through the module so a test's monkeypatch on the module attribute
        is honored, mirroring every other dynamic-lookup call site in this
        codebase), writes
        `embedding` + `embedding_model_id` straight onto the row (no
        content-hash side table involved), and pushes the vector into this
        db's warm `EmbeddingMatrix` (`brain.memory.embedding_matrix.
        build_embedding_matrix(self.db_path)`) so it is immediately
        recall-visible in this process without waiting for a lazy rebuild.

        Raises on any failure computing/persisting the embedding itself
        (provider/model error, sqlite error) — deliberately NOT fail-soft
        there. The caller decides how to handle a failed embed; e.g.
        `brain.engines.consolidation._dispatch` wraps its call in a local
        try/except and leaves the row's embedding NULL for the later idle
        backfill to pick up, rather than aborting the whole drain tick over
        one bad embed. `MemoryStore._reembed_or_clear` below is the other
        caller, with its own failure handling (clear + evict).

        The warm-matrix `put` AFTER the row commit is different: by that
        point the row is already durably embedded (the DB is the source of
        truth; the matrix is a cache that self-heals on rebuild — see
        `build_embedding_matrix`), so a `put` failure must NOT be reported
        as an embed failure (F1 #259 increment-3 red-team fix, F3) — it is
        logged and swallowed, never raised out of this method, so a caller
        counting embed successes/failures (the backfill's `errors` field)
        doesn't misclassify an already-committed row as failed.

        Computes, then delegates the persist step to `write_embedding` (F1
        #259 F3) — see that method for the row-UPDATE + matrix-`put` details.
        `write_embedding` exists as its own method so a caller that has
        ALREADY computed a vector elsewhere (consolidation's Pass-2 judge,
        F3 below) can persist it directly without a second, redundant
        provider call.
        """
        from brain.memory import embeddings as embeddings_mod

        provider = embeddings_mod.build_embedding_provider()
        vec = provider.embed(content).astype("float32")
        self.write_embedding(memory_id, vec, provider.model_id())

    def write_embedding(self, memory_id: str, vector: np.ndarray, model_id: str) -> None:
        """Persist a PRECOMPUTED embedding vector directly onto a row + the
        warm matrix, without calling the embedding provider (F1 #259 F3:
        consolidation's Pass-2 now embeds a candidate ONCE, before the Haiku
        judge runs — so the vector is available for the cosine
        `_related_existing` retrieval — and reuses that SAME vector here at
        promotion instead of a second, redundant `embed_row` compute; see
        `brain.engines.consolidation._dispatch`'s promote branch).

        Mirrors `embed_row`'s persist step exactly (row UPDATE + matrix
        `put`, with the identical matrix-put-failure-is-swallowed contract
        below) but skips the compute step — `embed_row` itself now computes
        the vector, then calls this method to do the actual persisting, so
        the two never drift apart.
        """
        from brain.memory.embedding_matrix import build_embedding_matrix

        vec = np.asarray(vector, dtype=np.float32)
        self._conn.execute(
            "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
            (vec.tobytes(), model_id, memory_id),
        )
        self._conn.commit()
        try:
            build_embedding_matrix(self.db_path).put(memory_id, vec)
        except Exception:  # noqa: BLE001 — row is already committed-embedded; a
            # matrix-cache write failure must not surface as an embed failure.
            logger.warning(
                "MemoryStore.write_embedding: warm-matrix put failed for id=%s after "
                "the row's embedding was already committed — the matrix "
                "self-heals on rebuild, so this is logged, not raised",
                memory_id,
                exc_info=True,
            )

    def _reembed_or_clear(self, memory_id: str, content: str) -> None:
        """Keep a row's embedding coherent with its content after ANY
        content-mutating path (F1 #259 step 5, flag-2 resolved: SYNCHRONOUS
        re-embed, mirroring `fade()`'s existing synchronous-rewrite
        behavior). Called AFTER the row's content UPDATE has already
        committed, from `fade()`, `update(memory_id, content=...)`, and
        `unfade()`.

        Row-id keying is NOT self-healing the way the old content-hash cache
        was: new content under the SAME id has no `embedding IS NULL` signal
        for the idle backfill to catch, so every content-changing path must
        re-embed synchronously or explicitly clear the now-stale vector —
        never leave a vector on the row that no longer matches its content.

        On success: re-embeds via `embed_row` (row + warm matrix updated
        together). On failure (provider/model error): CLEARS the row's
        `embedding`/`embedding_model_id` columns and evicts the matrix
        entry — a NULL row (picked up by the later idle backfill) is always
        safer than a vector silently describing stale content. Best-effort:
        an error clearing/evicting is logged, never raised — a content
        mutation must not fail because embedding bookkeeping failed.
        """
        try:
            self.embed_row(memory_id, content)
            return
        except Exception:  # noqa: BLE001 — degrade to clear, never raise
            logger.warning(
                "MemoryStore._reembed_or_clear: embed failed for id=%s — "
                "clearing the now-stale embedding instead",
                memory_id,
                exc_info=True,
            )
        try:
            self._conn.execute(
                "UPDATE memories SET embedding = NULL, embedding_model_id = NULL WHERE id = ?",
                (memory_id,),
            )
            self._conn.commit()
            from brain.memory.embedding_matrix import build_embedding_matrix

            build_embedding_matrix(self.db_path).evict(memory_id)
        except Exception:  # noqa: BLE001 — best-effort cleanup, never raise
            logger.warning(
                "MemoryStore._reembed_or_clear: failed to clear/evict stale embedding for id=%s",
                memory_id,
                exc_info=True,
            )

    def set_cluster_memberships(
        self,
        memberships: dict[str, int],
        centroids: np.ndarray,
        *,
        model_id: str,
    ) -> None:
        """Atomically replace `model_id`'s cluster memberships (on the
        `memories` row's `cluster_id`/`cluster_model_id` columns) + its
        centroids (`cluster_centroids` table) with the result of one
        clustering pass (F1 #259 increment 4).

        This is the row/table-based successor to the old
        `MemoryClusterStore.replace_pass` (`brain/memory/clustering.py`) —
        it ports that method's two load-bearing properties onto the new
        storage rather than reinventing them:

        WHOLESALE-REPLACE, scoped to `model_id`: every row currently
        tagged `cluster_model_id = model_id` is cleared FIRST (`cluster_id`
        / `cluster_model_id` -> NULL), then every memory id present in
        `memberships` gets its `cluster_id`/`cluster_model_id` (re)stamped.
        A memory id clustered by a PRIOR pass but absent from THIS pass's
        `memberships` (its embedding was evicted/faded/dropped out of the
        warm-matrix snapshot since) ends up NULL, not a stale `cluster_id`
        pointing at a centroid this pass may have deleted — mirroring
        `replace_pass`'s own delete-then-reinsert symmetry, just keyed by
        memory id instead of content_hash (I2: two byte-identical memories
        no longer share one tag, each gets its own — the intended identity
        change per spec §1).

        ONE transaction, ONE commit: the membership clear, the membership
        (re)stamps, AND the centroid replace below all share one
        uncommitted SQLite transaction, committed once at the end — a
        crash mid-write leaves this at exactly the LAST successfully
        committed pass's state (never a mix of old/new memberships, and
        never memberships without their matching centroids). This is what
        makes `run_clustering_pass` idempotent/resumable, same guarantee
        `replace_pass` provided for the old side table.
        """
        self._conn.execute(
            "UPDATE memories SET cluster_id = NULL, cluster_model_id = NULL "
            "WHERE cluster_model_id = ?",
            (model_id,),
        )
        self._conn.executemany(
            "UPDATE memories SET cluster_id = ?, cluster_model_id = ? WHERE id = ?",
            [(cluster_id, model_id, mem_id) for mem_id, cluster_id in memberships.items()],
        )
        self._conn.execute("DELETE FROM cluster_centroids WHERE model_id = ?", (model_id,))
        self._conn.executemany(
            "INSERT INTO cluster_centroids (model_id, cluster_id, centroid, dim) "
            "VALUES (?, ?, ?, ?)",
            [
                (model_id, i, centroid.astype(np.float32).tobytes(), centroid.shape[0])
                for i, centroid in enumerate(centroids)
            ],
        )
        self._conn.commit()

    def get_cluster_id(self, memory_id: str) -> tuple[int, str] | None:
        """`(cluster_id, cluster_model_id)` for `memory_id`'s row, or
        `None` when the row doesn't exist or hasn't been clustered
        (`cluster_id IS NULL`) — the raw row read `cluster_tag_for_memory`
        (`brain/memory/clustering.py`) layers its model_id-scoping check on
        top of (never bumps recall — this is a metadata read, not a
        surfacing event)."""
        row = self._conn.execute(
            "SELECT cluster_id, cluster_model_id FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None or row["cluster_id"] is None:
            return None
        return int(row["cluster_id"]), row["cluster_model_id"]

    def log_calibration_sample(
        self,
        query: str,
        candidate_ids: list[str],
        reranker_scores: list[float],
        reranker_model_id: str,
        candidate_docs: list[str] | None = None,
    ) -> None:
        """Log one recall turn's (query, candidate ids, reranker scores) row
        to `calibration_log` (F2a #250 inc4).

        `query` must be the literal raw `user_input` string the caller
        embedded/reranked this turn — byte-identical, never a synthesized
        or reconstructed query (spec Section 4 / acceptance #5). `candidate_
        ids` and `reranker_scores` are the ALREADY-COMPUTED per-turn rerank
        output (same order, 1:1) — this method does no scoring of its own.
        `reranker_model_id` is stamped per row so a later floor-derivation
        pass can filter to one reranker's score scale.

        `candidate_docs` (F2c inc1, data foundation only — spec §3 Addition
        B): the RECALL-TIME candidate doc-text snapshot, positionally
        aligned with `candidate_ids` 1:1 — the caller's already-in-hand
        content strings for this turn's candidates (e.g. `semantic_recall.
        run_semantic_recall`'s `real_documents`, sliced the same way
        `candidate_ids` itself is), never a re-fetch from `memories` at some
        later time (a memory can be edited/pruned/forgotten between being
        logged and being labeled/trained on, so a later re-fetch would
        silently drift from what was actually scored this turn). Optional
        and defaults to `None` (stored as SQL NULL) so callers that don't
        have doc text in hand keep working unchanged — this method does no
        fetching of its own.

        F2b (#276 §5): `reranker_scores` must be the caller's already-
        NORMALIZED per-query anchor-corrected value (`brain.memory.
        reranker.normalize_against_anchors`'s output), not the raw
        cross-encoder score — this method stamps every row it writes with
        `score_scale = CALIBRATION_SCORE_SCALE` ('normalized') accordingly.
        This method does no normalization itself; it trusts the caller the
        same way it already trusts `candidate_ids`/`reranker_scores` to be
        the already-computed per-turn output.

        ONE bounded INSERT — no embedding, no model call, off the hot path
        in every sense except this single cheap write (I6). Fail-soft is
        the CALLER's job (`semantic_recall.run_semantic_recall` wraps this
        call in its own try/except so a logging failure here can never
        break recall) — this method itself does not swallow errors, so a
        caller that forgets to guard it fails loudly instead of silently
        losing calibration data.
        """
        self._conn.execute(
            "INSERT INTO calibration_log "
            "(query, candidate_ids, reranker_scores, reranker_model_id, score_scale, "
            "candidate_docs) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                query,
                json.dumps(list(candidate_ids)),
                json.dumps([float(s) for s in reranker_scores]),
                reranker_model_id,
                CALIBRATION_SCORE_SCALE,
                json.dumps(list(candidate_docs)) if candidate_docs is not None else None,
            ),
        )
        self._conn.commit()

    def prune_calibration_log(
        self, *, window_days: float | None = None, now: datetime | None = None
    ) -> int:
        """Delete `calibration_log` rows whose `day_bucket` falls outside the
        rolling retention window (F2a #250 inc5, spec Section 5 / acceptance
        5b), keeping the table bounded instead of growing forever.

        `window_days` defaults to the live `calibration.retention_window_days`
        tunable (see `CALIBRATION_LOG_RETENTION_WINDOW_DAYS` above) when not
        passed explicitly — read at call time via `tunables.get_tunable` so an
        operator override applies with no restart, mirroring
        `brain/memory/reranker.py`'s `LATENCY_BUDGET_SECONDS` pattern. The
        window itself is PROVISIONAL (see the tunable's own comment) — its
        full derivation is finalized in F2a inc7 once the calibration sample
        size exists.

        `day_bucket` is a `YYYY-MM-DD` string (see the CREATE TABLE default
        above), so a lexicographic string comparison against the cutoff date
        is a correct date comparison with no parsing needed.

        Called from the daily calibration tick (`_run_calibration_tick` in
        `brain/bridge/supervisor.py`), off the hot path (I6) — never from a
        per-turn recall path. Returns the number of rows deleted (0 if none
        were due).
        """
        if window_days is None:
            window_days = tunables.get_tunable(
                "calibration.retention_window_days", CALIBRATION_LOG_RETENTION_WINDOW_DAYS
            )
        ref = now if now is not None else datetime.now(UTC)
        cutoff_bucket = (ref - timedelta(days=window_days)).strftime("%Y-%m-%d")
        cur = self._conn.execute(
            "DELETE FROM calibration_log WHERE day_bucket < ?", (cutoff_bucket,)
        )
        self._conn.commit()
        return cur.rowcount

    def sample_unlabeled_calibration_rows(self, limit: int) -> list[dict[str, Any]]:
        """Return up to `limit` `calibration_log` rows with no
        `local_judge_label` yet (F2a #250 inc6, spec Section 6/7) — the daily
        judge pass's SAMPLE, not every logged row (the spec's explicit
        "sampling IS the design"). `limit` bounds the local judge's daily
        compute on the no-AVX2 potato baseline — see
        `relevance_judge.CALIBRATION_SAMPLE_ROWS` for that value's
        derivation.

        Randomized via SQL `RANDOM()` rather than oldest/newest-N, so an
        unlabeled backlog doesn't systematically bias the sample toward one
        time-of-day's query mix. `candidate_ids` / `reranker_scores` are
        decoded from their stored JSON here so callers work with plain
        Python lists, not raw JSON strings — mirrors how `get()` decodes
        `metadata_json` before returning a `Memory`.

        Read-only: does not bump `recall_count` (reads `calibration_log`,
        not `memories`) and does not label anything itself — labeling +
        writeback is the caller's job (`relevance_judge.
        label_calibration_sample` + `write_calibration_labels` below).
        """
        rows = self._conn.execute(
            "SELECT id, query, candidate_ids, reranker_scores, reranker_model_id "
            "FROM calibration_log WHERE local_judge_label IS NULL "
            "ORDER BY RANDOM() LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "query": row["query"],
                "candidate_ids": json.loads(row["candidate_ids"]),
                "reranker_scores": json.loads(row["reranker_scores"]),
                "reranker_model_id": row["reranker_model_id"],
            }
            for row in rows
        ]

    def write_calibration_labels(
        self,
        row_id: int,
        local_judge_label: list[str],
        haiku_label: list[str | None],
        local_judge_raw_score: list[float | None] | None = None,
    ) -> None:
        """Write back the local judge's + Haiku tie-break's per-candidate
        labels for one `calibration_log` row (F2a #250 inc6, spec Section 6).

        Both lists are POSITIONALLY aligned with that row's `candidate_ids`
        (same convention `candidate_ids`/`reranker_scores` already use),
        stored as JSON in their respective TEXT columns. `haiku_label`
        entries are `None` except at the ambiguous-band positions the local
        judge routed to Haiku (non-ambiguous positions never call Haiku, per
        acceptance #7) — a `None` means "no override; the local judge's own
        provisional label at that position stands."

        `local_judge_raw_score` (F2c inc1, data foundation only — spec §3
        Addition A): the bge judge's RAW per-candidate score/logit, also
        POSITIONALLY aligned with `candidate_ids`, stored as JSON in
        `local_judge_raw_score`. A `None` entry means this position was
        never scored (the `"unknown"`/`"error"` label sentinels — a deleted
        candidate or a judge failure on that candidate, see
        `relevance_judge.label_calibration_sample`), distinct from a real
        score of 0.0. Optional and defaults to `None` (the whole column
        stays NULL for this row) so a caller that doesn't have raw scores
        in hand — e.g. any test exercising only the label-writing path —
        keeps working unchanged.

        Once `local_judge_label` is non-NULL the row no longer matches
        `sample_unlabeled_calibration_rows`'s `WHERE` clause, so a row is
        never re-sampled or re-labeled on a later tick.
        """
        self._conn.execute(
            "UPDATE calibration_log SET local_judge_label = ?, haiku_label = ?, "
            "local_judge_raw_score = ? WHERE id = ?",
            (
                json.dumps(local_judge_label),
                json.dumps(haiku_label),
                json.dumps(local_judge_raw_score) if local_judge_raw_score is not None else None,
                row_id,
            ),
        )
        self._conn.commit()

    def labeled_calibration_pairs(self, reranker_model_id: str) -> list[tuple[float, str]]:
        """`(reranker_score, effective_label)` pairs for the MOST RECENTLY
        COMPLETED DAY's LABELED `calibration_log` rows matching
        `reranker_model_id` (F2a #250 inc7, spec Section 7; pre-flip
        revision Change 1, "nimble floor") — the floor-derivation fit's
        ONLY caller/input (`floor_calibration.derive_and_persist_floor`).

        Pre-flip revision Change 1 DAY-SCOPES this read: before Change 1,
        this method drew from EVERY currently-retained labeled row for this
        model_id (the full multi-day retention window, pooled). That pooled
        read let one severely-stale day keep outvoting a genuine, fast
        corpus shift for as long as the retention window stayed wide — the
        confirmed root cause of a permanent floor lock (spec's T7 Part A).
        Change 1's fix: scope to the MOST RECENTLY COMPLETED DAY only —
        this method's own `day_bucket` boundary (Change 1's Open
        Reconfirmation "the single-day fit window's exact bound"), resolved
        here as: the MAX `day_bucket` among this model_id's own USABLE
        labeled rows (same WHERE clause as the pair read itself, below) —
        reusing the EXISTING `day_bucket` column (the same one `prune_
        calibration_log`'s retention window already keys on) rather than
        introducing a second, rolling-24h time-anchoring scheme. This
        naturally tracks whichever day a caller's own seeded/logged data
        actually lands on, with no dependency on wall-clock `now` at call
        time — a day is "the most recently completed one" once no later
        day's rows have been judge-labeled yet for this model_id.

        Each row's `candidate_ids` / `reranker_scores` / `local_judge_label`
        / `haiku_label` are all POSITIONALLY aligned (the convention every
        calibration_log writer/reader in this module already follows); this
        method zips them per-row and, per candidate position, takes the
        EFFECTIVE label as `haiku_label[i]` when non-null (the Haiku
        tie-break OVERRIDES the local judge at ambiguous-band positions),
        else `local_judge_label[i]` (spec Section 6/7's stated precedence).
        Positions labeled `"unknown"` (candidate no longer exists) or
        `"error"` (judge failure on that candidate) are SKIPPED — neither
        is a usable relevant/irrelevant ground-truth label for a threshold
        fit.

        F2b (#276 §5): additionally filters to `score_scale =
        CALIBRATION_SCORE_SCALE` ('normalized') — this is the floor-fit's
        ONLY sampler (see `floor_calibration.py`), so this is the one place
        the raw/normalized scale split actually matters. A pre-F2b row
        logged on the raw scale is excluded outright, never mixed into a
        fit trained against normalized-scale scores (mixing scales would
        silently corrupt the derived floor — the two are not comparable, a
        constant per-query offset separates them). The MAX-`day_bucket`
        lookup below is scoped by this same score_scale filter, so a
        raw-scale row can never be picked as "the most recent day" either.

        Read-only: does not bump `recall_count` and does not label or
        write anything (mirrors `sample_unlabeled_calibration_rows`'s own
        read-only posture).
        """
        max_day_row = self._conn.execute(
            "SELECT MAX(day_bucket) AS max_day FROM calibration_log "
            "WHERE reranker_model_id = ? AND local_judge_label IS NOT NULL AND score_scale = ?",
            (reranker_model_id, CALIBRATION_SCORE_SCALE),
        ).fetchone()
        most_recent_day = max_day_row["max_day"] if max_day_row is not None else None
        if most_recent_day is None:
            return []
        rows = self._conn.execute(
            "SELECT reranker_scores, local_judge_label, haiku_label FROM calibration_log "
            "WHERE reranker_model_id = ? AND local_judge_label IS NOT NULL "
            "AND score_scale = ? AND day_bucket = ?",
            (reranker_model_id, CALIBRATION_SCORE_SCALE, most_recent_day),
        ).fetchall()
        pairs: list[tuple[float, str]] = []
        for row in rows:
            scores = json.loads(row["reranker_scores"])
            local_labels = json.loads(row["local_judge_label"])
            haiku_labels = (
                json.loads(row["haiku_label"]) if row["haiku_label"] is not None else [None] * len(scores)
            )
            for score, local_label, haiku_label in zip(scores, local_labels, haiku_labels, strict=False):
                effective = haiku_label if haiku_label is not None else local_label
                if effective in ("relevant", "irrelevant"):
                    pairs.append((float(score), effective))
        return pairs

    def get_persisted_reranker_floor(self, reranker_model_id: str) -> dict[str, Any] | None:
        """Return ONLY the PERSISTED `reranker_floor_calibration` row for
        `reranker_model_id`, or `None` if none exists — never the transient
        bootstrap fallback `get_reranker_floor` serves on a miss (F2a
        inc8). Read-only: does not write or bump anything.

        Pre-flip revision Change 1: this is the persisted-only half
        `get_reranker_floor` below was refactored to share with
        `floor_calibration.derive_and_persist_floor`'s own data-starvation
        backstop, which needs to tell "a real prior row exists to hold"
        apart from "no row exists, and the caller would otherwise be
        looking at the transient bootstrap" — `get_reranker_floor` itself
        conflates the two (by design, for its OWN callers, which want
        SOME floor, persisted or not); the backstop must not.
        """
        row = self._conn.execute(
            "SELECT reranker_model_id, floor, raw_fit_floor, sample_pairs, is_cold_start, "
            "updated_at, score_scale "
            "FROM reranker_floor_calibration WHERE reranker_model_id = ?",
            (reranker_model_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "reranker_model_id": row["reranker_model_id"],
            "floor": float(row["floor"]),
            "raw_fit_floor": float(row["raw_fit_floor"]),
            "sample_pairs": int(row["sample_pairs"]),
            "is_cold_start": bool(row["is_cold_start"]),
            "updated_at": row["updated_at"],
            "score_scale": row["score_scale"],
        }

    def get_reranker_floor(self, reranker_model_id: str) -> dict[str, Any] | None:
        """Return the operative reranker floor for `reranker_model_id`
        (F2a #250 inc7/inc8, spec Section 7).

        Checks the PERSISTED `reranker_floor_calibration` row first (written
        by the daily calibration tick, `floor_calibration.derive_and_
        persist_floor`, via `get_persisted_reranker_floor` above) — if one
        exists, it is returned and this method does no further work.

        If no row exists yet (fresh install / early days / a brand-new
        reranker model_id that has never been calibrated — including, as of
        the pre-flip revision's Change 1, a deploy still inside the
        data-starvation backstop's ramp, since that backstop's no-prior-row
        edge case intentionally writes nothing), F2a inc8 (#250 §7 UPDATED,
        Roy 2026-09-18's bootstrap-floor ruling) serves a derived,
        process-wide-cached BOOTSTRAP floor instead of `None` —
        `floor_calibration.get_bootstrap_floor`, computed once (jina-only,
        torch-free) from the bundled `_FP16_GATE_PAIRS` and cached, never
        persisted to this table. This decouples semantic recall's EXISTENCE
        from the daily tick ever having fired: the old "no row -> None ->
        every caller falls back to lexical" contract permanently coupled
        recall to the tick (disable calibration, or recall running before
        the tick's first idle moment, silently and permanently demoted to
        lexical-only even with embeddings present).

        A persisted row, once the tick writes one, is read FIRST on every
        subsequent call and supersedes the bootstrap for good — the
        bootstrap cache is never consulted again for that model_id, so a
        stale bootstrap value can never shadow a real corpus-derived floor.

        Only returns `None` now on the bootstrap's OWN fail-soft path (the
        bootstrap computation itself raised — a reranker load/fit failure)
        — the pre-ruling contract, preserved as the last resort so a broken
        bootstrap still degrades this turn to lexical rather than crashing.

        Read-only: does not write or bump anything.
        """
        persisted = self.get_persisted_reranker_floor(reranker_model_id)
        if persisted is not None:
            return persisted
        from brain.memory.floor_calibration import get_bootstrap_floor

        return get_bootstrap_floor(reranker_model_id)

    def reranker_floor_is_stale(self, reranker_model_id: str) -> bool:
        """True iff `reranker_model_id`'s PERSISTED `reranker_floor_
        calibration` row is either ABSENT or still on the pre-F2b RAW score
        scale (F2b #276 §6, inc3's deploy-time one-time-recalibration
        trigger).

        Reads the persisted row DIRECTLY (not through `get_reranker_floor`,
        which serves a transient, never-persisted bootstrap floor on a miss
        — that in-memory fallback is irrelevant here: this check exists
        purely to decide whether the ON-DISK row needs one out-of-cycle
        `floor_calibration.derive_and_persist_floor` pass, so an ABSENT row
        must read as stale exactly like a present-but-'raw' one, not be
        masked by the bootstrap's existence).

        Called ONLY by the bridge-startup deploy-recalibration check
        (`brain.bridge.supervisor._run_deploy_recalibration_check`) — never
        by the per-recall floor gate itself, which always goes through
        `get_reranker_floor` (persisted-or-bootstrap) unconditionally and
        does not care about staleness on a per-turn basis.

        A model_id that has never had ANY row written (fresh install, or a
        newly-registered reranker model_id) reads as stale too — the
        out-of-cycle pass then runs F2a's own cold-start path (now
        normalized-scale per §5b), landing a scale-correct floor
        immediately instead of waiting on the bootstrap's transient,
        never-persisted default.

        Read-only: does not write or bump anything.
        """
        row = self._conn.execute(
            "SELECT score_scale FROM reranker_floor_calibration WHERE reranker_model_id = ?",
            (reranker_model_id,),
        ).fetchone()
        if row is None:
            return True
        return row["score_scale"] != CALIBRATION_SCORE_SCALE

    def write_reranker_floor(
        self,
        reranker_model_id: str,
        *,
        floor: float,
        raw_fit_floor: float,
        sample_pairs: int,
        is_cold_start: bool,
        score_scale: str = CALIBRATION_SCORE_SCALE,
    ) -> None:
        """Upsert this cycle's derived floor for `reranker_model_id` (F2a
        #250 inc7, spec Section 7) — the ONLY write path into
        `reranker_floor_calibration` (I1: a table in memories.db, never a
        side file). `INSERT ... ON CONFLICT DO UPDATE` keyed on
        `reranker_model_id` (its PRIMARY KEY): a model_id's row is replaced
        wholesale each accepted cycle, never accumulated — this table
        tracks the CURRENT floor per model_id, not a history of past ones.
        Pre-flip revision Change 1 removed the EMA smoothing this docstring
        used to cite here as the reason no history is kept — post-Change-1
        there is no smoothing left to carry history forward AT ALL: `floor`
        is simply THIS cycle's raw fit, and the prior value is either held
        untouched (never reaching this method — see below) or fully
        replaced, never blended.

        Called only when a cycle's derivation is ACCEPTED (the most
        recently completed day's raw fit, per Change 1 — no more cold-start
        branch) — a held/rejected cycle (`FloorDerivationOutcome.accepted
        is False`, Change 1's data-starvation backstop) must NOT call this,
        leaving the previously persisted row (or its absence) untouched.

        `score_scale` (F2b #276 §6): defaults to the CURRENT scale
        (`CALIBRATION_SCORE_SCALE` = 'normalized') — post-§5b, the only
        remaining caller (`floor_calibration.derive_and_persist_floor`'s
        real-fit branch; Change 1 removed its cold-start branch, which used
        to be this method's other caller) always scores/fits on the
        anchor-normalized scale, so it never needs to override this. The
        parameter exists (rather than a bare hardcoded value in the SQL) so
        a test can exercise a legacy 'raw' row without reaching around this
        method's public contract.
        """
        self._conn.execute(
            "INSERT INTO reranker_floor_calibration "
            "(reranker_model_id, floor, raw_fit_floor, sample_pairs, is_cold_start, "
            "updated_at, score_scale) "
            "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?) "
            "ON CONFLICT(reranker_model_id) DO UPDATE SET "
            "floor = excluded.floor, raw_fit_floor = excluded.raw_fit_floor, "
            "sample_pairs = excluded.sample_pairs, is_cold_start = excluded.is_cold_start, "
            "updated_at = excluded.updated_at, score_scale = excluded.score_scale",
            (
                reranker_model_id,
                float(floor),
                float(raw_fit_floor),
                int(sample_pairs),
                int(is_cold_start),
                str(score_scale),
            ),
        )
        self._conn.commit()

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

        # F1 #259 increment-2 red-team fix (crash-window data integrity): NULL
        # the embedding/embedding_model_id columns IN THE SAME UPDATE/commit
        # as the content change, not in the separate re-embed commit below.
        # A crash between "content committed" and "re-embed committed" used
        # to leave the row durably {new content, OLD embedding} — a stale
        # vector with no `embedding IS NULL` signal for the idle backfill to
        # catch, so recall could keep surfacing the memory on its
        # pre-mutation content indefinitely. Folding the NULL into this same
        # UPDATE makes the durable intermediate state {new content, NULL
        # embedding}: backfill-eligible and crash-safe.
        if "content" in fields:
            column_map["embedding"] = ("embedding", None)
            column_map["embedding_model_id"] = ("embedding_model_id", None)

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

        # F1 #259 step 5 (increment-2 red-team fix): a content mutation
        # invalidates the row's embedding — row-id keying has no
        # `embedding IS NULL` signal for the backfill to catch new content
        # under the SAME id, so re-embed synchronously (or clear on
        # failure) right here, only when `content` was actually one of the
        # updated fields.
        if "content" in fields:
            # Evict the in-memory matrix entry at the same point the row's
            # embedding went NULL (above), so a concurrent recall in the
            # brief re-embed window below degrades to lexical rather than
            # serving the stale in-memory vector.
            try:
                from brain.memory.embedding_matrix import build_embedding_matrix

                build_embedding_matrix(self.db_path).evict(memory_id)
            except Exception:  # noqa: BLE001 — best-effort, must not block the write
                logger.warning(
                    "MemoryStore.update: matrix evict failed for id=%s", memory_id, exc_info=True
                )
            # Happy-path synchronous re-embed repopulates embedding + matrix;
            # on failure the row is already NULL (self-healing) — the
            # clear-on-failure branch in `_reembed_or_clear` is now
            # belt-and-suspenders, since the row is already NULL going in.
            self._reembed_or_clear(memory_id, fields["content"])

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
        # F1 #259 increment-2 red-team fix: embedding/embedding_model_id are
        # NULLed IN THE SAME UPDATE/commit as the content change, so the
        # durable intermediate state after a crash right here is {summary
        # content, NULL embedding} — backfill-eligible, never a stale vector
        # against the pre-fade content. See `update()`'s matching comment.
        self._conn.execute(
            "UPDATE memories SET content_snapshot = content, content = ?, "
            "state = 'fading', embedding = NULL, embedding_model_id = NULL "
            "WHERE id = ?",
            (summary, memory_id),
        )
        self._conn.commit()
        # Evict the in-memory matrix entry at the same point, so a
        # concurrent recall in the brief re-embed window below degrades to
        # lexical rather than serving the stale in-memory vector.
        try:
            from brain.memory.embedding_matrix import build_embedding_matrix

            build_embedding_matrix(self.db_path).evict(memory_id)
        except Exception:  # noqa: BLE001 — best-effort, must not block fade()
            logger.warning(
                "MemoryStore.fade: matrix evict failed for id=%s", memory_id, exc_info=True
            )
        # F1 #259 step 5: content changed (full body -> summary) — re-embed
        # synchronously so the row's vector reflects the faded summary, not
        # the pre-fade content (or clear on failure; see
        # `_reembed_or_clear`'s docstring). On failure the row is already
        # NULL (self-healing) — that clear-on-failure branch is now
        # belt-and-suspenders.
        self._reembed_or_clear(memory_id, summary)

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
        restored_content = row["content_snapshot"]
        # F1 #259 increment-2 red-team fix: embedding/embedding_model_id are
        # NULLed IN THE SAME UPDATE/commit as the content change, so the
        # durable intermediate state after a crash right here is {restored
        # content, NULL embedding} — backfill-eligible, never a stale vector
        # against the pre-unfade (summary) content. See `update()`'s
        # matching comment.
        self._conn.execute(
            "UPDATE memories SET content = content_snapshot, content_snapshot = NULL, "
            "state = 'active', embedding = NULL, embedding_model_id = NULL "
            "WHERE id = ?",
            (memory_id,),
        )
        self._conn.commit()
        # Evict the in-memory matrix entry at the same point, so a
        # concurrent recall in the brief re-embed window below degrades to
        # lexical rather than serving the stale in-memory vector.
        try:
            from brain.memory.embedding_matrix import build_embedding_matrix

            build_embedding_matrix(self.db_path).evict(memory_id)
        except Exception:  # noqa: BLE001 — best-effort, must not block unfade()
            logger.warning(
                "MemoryStore.unfade: matrix evict failed for id=%s", memory_id, exc_info=True
            )
        # F1 #259 step 5: content changed back (summary -> restored full
        # body) — re-embed synchronously so the row's vector reflects the
        # restored content (or clear on failure; see `_reembed_or_clear`'s
        # docstring). On failure the row is already NULL (self-healing) —
        # that clear-on-failure branch is now belt-and-suspenders.
        self._reembed_or_clear(memory_id, restored_content)

    def hard_delete(self, memory_id: str) -> None:
        """Drop the row. Caller MUST write the graveyard entry first.
        Raises KeyError if unknown.
        """
        if not self._conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone():
            raise KeyError(f"Unknown memory id: {memory_id!r}")
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        # F1 #259 step 5: the row (and its embedding) is gone — evict the
        # warm-matrix entry too, best-effort, so a deleted memory's vector
        # cannot linger in the matrix (the resurrection hazard `EmbeddingMatrix`
        # itself guards against on the BUILD side; this is the delete-side
        # half of "never orphan a vector on hard_delete", I2/spec §2).
        try:
            from brain.memory.embedding_matrix import build_embedding_matrix

            build_embedding_matrix(self.db_path).evict(memory_id)
        except Exception:  # noqa: BLE001 — best-effort, hard_delete must not fail over this
            logger.warning(
                "MemoryStore.hard_delete: matrix evict failed for id=%s", memory_id, exc_info=True
            )

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

    def list_active_since(
        self, cursor: tuple[str, str] | None, *, limit: int
    ) -> list[Memory]:
        """Return up to `limit` active memories strictly after `cursor`,
        ordered ASCENDING by (created_at, id) (oldest-of-the-remainder
        first).

        `cursor` is `(created_at_iso, id)` of the last row already
        processed, or `None` to start from the beginning of history. Uses a
        composite keyset comparison — `created_at > ts OR (created_at = ts
        AND id > id)` — rather than filtering on `created_at` alone, so that
        multiple rows sharing an IDENTICAL `created_at` (common after a bulk
        migrator import — see `brain/migrator/emergence_kit.py` — since
        `brain/migrator/transform.py` derives `created_at` from the source
        export's own timestamp string) are still individually reachable: a
        strict `created_at > ts` filter would make every row but the last
        one at a shared timestamp permanently invisible to a caller that
        pins its cursor at that timestamp (the embedding backfill did
        exactly this — see brain/memory/embedding_backfill.py). `id` is a
        UUID4 string (TEXT PRIMARY KEY); the comparison is a plain
        lexicographic tiebreak, not creation order — it only needs to be a
        stable TOTAL order so no row at a shared timestamp is skipped or
        revisited, not a meaningful one. A bounded, cursor-paged sibling of
        `list_active()` for a caller that walks the WHOLE corpus a little at
        a time across many calls rather than loading it all at once — same
        `active = 1` filter as `list_active()`. See `list_unembedded_since`
        below for the sibling scoped to the embedding backlog predicate (the
        embedding backfill's own backlog query, F1 #259 increment 3).
        """
        if cursor is None:
            sql = (
                "SELECT * FROM memories WHERE active = 1 "
                "ORDER BY created_at ASC, id ASC LIMIT ?"
            )
            params: list[Any] = [limit]
        else:
            cursor_ts, cursor_id = cursor
            sql = (
                "SELECT * FROM memories WHERE active = 1 AND "
                "(created_at > ? OR (created_at = ? AND id > ?)) "
                "ORDER BY created_at ASC, id ASC LIMIT ?"
            )
            params = [cursor_ts, cursor_ts, cursor_id, limit]
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_memory(row) for row in rows]

    def list_unembedded_since(
        self,
        cursor: tuple[str, str] | None,
        *,
        limit: int,
        current_model_id: str,
        min_chars: int,
    ) -> list[Memory]:
        """Return up to `limit` ACTIVE memories that are backlog for the
        embedding backfill, strictly after `cursor`, ordered ASCENDING by
        (created_at, id) — the embedding backfill's own backlog query (F1
        #259 increment 3; model-mismatch clause + SQL-level short-row
        exclusion folded in on Fixing's inc3 spec-gap, Planning-ruled
        2026-09-16).

        Backlog membership is `embedding IS NULL OR embedding_model_id !=
        current_model_id` — determined directly off the row's OWN columns,
        not a content-hash side-table lookup: unlike the old `embeddings.db`
        cache, a row's presence here is purely a function of its current
        `embedding`/`embedding_model_id` values, so a row that gets (re-)
        embedded under the current model (by this backfill, by embed-on-
        write at promotion, or by a synchronous re-embed in
        `fade`/`update`/`unfade`) simply stops appearing in this query's
        results — no separate bookkeeping needed.

        The `embedding_model_id != current_model_id` half restores the
        model-scoped self-healing the old content-hash cache had: after a
        `MODEL_EMBEDDING` swap, old-model rows stay non-NULL (so an
        IS-NULL-only backlog would never re-embed them) but the warm matrix
        filters them out by model_id, so without this clause they'd fall to
        lexical recall forever with no re-embed path. It's a no-op in
        steady state (every row already matches `current_model_id`) and
        only fires on a swap.

        `min_chars` excludes rows too short to ever be embed-worthy
        (`MIN_CHARS_TO_EMBED` — the caller's constant, passed through rather
        than duplicated here) directly in SQL rather than filtering them out
        Python-side after the fact: this is what keeps the cursor-reset
        no-op cheap in steady state (see `run_embedding_backfill_tick`) —
        once every EMBEDDABLE row is embedded, this query returns empty
        without ever re-scanning the permanently-short rows on every tick.
        `length()` on a SQLite TEXT column counts characters, matching
        Python's `len()`; `content` is `NOT NULL` per schema so no NULL
        handling is needed here.

        Same composite keyset cursor semantics as `list_active_since` (see
        that method's docstring for the tied-created_at rationale) — this is
        its sibling scoped to the backlog predicate above. The caller
        (embedding backfill) decides whether to persist a forward cursor
        after a tick, and — as of F1 #259 increment 7 — that decision is
        `hit_batch_limit`-aware, not simply "got back fewer than `limit`
        rows": a tick that stops early because it hit its own `batch_size`
        cap (rows still remain in the fetched window) DOES persist a forward
        cursor even though it saw fewer than `limit` candidates, so a
        long-lived persona's no-sleep drain keeps making forward progress
        instead of re-scanning full history every tick. A cursor is only
        reset to `None` when a tick both (a) did NOT stop early on its batch
        cap and (b) fetched fewer than `limit` rows — i.e. it genuinely
        EXHAUSTED the whole currently-backlogged set in one scan. See
        `run_embedding_backfill_tick`'s closing comment in
        `brain/memory/embedding_backfill.py` for the exact condition.
        Backlog membership is not append-only per id regardless — a row can
        re-enter it a SECOND time (a later content edit whose synchronous
        re-embed attempt fails — see `_reembed_or_clear` — or a model swap)
        at a `created_at` position the cursor may have already passed — so a
        persisted forward position is only ever a safe scan-cost
        optimization, never a correctness mechanism (a missing/stale cursor
        just means more re-scanning, never a skipped row).
        """
        if cursor is None:
            sql = (
                "SELECT * FROM memories WHERE active = 1 "
                "AND (embedding IS NULL OR embedding_model_id != ?) "
                "AND length(content) >= ? "
                "ORDER BY created_at ASC, id ASC LIMIT ?"
            )
            params: list[Any] = [current_model_id, min_chars, limit]
        else:
            cursor_ts, cursor_id = cursor
            sql = (
                "SELECT * FROM memories WHERE active = 1 "
                "AND (embedding IS NULL OR embedding_model_id != ?) "
                "AND length(content) >= ? AND "
                "(created_at > ? OR (created_at = ? AND id > ?)) "
                "ORDER BY created_at ASC, id ASC LIMIT ?"
            )
            params = [current_model_id, min_chars, cursor_ts, cursor_ts, cursor_id, limit]
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_memory(row) for row in rows]

    def count_unembedded(self, *, current_model_id: str, min_chars: int) -> int:
        """Return a plain, cursor-free COUNT of `list_unembedded_since`'s own
        backlog predicate (F1 #259 increment 6): active, long-enough rows
        with `embedding IS NULL OR embedding_model_id != current_model_id`.

        Unlike `list_unembedded_since`, this ignores any persisted backfill
        cursor entirely — it is a full-table count from the top, used by the
        one-go backfill CLI (`nell embed backfill`) to size its progress
        ticker's denominator before the drain starts, and by
        `run_embedding_backfill_to_completion` to report the TRUE remaining
        backlog after the drain stops (a scattered permanently-failing row
        can sit BEHIND a persisted forward cursor and so never reappear in a
        cursor-bounded `list_unembedded_since` call again within the same
        run, even though it is still un-embedded — this method still counts
        it, because it does not consult the cursor at all).
        """
        sql = (
            "SELECT COUNT(*) FROM memories WHERE active = 1 "
            "AND (embedding IS NULL OR embedding_model_id != ?) "
            "AND length(content) >= ?"
        )
        return int(self._conn.execute(sql, (current_model_id, min_chars)).fetchone()[0])

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
