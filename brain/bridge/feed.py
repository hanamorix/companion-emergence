"""Visible inner life feed — merges five source streams into a typed
journal-voice list for the FeedPanel frontend.

The feed is read-only against existing substrate: dream + research
content come from MemoryStore; soul crystallizations from
soul_audit.jsonl; delivered outreach + voice-edit proposals from
initiate_audit.jsonl. Each entry is wrapped with a type-specific
opener phrase ("I dreamed", "I've been researching", etc.) so the
frontend renders journal prose rather than categorized logs.

No new LLM calls. No new schema. The merge is fault-isolated per
source — any one source failing leaves the others usable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


FeedEntryType = Literal["dream", "research", "soul", "outreach", "voice_edit"]


TYPE_OPENER: dict[FeedEntryType, str] = {
    "dream": "I dreamed",
    "research": "I've been researching",
    "soul": "I noticed",
    "outreach": "I reached out",
    "voice_edit": "I wanted to change",
}


@dataclass(frozen=True)
class FeedEntry:
    """One journal entry in the visible inner life feed."""

    type: FeedEntryType
    ts: str  # ISO8601 with timezone
    opener: str  # journal-voice phrase from TYPE_OPENER
    body: str  # existing summary content (Nell's voice)
    audit_id: str | None = None  # cross-reference (outreach + voice_edit only)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "ts": self.ts,
            "opener": self.opener,
            "body": self.body,
            "audit_id": self.audit_id,
        }


def build_dream_entries(persona_dir: Path, *, limit: int) -> list[FeedEntry]:
    """Read up to `limit` dream memories from MemoryStore, newest first."""
    from brain.memory.store import MemoryStore

    db_path = persona_dir / "memories.db"
    if not db_path.exists():
        return []

    try:
        store = MemoryStore(db_path, integrity_check=False)
    except Exception:
        logger.exception("feed: opening MemoryStore for dream source failed")
        return []

    try:
        try:
            mems = store.list_by_type("dream", active_only=True, limit=limit)
        except Exception:
            logger.exception("feed: MemoryStore.list_by_type('dream') failed")
            return []
    finally:
        store.close()

    return [
        FeedEntry(
            type="dream",
            ts=mem.created_at.isoformat(),
            opener=TYPE_OPENER["dream"],
            body=mem.content,
            audit_id=None,
        )
        for mem in mems
        if mem.content
    ]


def build_research_entries(persona_dir: Path, *, limit: int) -> list[FeedEntry]:
    """Read up to `limit` research memories from MemoryStore, newest first."""
    from brain.memory.store import MemoryStore

    db_path = persona_dir / "memories.db"
    if not db_path.exists():
        return []

    try:
        store = MemoryStore(db_path, integrity_check=False)
    except Exception:
        logger.exception("feed: opening MemoryStore for research source failed")
        return []

    try:
        try:
            mems = store.list_by_type("research", active_only=True, limit=limit)
        except Exception:
            logger.exception("feed: MemoryStore.list_by_type('research') failed")
            return []
    finally:
        store.close()

    return [
        FeedEntry(
            type="research",
            ts=mem.created_at.isoformat(),
            opener=TYPE_OPENER["research"],
            body=mem.content,
            audit_id=None,
        )
        for mem in mems
        if mem.content
    ]


def build_soul_entries(persona_dir: Path, *, limit: int) -> list[FeedEntry]:
    """Read soul crystallizations from soul_audit.jsonl, newest first.

    Filters to entries where `crystallization_id` is non-null — those
    are the marked moments where a candidate became permanent soul.
    Defer / reject / parse_error rows are operational noise.
    """
    from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

    audit_path = persona_dir / "soul_audit.jsonl"
    if not audit_path.exists():
        return []

    try:
        rows = read_jsonl_skipping_corrupt(audit_path)
    except Exception:
        logger.exception("feed: read_jsonl_skipping_corrupt for soul_audit failed")
        return []

    crystallized = [r for r in rows if r.get("crystallization_id")]
    crystallized.sort(key=lambda r: r.get("ts") or "", reverse=True)
    crystallized = crystallized[:limit]

    out: list[FeedEntry] = []
    for r in crystallized:
        body = r.get("candidate_text") or r.get("why_it_matters")
        ts = r.get("ts")
        if body and ts:
            out.append(
                FeedEntry(
                    type="soul",
                    ts=ts,
                    opener=TYPE_OPENER["soul"],
                    body=body,
                    audit_id=None,
                )
            )
    return out


def _read_initiate_audit(persona_dir: Path) -> list[dict]:
    """Shared helper — load initiate_audit.jsonl, fault-isolated."""
    from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

    audit_path = persona_dir / "initiate_audit.jsonl"
    if not audit_path.exists():
        return []
    try:
        return read_jsonl_skipping_corrupt(audit_path)
    except Exception:
        logger.exception("feed: read_jsonl_skipping_corrupt for initiate_audit failed")
        return []


def _is_delivered(row: dict) -> bool:
    """A row counts as 'delivered' iff decision was send_* AND delivery says so."""
    if row.get("decision") not in ("send_notify", "send_quiet"):
        return False
    delivery = row.get("delivery") or {}
    return delivery.get("current_state") == "delivered"


def build_outreach_entries(persona_dir: Path, *, limit: int) -> list[FeedEntry]:
    """Delivered, message-kind outbound initiations from initiate_audit.jsonl."""
    rows = _read_initiate_audit(persona_dir)
    keep = [r for r in rows if r.get("kind") == "message" and _is_delivered(r)]
    keep.sort(key=lambda r: r.get("ts") or "", reverse=True)
    keep = keep[:limit]

    out: list[FeedEntry] = []
    for r in keep:
        body = r.get("tone_rendered")
        ts = r.get("ts")
        if body and ts:
            out.append(
                FeedEntry(
                    type="outreach",
                    ts=ts,
                    opener=TYPE_OPENER["outreach"],
                    body=body,
                    audit_id=r.get("audit_id"),
                )
            )
    return out


def build_voice_edit_entries(persona_dir: Path, *, limit: int) -> list[FeedEntry]:
    """Delivered voice-edit proposals from initiate_audit.jsonl."""
    rows = _read_initiate_audit(persona_dir)
    keep = [r for r in rows if r.get("kind") == "voice_edit_proposal" and _is_delivered(r)]
    keep.sort(key=lambda r: r.get("ts") or "", reverse=True)
    keep = keep[:limit]

    out: list[FeedEntry] = []
    for r in keep:
        body = r.get("tone_rendered")
        ts = r.get("ts")
        if body and ts:
            out.append(
                FeedEntry(
                    type="voice_edit",
                    ts=ts,
                    opener=TYPE_OPENER["voice_edit"],
                    body=body,
                    audit_id=r.get("audit_id"),
                )
            )
    return out


def build_feed(persona_dir: Path, *, limit: int = 50) -> list[FeedEntry]:
    """Merge all 5 source streams into a single ts-desc feed, capped at limit.

    Fault isolation: each per-source builder is called inside its own
    try/except so a single source's failure logs an exception and returns
    an empty list for that stream, leaving the other four usable. The
    feed always returns SOMETHING (possibly empty) — never raises.
    """
    builders = (
        build_dream_entries,
        build_research_entries,
        build_soul_entries,
        build_outreach_entries,
        build_voice_edit_entries,
    )

    merged: list[FeedEntry] = []
    for builder in builders:
        try:
            merged.extend(builder(persona_dir, limit=limit))
        except Exception:
            logger.exception("feed: source builder %s failed", builder.__name__)

    merged.sort(key=lambda e: e.ts, reverse=True)
    return merged[:limit]
