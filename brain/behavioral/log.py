"""Behavioral log — append-only JSONL of creative_dna and journal lifecycle changes.

Per spec §3.4: focused biographical record of CHANGES only (not all behavior).
Read by chat composition (recent growth block) and by the creative_dna
crystallizer (avoid reproposing recently-dropped names).

Pure narrative substrate: nothing in the framework decides anything based on
this log except the brain itself, via the chat system message. Writes are
atomic single-line JSONL appends; reads skip corrupt lines via the existing
brain.health.jsonl_reader helper. No schema migration needed for v1
(retention unbounded; ~50KB/year worst case).

OG reference: NellBrain/data/behavioral_log.jsonl (different scope — OG
logged every daemon fire and conversation; v1 narrows to lifecycle changes
of creative_dna + journal).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.utils.time import iso_utc

_VALID_KINDS = frozenset(
    {
        "creative_dna_active_added",
        "creative_dna_emerging_added",
        "creative_dna_emerging_promoted",
        "creative_dna_active_demoted",
        "creative_dna_fading_dropped",
        "journal_entry_added",
        # body lifecycle — emitted by brain/body/events.py when an add_memory
        # commit lands with climax >= 7. Same wire shape as journal_entry_added
        # (uses source/reflex_arc_name/emotional_state slots).
        "climax_event",
    }
)


def append_behavioral_event(
    path: Path,
    *,
    kind: str,
    name: str,
    timestamp: datetime,
    # creative_dna lifecycle fields:
    reasoning: str | None = None,
    evidence_memory_ids: Iterable[str] = (),
    # journal_entry_added fields:
    source: str | None = None,
    reflex_arc_name: str | None = None,
    emotional_state: dict[str, float] | None = None,
) -> None:
    """Append one behavioral event as a single JSON line.

    Atomic per JSONL line write semantics. Caller passes a tz-aware UTC
    `timestamp`; the function serialises via `iso_utc`.

    Raises:
        ValueError: if `kind` is not one of the 6 valid kinds.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"behavioral_log: unknown kind {kind!r}")

    if kind in {"journal_entry_added", "climax_event"}:
        entry: dict[str, Any] = {
            "timestamp": iso_utc(timestamp),
            "kind": kind,
            "name": name,
            "source": source,
            "reflex_arc_name": reflex_arc_name,
            "emotional_state": dict(emotional_state or {}),
        }
    else:
        entry = {
            "timestamp": iso_utc(timestamp),
            "kind": kind,
            "name": name,
            "reasoning": reasoning or "",
            "evidence_memory_ids": list(evidence_memory_ids),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_behavioral_log(
    path: Path,
    *,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    """Read all entries (or those at-or-after `since`) oldest-first.

    Corrupt lines are skipped silently via read_jsonl_skipping_corrupt.
    Missing file returns empty list.
    """
    if not path.exists():
        return []
    entries = list(read_jsonl_skipping_corrupt(path))
    if since is None:
        return entries
    cutoff_iso = iso_utc(since)
    return [e for e in entries if e.get("timestamp", "") >= cutoff_iso]
