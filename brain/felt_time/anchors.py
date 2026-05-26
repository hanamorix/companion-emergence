"""anchors.py — unified anchor stream from existing JSONL sources.

Spec §2 anchors.py + §4 anchor types: dream, growth, soul, weather_shift.
Pure functions; no side effects.

Line numbers in source_ref are valid-entry-index (1-based count of dicts
yielded by iter_jsonl_skipping_corrupt), not raw file-line numbers.
Corrupt/blank lines are already skipped by the canonical reader, so the
index is contiguous over valid entries. Downstream callers treat
source_ref as an audit-trail string, not a seek offset, so this is fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from brain.felt_time.state import Anchor
from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt


@dataclass(frozen=True)
class _Source:
    """One anchor source. label_key names the field carrying the anchor text;
    ts_key names the timestamp field; events (if set) restricts which entries
    count as anchors via their "event" field."""

    filename: str
    label_key: str
    ts_key: str = "ts"
    events: frozenset[str] | None = None  # None → all entries


# source-type -> _Source
_SOURCES: dict[str, _Source] = {
    "dream": _Source("dreams.log.jsonl", "summary"),
    "growth": _Source("growth.log.jsonl", "title"),
    "soul": _Source("soul.log.jsonl", "moment_label"),
    "weather_shift": _Source("weather_shifts.log.jsonl", "label"),
    "arc": _Source(
        "arcs.log.jsonl",
        "title",
        ts_key="ts_iso",
        events=frozenset({"arc_opened", "arc_closed"}),
    ),
}


def extract_all(persona_dir: Path) -> list[Anchor]:
    """Return every anchor across all sources, sorted by ts ascending."""
    anchors: list[Anchor] = []
    for type_, src in _SOURCES.items():
        path = persona_dir / src.filename
        # enumerate wraps the canonical reader to produce a valid-entry-index
        # for source_ref (1-based, contiguous over non-corrupt dicts).
        for entry_idx, entry in enumerate(iter_jsonl_skipping_corrupt(path), start=1):
            if src.events is not None and entry.get("event") not in src.events:
                continue
            ts = entry.get(src.ts_key)
            label = entry.get(src.label_key)
            if not ts or not label:
                continue
            anchors.append(
                Anchor(
                    type=type_,
                    ts=ts,
                    label=str(label),
                    source_ref=f"{src.filename}:{entry_idx}",
                )
            )
    anchors.sort(key=lambda a: a.ts)
    return anchors


def scan_since(persona_dir: Path, since_ts: str | None) -> list[Anchor]:
    """Return anchors with ts strictly greater than since_ts.

    since_ts=None returns all anchors (supervisor first-tick path).
    """
    all_anchors = extract_all(persona_dir)
    if since_ts is None:
        return all_anchors
    return [a for a in all_anchors if a.ts > since_ts]
