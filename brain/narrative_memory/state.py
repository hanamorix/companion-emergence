"""ArcsState persistence + JSONL lifecycle log.

Two files in <persona_dir>:
  - arcs_state.json  — current open + recently_closed (snapshot)
  - arcs.log.jsonl   — append-only lifecycle events (source of truth on recovery)

Recovery model (spec §3): if state.json is corrupt or staler than the
newest log event, replay arcs.log.jsonl from beginning to rebuild state.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.health.attempt_heal import save_with_backup
from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt
from brain.narrative_memory.arc import Arc, ArcMember
from brain.utils.file_lock import file_lock

# State + log filenames inside persona_dir
STATE_FILENAME = "arcs_state.json"
LOG_FILENAME = "arcs.log.jsonl"

# Cap for in-state recently_closed list. Older closed arcs remain reachable
# via JSONL log scan (recall_arc fallback path).
RECENTLY_CLOSED_CAP = 20


@dataclass
class ArcsState:
    """Snapshot of current open arcs + recently-closed arcs.

    Atomic-written to arcs_state.json on every pass. last_pass_ts_iso
    is the pass-completion wall-clock — used to detect state staleness
    vs. JSONL log on recovery.

    replayed is True iff this state was rebuilt from JSONL log on
    bridge startup (rather than loaded as-is). Cleared on next pass.
    """

    open: dict[str, Arc] = field(default_factory=dict)
    recently_closed: list[Arc] = field(default_factory=list)
    last_pass_ts_iso: str | None = None
    replayed: bool = False


def save_state(persona_dir: Path, state: ArcsState) -> None:
    """Atomic JSON save of ArcsState via save_with_backup.

    Arc + ArcMember dataclasses serialise via _arc_to_dict; tuples become
    lists in JSON, deserialised back to tuples on load.
    """
    payload = {
        "open": {arc_id: _arc_to_dict(arc) for arc_id, arc in state.open.items()},
        "recently_closed": [_arc_to_dict(arc) for arc in state.recently_closed],
        "last_pass_ts_iso": state.last_pass_ts_iso,
    }
    save_with_backup(persona_dir / STATE_FILENAME, payload)


def _arc_to_dict(arc: Arc) -> dict[str, Any]:
    return {
        "id": arc.id,
        "state": arc.state,
        "seed_anchor_type": arc.seed_anchor_type,
        "seed_anchor_ref": arc.seed_anchor_ref,
        "seed_memory_ids": list(arc.seed_memory_ids),
        "title": arc.title,
        "opened_at_iso": arc.opened_at_iso,
        "lived_age_at_open": arc.lived_age_at_open,
        "last_extended_at_iso": arc.last_extended_at_iso,
        "closed_at_iso": arc.closed_at_iso,
        "lived_age_at_close": arc.lived_age_at_close,
        "members": [
            {
                "memory_id": m.memory_id,
                "joined_at_iso": m.joined_at_iso,
                "lived_age_at_join": m.lived_age_at_join,
                "salience_at_join": m.salience_at_join,
            }
            for m in arc.members
        ],
    }


def _arc_from_dict(d: dict[str, Any]) -> Arc:
    return Arc(
        id=d["id"],
        state=d["state"],
        seed_anchor_type=d["seed_anchor_type"],
        seed_anchor_ref=d["seed_anchor_ref"],
        seed_memory_ids=tuple(d["seed_memory_ids"]),
        title=d["title"],
        opened_at_iso=d["opened_at_iso"],
        lived_age_at_open=d["lived_age_at_open"],
        last_extended_at_iso=d["last_extended_at_iso"],
        closed_at_iso=d.get("closed_at_iso"),
        lived_age_at_close=d.get("lived_age_at_close"),
        members=tuple(
            ArcMember(
                memory_id=m["memory_id"],
                joined_at_iso=m["joined_at_iso"],
                lived_age_at_join=m["lived_age_at_join"],
                salience_at_join=m["salience_at_join"],
            )
            for m in d.get("members", [])
        ),
    )


def append_event(persona_dir: Path, event: dict[str, Any]) -> None:
    """Append one lifecycle event to arcs.log.jsonl with file_lock + fsync."""
    raise NotImplementedError


def load_or_recover(persona_dir: Path) -> ArcsState:
    """Load arcs_state.json, falling back to JSONL log replay on miss/corrupt.

    Returns fresh empty ArcsState if both are missing.
    """
    raise NotImplementedError
