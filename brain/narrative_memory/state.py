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
    """Atomic JSON save of ArcsState via save_with_backup."""
    raise NotImplementedError


def append_event(persona_dir: Path, event: dict[str, Any]) -> None:
    """Append one lifecycle event to arcs.log.jsonl with file_lock + fsync."""
    raise NotImplementedError


def load_or_recover(persona_dir: Path) -> ArcsState:
    """Load arcs_state.json, falling back to JSONL log replay on miss/corrupt.

    Returns fresh empty ArcsState if both are missing.
    """
    raise NotImplementedError
