"""Arc lifecycle storage — graveyard + snapshot.

Two persistent files per persona:

  removed_arcs.jsonl   — append-only graveyard, one JSON object per line.
                         Captures full arc state at removal so data is
                         recoverable even if reflex_arcs.json is nuked.

  .last_arc_snapshot.json — single JSON file with the post-tick arc set,
                            read at start of each tick to detect user
                            file-edits via diff. Atomic write via
                            json_io.write_json.

The graveyard is the source of truth for "did Hana remove this in the last
15 days?" — gate 3 in §6 of the spec consults `recently_removed_names`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from brain.engines.reflex import ReflexArc
from brain.health.adaptive import compute_treatment
from brain.health.json_io import Recovery, append_jsonl, read_json_result, read_jsonl, write_json
from brain.utils.time import iso_utc

logger = logging.getLogger(__name__)

GRAVEYARD_FILENAME = "removed_arcs.jsonl"
SNAPSHOT_FILENAME = ".last_arc_snapshot.json"


def append_removed_arc(
    persona_dir: Path,
    *,
    arc: ReflexArc,
    removed_at: datetime,
    removed_by: str,  # "user_edit" | "brain_self_prune"
    reasoning: str | None,
) -> None:
    """Atomic append to removed_arcs.jsonl."""
    if removed_by not in ("user_edit", "brain_self_prune"):
        raise ValueError(
            f"removed_by must be 'user_edit' or 'brain_self_prune', got {removed_by!r}"
        )
    path = persona_dir / GRAVEYARD_FILENAME
    entry = {
        "name": arc.name,
        "removed_at": iso_utc(removed_at),
        "removed_by": removed_by,
        "reasoning": reasoning,
        "trigger_snapshot": dict(arc.trigger),
        "description_snapshot": arc.description,
        "prompt_template_snapshot": arc.prompt_template,
    }
    # Audit 2026-05-07 P2-1: append via OS append-mode semantics
    # rather than read-rewrite-replace. Concurrent removals (e.g.
    # supervisor heartbeat pruning + user-edit detection) used to
    # race on the .new temp path and clobber entries.
    # Audit 2026-05-08 cross-platform: wrap with file_lock for
    # Windows correctness — POSIX has atomic O_APPEND but Windows
    # doesn't, so concurrent appenders interleave mid-line there.
    append_jsonl(path, entry, fsync=True, nonfinite="null")


def read_removed_arcs(persona_dir: Path) -> list[dict]:
    """Read all graveyard entries oldest-first. Skips corrupt lines."""
    # RAISE: this is the grace gate against re-creating a removed arc; an unreadable file must not read as "none".
    return read_jsonl(persona_dir / GRAVEYARD_FILENAME, recover=Recovery.RAISE)


def recently_removed_names(
    persona_dir: Path,
    *,
    now: datetime,
    grace_days: float,
) -> set[str]:
    """Return names removed within the grace window. Spec gate 3 uses this."""
    cutoff = now - timedelta(days=grace_days)
    names: set[str] = set()
    for entry in read_removed_arcs(persona_dir):
        ts_raw = entry.get("removed_at")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        if ts >= cutoff:
            name = entry.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def write_arc_snapshot(
    persona_dir: Path,
    *,
    arcs: list[ReflexArc],
    snapshot_at: datetime,
) -> None:
    """Atomic write of .last_arc_snapshot.json via json_io.write_json."""
    path = persona_dir / SNAPSHOT_FILENAME
    payload = {
        "version": 1,
        "snapshot_at": iso_utc(snapshot_at),
        "arcs": [arc.to_dict() for arc in arcs],
    }
    treatment = compute_treatment(persona_dir, SNAPSHOT_FILENAME)
    write_json(path, payload, backups=treatment.backup_count)


def read_arc_snapshot(persona_dir: Path) -> list[ReflexArc] | None:
    """Read .last_arc_snapshot.json. Returns None if missing or empty."""
    path = persona_dir / SNAPSHOT_FILENAME
    if not path.exists():
        return None

    def _default() -> dict:
        return {"version": 1, "snapshot_at": "", "arcs": []}

    _res = read_json_result(path, default=_default, recover=Recovery.HEAL)
    data, anomaly = _res.data, _res.anomaly
    if anomaly is not None:
        logger.warning(
            "arc snapshot at %s anomaly %s (action=%s)",
            path,
            anomaly.kind,
            anomaly.action,
        )
    arcs_raw = data.get("arcs", [])
    if not arcs_raw:
        return None
    arcs: list[ReflexArc] = []
    for arc_data in arcs_raw:
        try:
            arcs.append(ReflexArc.from_dict(arc_data))
        except (KeyError, ValueError) as exc:
            logger.warning("skipping snapshot arc with schema error: %s", exc)
    return arcs
