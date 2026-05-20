"""ArcsState persistence + JSONL lifecycle log.

Two files in <persona_dir>:
  - arcs_state.json  — current open + recently_closed (snapshot)
  - arcs.log.jsonl   — append-only lifecycle events (source of truth on recovery)

Recovery model (spec §3): if state.json is corrupt or staler than the
newest log event, replay arcs.log.jsonl from beginning to rebuild state.
"""

from __future__ import annotations

import json
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
        "max_member_emotion_normalised": arc.max_member_emotion_normalised,
        "dominant_non_grief_emotion": (
            list(arc.dominant_non_grief_emotion)
            if arc.dominant_non_grief_emotion is not None
            else None
        ),
    }


def _arc_from_dict(d: dict[str, Any]) -> Arc:
    raw_dom = d.get("dominant_non_grief_emotion")
    dom: tuple[str, float] | None = None
    if raw_dom and len(raw_dom) == 2:
        dom = (str(raw_dom[0]), float(raw_dom[1]))
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
        max_member_emotion_normalised=float(d.get("max_member_emotion_normalised") or 0.0),
        dominant_non_grief_emotion=dom,
    )


def append_event(persona_dir: Path, event: dict[str, Any]) -> None:
    """Append one lifecycle event to arcs.log.jsonl with file_lock + fsync.

    Mirrors the brain.growth.arc_storage.append_removed_arc precedent —
    file_lock for cross-process safety, fsync for durability.
    """
    log_path = persona_dir / LOG_FILENAME
    line = json.dumps(event) + "\n"
    with file_lock(log_path):
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(line)
            fp.flush()
            import os

            os.fsync(fp.fileno())


def load_or_recover(persona_dir: Path) -> ArcsState:
    """Load arcs_state.json, falling back to JSONL log replay on miss/corrupt.

    Per spec §3 recovery model:
      1. If state.json exists AND last_pass_ts_iso is newer than the newest
         log event, load as-is.
      2. Otherwise, replay arcs.log.jsonl from beginning to reconstruct state.
      3. Empty dir → fresh empty ArcsState.
    """
    state_path = persona_dir / STATE_FILENAME
    log_path = persona_dir / LOG_FILENAME

    loaded_state: ArcsState | None = None
    if state_path.exists():
        try:
            raw = json.loads(state_path.read_text())
            loaded_state = ArcsState(
                open={arc_id: _arc_from_dict(d) for arc_id, d in raw.get("open", {}).items()},
                recently_closed=[_arc_from_dict(d) for d in raw.get("recently_closed", [])],
                last_pass_ts_iso=raw.get("last_pass_ts_iso"),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            loaded_state = None

    if loaded_state is not None and not _log_newer_than_state(log_path, loaded_state):
        return loaded_state

    if not log_path.exists():
        # No state + no log → fresh empty state
        return ArcsState()

    return _replay_from_log(log_path)


def _log_newer_than_state(log_path: Path, state: ArcsState) -> bool:
    """True iff the JSONL log's newest event is newer than state.last_pass_ts_iso.

    Conservative: if the state has no last_pass_ts_iso or any event has no
    ts_iso, treat as stale (force replay).
    """
    if not log_path.exists():
        return False
    if state.last_pass_ts_iso is None:
        return True
    state_ts = state.last_pass_ts_iso
    newest_event_ts: str | None = None
    for event in iter_jsonl_skipping_corrupt(log_path):
        ts = event.get("ts_iso")
        if isinstance(ts, str) and (newest_event_ts is None or ts > newest_event_ts):
            newest_event_ts = ts
    if newest_event_ts is None:
        return False
    return newest_event_ts > state_ts


def _replay_from_log(log_path: Path) -> ArcsState:
    """Reconstruct ArcsState by replaying arcs.log.jsonl event-by-event.

    Idempotent across event types. Sets replayed=True so the frontend
    recovery banner can fire on next persona-state read.
    """
    state = ArcsState(replayed=True)

    for event in iter_jsonl_skipping_corrupt(log_path):
        kind = event.get("event")
        arc_id = event.get("arc_id")
        if kind == "arc_opened" and isinstance(arc_id, str):
            state.open[arc_id] = Arc(
                id=arc_id,
                state="open",
                seed_anchor_type=event.get("seed_anchor_type", "dream"),
                seed_anchor_ref=event.get("seed_anchor_ref", ""),
                seed_memory_ids=tuple(event.get("seed_memory_ids", [])),
                title=event.get("title", ""),
                opened_at_iso=event.get("ts_iso", ""),
                lived_age_at_open=float(event.get("lived_age_hours", 0.0)),
                last_extended_at_iso=event.get("ts_iso", ""),
                closed_at_iso=None,
                lived_age_at_close=None,
                members=(),
                max_member_emotion_normalised=0.0,
                dominant_non_grief_emotion=None,
            )
        elif kind == "member_added" and isinstance(arc_id, str) and arc_id in state.open:
            arc = state.open[arc_id]
            member_id = event.get("memory_id", "")
            # Idempotent — skip if already a member.
            if any(m.memory_id == member_id for m in arc.members):
                continue
            new_member = ArcMember(
                memory_id=member_id,
                joined_at_iso=event.get("ts_iso", ""),
                lived_age_at_join=float(event.get("lived_age_hours", 0.0)),
                salience_at_join=float(event.get("salience_at_join", 0.0)),
            )
            state.open[arc_id] = _arc_with_member(arc, new_member, event.get("ts_iso", ""))
        elif kind == "member_evicted" and isinstance(arc_id, str) and arc_id in state.open:
            arc = state.open[arc_id]
            evicted_id = event.get("memory_id", "")
            new_members = tuple(m for m in arc.members if m.memory_id != evicted_id)
            state.open[arc_id] = _arc_replace_members(arc, new_members)
        elif kind == "arc_closed" and isinstance(arc_id, str) and arc_id in state.open:
            arc = state.open.pop(arc_id)
            # max_member_emotion_normalised and dominant_non_grief_emotion are NOT
            # stored in the JSONL arc_closed event — they are populated by run_pass at
            # close time and only live in arcs_state.json. Replayed recently_closed
            # arcs carry the open-arc defaults (0.0 / None); grief breadcrumbs have
            # already been written and are unaffected by replay.
            closed = Arc(
                id=arc.id,
                state="closed",
                seed_anchor_type=arc.seed_anchor_type,
                seed_anchor_ref=arc.seed_anchor_ref,
                seed_memory_ids=arc.seed_memory_ids,
                title=arc.title,
                opened_at_iso=arc.opened_at_iso,
                lived_age_at_open=arc.lived_age_at_open,
                last_extended_at_iso=arc.last_extended_at_iso,
                closed_at_iso=event.get("ts_iso"),
                lived_age_at_close=float(event.get("lived_age_hours", 0.0)),
                members=arc.members,
                max_member_emotion_normalised=arc.max_member_emotion_normalised,
                dominant_non_grief_emotion=arc.dominant_non_grief_emotion,
            )
            state.recently_closed.append(closed)

    # Cap recently_closed at RECENTLY_CLOSED_CAP after replay.
    if len(state.recently_closed) > RECENTLY_CLOSED_CAP:
        state.recently_closed = state.recently_closed[-RECENTLY_CLOSED_CAP:]
    return state


def _arc_with_member(arc: Arc, member: ArcMember, ts_iso: str) -> Arc:
    return Arc(
        id=arc.id,
        state=arc.state,
        seed_anchor_type=arc.seed_anchor_type,
        seed_anchor_ref=arc.seed_anchor_ref,
        seed_memory_ids=arc.seed_memory_ids,
        title=arc.title,
        opened_at_iso=arc.opened_at_iso,
        lived_age_at_open=arc.lived_age_at_open,
        last_extended_at_iso=ts_iso or arc.last_extended_at_iso,
        closed_at_iso=arc.closed_at_iso,
        lived_age_at_close=arc.lived_age_at_close,
        members=arc.members + (member,),
        max_member_emotion_normalised=arc.max_member_emotion_normalised,
        dominant_non_grief_emotion=arc.dominant_non_grief_emotion,
    )


def _arc_replace_members(arc: Arc, members: tuple[ArcMember, ...]) -> Arc:
    return Arc(
        id=arc.id,
        state=arc.state,
        seed_anchor_type=arc.seed_anchor_type,
        seed_anchor_ref=arc.seed_anchor_ref,
        seed_memory_ids=arc.seed_memory_ids,
        title=arc.title,
        opened_at_iso=arc.opened_at_iso,
        lived_age_at_open=arc.lived_age_at_open,
        last_extended_at_iso=arc.last_extended_at_iso,
        closed_at_iso=arc.closed_at_iso,
        lived_age_at_close=arc.lived_age_at_close,
        members=members,
        max_member_emotion_normalised=arc.max_member_emotion_normalised,
        dominant_non_grief_emotion=arc.dominant_non_grief_emotion,
    )
