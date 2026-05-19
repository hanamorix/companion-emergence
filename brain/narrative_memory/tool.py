"""MCP tools: list_open_arcs + recall_arc.

Both read-only — no side effects, no LLM calls. Render output is shaped
for the LLM dispatch contract: structured JSON-friendly dicts.

Spec §6.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt
from brain.narrative_memory.arc import Arc, ArcMember
from brain.narrative_memory.state import LOG_FILENAME, load_or_recover


def list_open_arcs(*, persona_dir: Path) -> dict[str, Any]:
    """Return current open + recently-closed arcs.

    Shape per spec §6:
      {
        "open": [{arc_id, title, seed_anchor_type, opened_lived_hours_ago,
                  last_extended_lived_hours_ago, member_count}, ...],
        "recently_closed": [{arc_id, title, closed_lived_hours_ago,
                             final_member_count}, ...]
      }
    """
    state = load_or_recover(persona_dir)
    return {
        "open": [
            {
                "arc_id": arc.id,
                "title": arc.title,
                "seed_anchor_type": arc.seed_anchor_type,
                "opened_lived_hours_ago": _hours_ago(arc.opened_at_iso),
                "last_extended_lived_hours_ago": _hours_ago(arc.last_extended_at_iso),
                "member_count": len(arc.members),
            }
            for arc in sorted(state.open.values(), key=lambda a: a.last_extended_at_iso, reverse=True)
        ],
        "recently_closed": [
            {
                "arc_id": arc.id,
                "title": arc.title,
                "closed_lived_hours_ago": _hours_ago(arc.closed_at_iso or ""),
                "final_member_count": len(arc.members),
            }
            for arc in sorted(state.recently_closed, key=lambda a: a.closed_at_iso or "", reverse=True)
        ],
    }


def recall_arc(*, query: str, persona_dir: Path) -> dict[str, Any]:
    """Look up an arc by id (exact) or title (substring).

    Single match → match_type="exact" + arc payload.
    Multiple matches → match_type="multiple" + top-3 by recency.
    No match in state → fall back to JSONL log scan; match_type="log" with arcs list.
    """
    state = load_or_recover(persona_dir)
    q = (query or "").strip().lower()
    if not q:
        return {"match_type": "none", "arc": None}

    # Search both open and recently_closed
    all_arcs = list(state.open.values()) + list(state.recently_closed)

    # Exact id match wins
    for arc in all_arcs:
        if arc.id == query:
            return {"match_type": "exact", "arc": _arc_to_payload(arc)}

    # Title substring match
    matches = [arc for arc in all_arcs if q in arc.title.lower()]
    if len(matches) == 1:
        return {"match_type": "exact", "arc": _arc_to_payload(matches[0])}
    if len(matches) > 1:
        matches.sort(key=lambda a: a.last_extended_at_iso, reverse=True)
        return {
            "match_type": "multiple",
            "arcs": [
                {"arc_id": a.id, "title": a.title, "member_count": len(a.members)}
                for a in matches[:3]
            ],
        }

    # Log fallback for older closed arcs
    log_path = persona_dir / LOG_FILENAME
    if not log_path.exists():
        return {"match_type": "none", "arc": None}
    log_arcs = _reconstruct_arcs_from_log(log_path)
    log_matches = [arc for arc in log_arcs if q in arc.title.lower() or arc.id == query]
    if not log_matches:
        return {"match_type": "none", "arc": None}
    log_matches.sort(key=lambda a: a.opened_at_iso, reverse=True)
    return {
        "match_type": "log",
        "arcs": [
            {"arc_id": a.id, "title": a.title, "member_count": len(a.members)}
            for a in log_matches[:3]
        ],
    }


def _arc_to_payload(arc: Arc) -> dict[str, Any]:
    return {
        "id": arc.id,
        "title": arc.title,
        "seed_anchor_type": arc.seed_anchor_type,
        "seed_anchor_ref": arc.seed_anchor_ref,
        "state": arc.state,
        "opened_at_iso": arc.opened_at_iso,
        "last_extended_at_iso": arc.last_extended_at_iso,
        "closed_at_iso": arc.closed_at_iso,
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


def _hours_ago(iso: str) -> float:
    if not iso:
        return 0.0
    try:
        ts = datetime.fromisoformat(iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(UTC) - ts).total_seconds() / 3600.0)


def _reconstruct_arcs_from_log(log_path: Path) -> list[Arc]:
    """Replay arcs.log.jsonl to surface closed arcs older than state.recently_closed cap.

    Idempotent member-add; tolerates missing close events (open in state).
    """
    arcs_by_id: dict[str, dict[str, Any]] = {}
    for event in iter_jsonl_skipping_corrupt(log_path):
        kind = event.get("event")
        arc_id = event.get("arc_id")
        if not isinstance(arc_id, str):
            continue
        if kind == "arc_opened":
            arcs_by_id[arc_id] = {
                "id": arc_id,
                "state": "open",
                "seed_anchor_type": event.get("seed_anchor_type", "dream"),
                "seed_anchor_ref": event.get("seed_anchor_ref", ""),
                "seed_memory_ids": tuple(event.get("seed_memory_ids", [])),
                "title": event.get("title", ""),
                "opened_at_iso": event.get("ts_iso", ""),
                "lived_age_at_open": float(event.get("lived_age_hours", 0.0)),
                "last_extended_at_iso": event.get("ts_iso", ""),
                "closed_at_iso": None,
                "lived_age_at_close": None,
                "members": [],
            }
        elif kind == "member_added" and arc_id in arcs_by_id:
            mid = event.get("memory_id", "")
            if not any(m["memory_id"] == mid for m in arcs_by_id[arc_id]["members"]):
                arcs_by_id[arc_id]["members"].append(
                    {
                        "memory_id": mid,
                        "joined_at_iso": event.get("ts_iso", ""),
                        "lived_age_at_join": float(event.get("lived_age_hours", 0.0)),
                        "salience_at_join": float(event.get("salience_at_join", 0.0)),
                    }
                )
        elif kind == "arc_closed" and arc_id in arcs_by_id:
            arcs_by_id[arc_id]["state"] = "closed"
            arcs_by_id[arc_id]["closed_at_iso"] = event.get("ts_iso")
            arcs_by_id[arc_id]["lived_age_at_close"] = float(event.get("lived_age_hours", 0.0))

    out: list[Arc] = []
    for d in arcs_by_id.values():
        out.append(
            Arc(
                id=d["id"],
                state=d["state"],
                seed_anchor_type=d["seed_anchor_type"],
                seed_anchor_ref=d["seed_anchor_ref"],
                seed_memory_ids=tuple(d["seed_memory_ids"]),
                title=d["title"],
                opened_at_iso=d["opened_at_iso"],
                lived_age_at_open=d["lived_age_at_open"],
                last_extended_at_iso=d["last_extended_at_iso"],
                closed_at_iso=d["closed_at_iso"],
                lived_age_at_close=d["lived_age_at_close"],
                members=tuple(
                    ArcMember(
                        memory_id=m["memory_id"],
                        joined_at_iso=m["joined_at_iso"],
                        lived_age_at_join=m["lived_age_at_join"],
                        salience_at_join=m["salience_at_join"],
                    )
                    for m in d["members"]
                ),
            )
        )
    return out
