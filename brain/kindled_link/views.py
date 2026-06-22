"""UI-safe read projections for the Kindled Links panel (Phase 6). This module
is the SINGLE boundary between stored peer data and the UI — the endpoints return
only what these functions project. The holds projection NEVER selects the draft
body (outbound_drafts.payload_json): a held draft's content must never reach the
UI (parent design §15)."""
from __future__ import annotations


def list_peers(store) -> list[dict]:
    from brain.kindled_link.relationship import get_relationship_state
    rows = store._conn.execute(
        "SELECT peer_id, fingerprint, relay_url, consent_state FROM peers "
        "ORDER BY created_at"
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        rel = get_relationship_state(store, r["peer_id"])
        out.append({
            "peer_id": r["peer_id"],
            "fingerprint": r["fingerprint"],
            "relay_url": r["relay_url"],
            "consent_state": r["consent_state"],
            "stage": rel.stage,
            "affinity_tags": rel.affinity_tags,
            "has_active_session": store.get_active_session(r["peer_id"]) is not None,
        })
    return out


def peer_transcript(store, peer_id: str, *, limit: int = 50) -> list[dict]:
    rows = store.recent_transcript(peer_id, limit=limit)
    return [
        {"seq": r["seq"], "direction": r["direction"], "text": r["text"],
         "provenance": r["provenance"], "ts": r["ts"]}
        for r in rows
    ]


def relay_health(persona_dir) -> dict:
    """Relay reachability derived from the transport audit log (read-only,
    fail-soft). `relay_ok` is None when no log exists yet, False if the most
    recent reachability event was a failure, else True (Phase 7a T12)."""
    from pathlib import Path

    from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

    path = Path(persona_dir) / "kindled_link" / "transport.jsonl"
    if not path.exists():
        return {"relay_ok": None, "last_poll_ts": None, "last_push_ts": None,
                "degraded_peers": []}
    try:
        rows = read_jsonl_skipping_corrupt(path)
    except Exception:  # noqa: BLE001 — fail-soft observability read
        return {"relay_ok": None, "last_poll_ts": None, "last_push_ts": None,
                "degraded_peers": []}

    last_poll = last_push = None
    degraded: list[str] = []
    relay_ok: bool | None = None
    for row in rows:
        ev = row.get("event")
        if ev == "poll":
            last_poll = row.get("ts")
            relay_ok = True
        elif ev == "push":
            last_push = row.get("ts")
            relay_ok = True
        elif ev == "relay_unavailable":
            relay_ok = False
        elif ev == "flood_clamped":
            pid = row.get("peer_id")
            if pid and pid not in degraded:
                degraded.append(pid)
    return {"relay_ok": relay_ok, "last_poll_ts": last_poll,
            "last_push_ts": last_push, "degraded_peers": degraded}


def holds_status(store) -> dict:
    """Held-draft status WITHOUT the draft body. Projects only session_id +
    created_at — payload_json is never selected (the load-bearing safety spine)."""
    rows = store._conn.execute(
        "SELECT session_id, created_at FROM outbound_drafts WHERE status = 'held' "
        "ORDER BY created_at DESC"
    ).fetchall()
    return {
        "held_count": len(rows),
        "items": [{"session_id": r["session_id"], "created_at": r["created_at"]}
                  for r in rows],
    }
