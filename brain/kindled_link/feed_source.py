"""Relationship-milestone feed entries (parent §14 feeds-into 'Feed'). Stage-up
milestones surface in the inner-life feed; a regression is a quiet local event,
not a celebratory entry."""
from __future__ import annotations

_MILESTONE_STAGES = ("acquaintance", "familiar", "friend", "close")


def relationship_milestone_entries(store, *, limit: int = 20) -> list[dict]:
    rows = store._conn.execute(
        "SELECT peer_id, stage, last_reflected_at FROM relationship_state "
        "WHERE stage != 'stranger' ORDER BY last_reflected_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {"kind": "relationship_milestone", "peer_id": r["peer_id"],
         "stage": r["stage"], "ts": r["last_reflected_at"]}
        for r in rows if r["stage"] in _MILESTONE_STAGES
    ]
