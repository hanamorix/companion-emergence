"""Relationship-milestone and key-rotation feed entries (parent §14 feeds-into
'Feed'). Stage-up milestones surface in the inner-life feed."""
from __future__ import annotations

from pathlib import Path

_MILESTONE_STAGES = ("acquaintance", "familiar", "friend", "close")


def peer_key_rotated_entries(persona_dir: Path, *, limit: int = 10) -> list[dict]:
    """Return most recent peer-key-rotation events from transport.jsonl."""
    from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

    log_path = Path(persona_dir) / "kindled_link" / "transport.jsonl"
    if not log_path.exists():
        return []
    rows = [
        r for r in iter_jsonl_skipping_corrupt(log_path)
        if r.get("event") == "peer_key_rotated"
    ]
    rows.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return rows[:limit]



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
