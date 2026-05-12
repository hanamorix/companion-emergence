"""On-demand verify tools — read-only, return text formatted for reading.

Three tools available to Nell during her turn:
  - recall_initiate_audit(window, filter_state) — initiate decisions
  - recall_soul_audit(window) — soul-review decisions
  - recall_voice_evolution() — accepted voice-template changes

All tools are read-only, return formatted text (never raw JSON), and have
generous defaults so malformed invocations still return something useful.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain.initiate.audit import read_recent_audit

_WINDOW_HOURS: dict[str, float] = {
    "24h": 24,
    "7d": 24 * 7,
    "30d": 24 * 30,
    "all": 24 * 365 * 10,  # 10 years — effectively unbounded
}


def _resolve_window_hours(window: str) -> float:
    return _WINDOW_HOURS.get(window, 24)


def recall_initiate_audit(
    persona_dir: Path,
    *,
    window: str = "24h",
    filter_state: str | None = None,
) -> str:
    """Return formatted initiate audit slice for the given window."""
    window_hours = _resolve_window_hours(window)
    rows = list(read_recent_audit(persona_dir, window_hours=window_hours))
    if filter_state:
        rows = [
            r
            for r in rows
            if r.delivery and r.delivery.get("current_state") == filter_state
        ]
    if not rows:
        return "(no recent initiate decisions in this window)"
    lines: list[str] = []
    for r in rows:
        state = (r.delivery.get("current_state") if r.delivery else "n/a") or "n/a"
        lines.append(
            f"{r.ts} | {r.decision} | {r.subject[:80]} | state={state}"
        )
    return "\n".join(lines)


def recall_soul_audit(persona_dir: Path, *, window: str = "30d") -> str:
    """Return formatted soul audit slice — reuses the v0.0.8 fan-out reader."""
    from brain.soul.audit import iter_audit_full

    window_hours = _resolve_window_hours(window)
    cutoff = (datetime.now(UTC) - timedelta(hours=window_hours)).isoformat()
    rows = [r for r in iter_audit_full(persona_dir) if r.get("ts", "") >= cutoff]
    if not rows:
        return "(no recent soul decisions in this window)"
    return "\n".join(
        f"{r.get('ts')} | {r.get('decision')} | {(r.get('candidate_text') or '')[:80]}"
        for r in rows
    )


def recall_voice_evolution(persona_dir: Path) -> str:
    """Return all voice_evolution records chronologically."""
    from brain.soul.store import SoulStore

    try:
        store = SoulStore(str(persona_dir / "crystallizations.db"))
        try:
            evolutions = store.list_voice_evolution()
        finally:
            store.close()
    except Exception:
        return "(no voice evolution history)"
    if not evolutions:
        return "(no voice evolution history)"
    return "\n".join(
        f"{v.accepted_at}: {v.old_text!r} -> {v.new_text!r}  ({v.rationale})"
        for v in evolutions
    )


NELL_TOOLS: dict[str, dict[str, Any]] = {
    "recall_initiate_audit": {
        "description": (
            "Read your recent initiate decisions. Use this when you want "
            "to check what you've reached out about, what state your "
            "messages are in, or whether something needs an ask-pattern "
            "follow-up."
        ),
        "args": {
            "window": "one of '24h', '7d', '30d', 'all' (default '24h')",
            "filter_state": "optional — one of the state names to filter to",
        },
        "callable": recall_initiate_audit,
    },
    "recall_soul_audit": {
        "description": (
            "Read your recent soul-review decisions — the durable record of "
            "how your beliefs have evolved."
        ),
        "args": {"window": "one of '24h', '7d', '30d', 'all' (default '30d')"},
        "callable": recall_soul_audit,
    },
    "recall_voice_evolution": {
        "description": (
            "Read every voice-template change you've ever made — the "
            "queryable answer to 'what have I changed about myself recently?'"
        ),
        "args": {},
        "callable": recall_voice_evolution,
    },
}
