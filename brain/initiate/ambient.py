"""Always-on verify slice for prompt construction.

Injected into every chat prompt's system message between persona context
and ambient memory. Two jobs:

1. Prevent 'I forgot I already reached out' — recent outbound (last 5)
   stays in ambient context.
2. Surface acknowledged_unclear so the ask-pattern has a hook — Nell can
   choose to bring up 'did you see what I sent earlier' organically.

Returns None when there's no relevant history (fresh install). Otherwise
returns a formatted text block ready to splice into the system message.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.initiate.audit import read_recent_audit


def build_outbound_recall_block(
    persona_dir: Path,
    *,
    now: datetime | None = None,
    recent_cap: int = 5,
) -> str | None:
    """Return the always-on verify slice as text, or None if empty."""
    now = now or datetime.now(UTC)

    # Pull last 24h of audit; we'll filter inline.
    rows = list(read_recent_audit(persona_dir, window_hours=24, now=now))

    # Recent outbound: actual sends (send_notify / send_quiet), latest first, capped.
    sent_rows = [
        r
        for r in rows
        if r.decision in ("send_notify", "send_quiet") and r.delivery is not None
    ]
    sent_rows.sort(key=lambda r: r.ts, reverse=True)
    sent_rows = sent_rows[:recent_cap]

    # Pending uncertainty: acknowledged_unclear states from the last 24h.
    unclear_rows = [
        r
        for r in sent_rows
        if r.delivery and r.delivery.get("current_state") == "acknowledged_unclear"
    ]

    if not sent_rows:
        return None

    lines = ["Recent outbound:"]
    for r in sent_rows:
        urgency = "notify" if r.decision == "send_notify" else "quiet"
        state = r.delivery.get("current_state", "delivered") if r.delivery else "?"
        preview = r.subject[:60] if r.subject else "(no subject)"
        lines.append(f'- {r.ts} ({urgency}) — "{preview}" — state: {state}')

    if unclear_rows:
        lines.append("")
        lines.append("Pending uncertainty:")
        for r in unclear_rows:
            preview = r.subject[:60] if r.subject else "(no subject)"
            lines.append(
                f'- {r.ts} — "{preview}" — acknowledged_unclear '
                "(no clear topical thread since you saw it)"
            )

    return "\n".join(lines)
