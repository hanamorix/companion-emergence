"""brain.kindled_link.audit — append-only transport audit logger.

Writes one compact JSON line per event to
``<persona_dir>/kindled_link/transport.jsonl``.  Fail-soft: a logging failure
must never raise into the caller (a poll/push failure is bad; a log failure
must never compound it).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from brain.health.json_io import append_jsonl

logger = logging.getLogger(__name__)


def log_transport(
    persona_dir: Path,
    *,
    event: str,
    peer_id: str | None = None,
    session_id: str | None = None,
    seq: int | None = None,
    reject_reason: str | None = None,
    count: int | None = None,
    relay_ok: bool | None = None,
    error: str | None = None,
    now: datetime,
) -> None:
    """Append one transport-event row to the audit log.

    Only non-None optional fields are included (compact rows).
    ``now`` is passed in explicitly — no internal clock call — so tests can
    pin the timestamp without monkeypatching.
    """
    row: dict = {"ts": now.isoformat(), "event": event}
    if peer_id is not None:
        row["peer_id"] = peer_id
    if session_id is not None:
        row["session_id"] = session_id
    if seq is not None:
        row["seq"] = seq
    if reject_reason is not None:
        row["reject_reason"] = reject_reason
    if count is not None:
        row["count"] = count
    if relay_ok is not None:
        row["relay_ok"] = relay_ok
    if error is not None:
        row["error"] = error[:200]  # #367: why the relay failed, not just that it did

    try:
        append_jsonl(
            Path(persona_dir) / "kindled_link" / "transport.jsonl", row, on_error="log", nonfinite="null"
        )
    except Exception:
        logger.exception("transport audit append failed")
