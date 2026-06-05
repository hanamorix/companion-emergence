"""Per-call token-usage audit log. Best-effort — never breaks a turn."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

def log_usage(persona_dir: Path | None, *, call_type: str, model: str, frame: dict) -> None:
    if persona_dir is None:
        return
    try:
        u = frame.get("usage") or {}
        row = {
            "ts": datetime.now(UTC).isoformat(), "call_type": call_type, "model": model,
            "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": u.get("cache_read_input_tokens"),
            "total_cost_usd": frame.get("total_cost_usd"), "num_turns": frame.get("num_turns"),
            "duration_ms": frame.get("duration_ms"), "session_id": frame.get("session_id"),
        }
        persona_dir.mkdir(parents=True, exist_ok=True)
        with (persona_dir / "chat_usage.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:  # noqa: BLE001 — instrumentation must never break a turn
        log.debug("log_usage failed", exc_info=True)
