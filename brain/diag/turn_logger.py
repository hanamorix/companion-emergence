"""Per-turn diagnostic logger for the claude-cli provider (OFF BY DEFAULT).

Captures, for one chat turn, the raw stdin SENT to the claude subprocess and the raw
output RECEIVED *before* ``_truncate_at_role_leak`` (or, on the streaming path, the raw
``StreamDone`` content — that path does not truncate), plus send/receive timestamps and
the usage/cache-token block. Written as one JSONL record per turn to ``turn_diag.jsonl``
next to the persona's other logs.

This exists to instrument the S1 monologue-bleed hunt (see
hunts/monologue-bleed-s1/): the mitigations strip the transcript-continuation before the
stored logs ever see it, so this is the only way to observe the pre-truncation "script".

Design (see the change record changes/turn-logger/):
  * OFF by default — gated by the tunable ``diag.turn_logger_enabled`` (default False).
    Active only on a throwaway test persona (Canary), never in normal use.
  * FAIL-OPEN — any error here is swallowed; a debug-log failure must never affect a turn.
  * CONCURRENCY — records carry full prompts (>> PIPE_BUF), so appends are serialized by a
    module lock to avoid torn lines. Every writer funnels through ``log_turn``.
  * BEHAVIOR-PRESERVING — pure observation; reads existing strings, re-materializes
    ``messages`` into new dicts (never mutates them, consumes no generator).
"""
from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain import tunables

_ENABLED_KEY = "diag.turn_logger_enabled"
_PATH_KEY = "diag.turn_logger_path"
tunables.register(_ENABLED_KEY, False)
tunables.register(_PATH_KEY, "")

_WLOCK = threading.Lock()


def enabled() -> bool:
    return bool(tunables.get_tunable(_ENABLED_KEY, False))


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_messages(messages: Iterable[Any] | None) -> list[dict]:
    """Serialize chat messages to {role, text} without ever raising on multimodal
    (image) blocks — a naive json.dumps of those could raise and, under fail-open,
    silently drop exactly the image turns."""
    out: list[dict] = []
    for m in messages or []:
        try:
            role = getattr(m, "role", None)
            text: str | None = None
            ctf = getattr(m, "content_text", None)
            if callable(ctf):
                try:
                    text = ctf()
                except Exception:  # noqa: BLE001
                    text = None
            if text is None:
                c = getattr(m, "content", None)
                text = c if isinstance(c, str) else f"<non-text content: {type(c).__name__}>"
            out.append({"role": role, "text": text})
        except Exception:  # noqa: BLE001
            out.append({"role": "?", "text": "<unserializable message>"})
    return out


def _extract_usage(frame: Any) -> dict:
    """Pull the usage/cache fields from a claude-cli result frame. (log_usage writes
    directly and returns None, so we duplicate the small extraction here.)"""
    if not isinstance(frame, dict):
        return {}
    out: dict = {}
    usage = frame.get("usage")
    if isinstance(usage, dict):
        for k in ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens"):
            if k in usage:
                out[k] = usage[k]
    for k in ("total_cost_usd", "num_turns", "duration_ms"):
        if k in frame:
            out[k] = frame[k]
    return out


def _resolve_path(persona_dir: Any) -> Path:
    override = tunables.get_tunable(_PATH_KEY, "")
    if override:
        return Path(override)
    return Path(persona_dir) / "turn_diag.jsonl"


def log_turn(
    persona_dir: Any,
    *,
    path: str,
    system: str | None,
    messages: Iterable[Any] | None,
    volatile: str | None,
    sent_blob: str | None,
    sent_ts: str,
    received_raw: str | None,
    received_ts: str,
    usage_frame: Any,
) -> None:
    """Append one turn record. No-op when disabled; fail-open on any error.

    ``sent_blob`` is the ACTUAL stdin handed to the subprocess for this path
    (flat_prompt for text/mcp/stream; stdin_payload for image). ``volatile`` is the
    volatile suffix (text/stream) or full_system (image). ``received_raw`` is the
    pre-truncation content (text/image/mcp) or the raw StreamDone content (stream)."""
    if not enabled() or not persona_dir:
        return
    try:
        record = {
            "turn_id": uuid.uuid4().hex,
            "path": path,
            "sent_ts": sent_ts,
            "received_ts": received_ts,
            "sent": {
                "system": system,
                "messages": _safe_messages(messages),
                "volatile_or_full_system": volatile,
                "stdin_sent": sent_blob,
            },
            "received_raw": received_raw,
            "usage": _extract_usage(usage_frame),
        }
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        out = _resolve_path(persona_dir)
        with _WLOCK:
            _write_line(out, line)
    except Exception:  # noqa: BLE001 — diagnostic logging must never break a turn
        pass


def _write_line(out: Path, line: str) -> None:
    """The single append site — always called under ``_WLOCK`` from ``log_turn``.
    Factored out so the guard's scope (write executes while the lock is held) is
    deterministically testable."""
    with out.open("a", encoding="utf-8") as f:
        f.write(line)
