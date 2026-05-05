"""SP-4 BUFFER + CLOSE stages — per-persona session JSONL files.

One file per session: <persona_dir>/active_conversations/<session_id>.jsonl
Each line is a JSON-encoded turn record:
  {session_id, speaker, text, ts}

The append-only JSONL design is faithful to OG nell_conversation_ingest.py.
Reads tolerate corrupt lines via read_jsonl_skipping_corrupt.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

# session_id grammar: UUIDs (hyphens included) and the `sess_<8 hex>` fallback
# both fit. Letters / digits / hyphen / underscore only — no slashes, dots, or
# other path-traversal characters. Capped at 64 chars to bound the filename.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _active_conversations_dir(persona_dir: Path) -> Path:
    d = persona_dir / "active_conversations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/active_conversations/<session_id>.jsonl.

    Validates session_id against _SESSION_ID_RE so a path-traversal string
    (e.g., '../../etc/passwd') or empty string can't escape the buffer dir.
    The HTTP bridge constrains session_id to UUID at the request-model
    layer, but other callers (chat engine, future MCP, tests) inherit
    this validation defensively.
    """
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(
            f"invalid session_id {session_id!r} — must match "
            f"[A-Za-z0-9_-]{{1,64}}"
        )
    return _active_conversations_dir(persona_dir) / f"{session_id}.jsonl"


def ingest_turn(persona_dir: Path, turn: dict) -> str:
    """Append one turn to the session buffer. Returns session_id.

    Required keys in ``turn``: speaker, text.
    Optional:
        session_id  — if absent, a new UUID-based session id is generated.
        ts          — ISO-8601 timestamp; defaults to now (UTC).

    Buffer path: <persona_dir>/active_conversations/<session_id>.jsonl
    """
    session_id: str = turn.get("session_id") or f"sess_{uuid.uuid4().hex[:8]}"
    record = {
        "session_id": session_id,
        "speaker": turn.get("speaker", "unknown"),
        "text": (turn.get("text") or "").strip(),
        "ts": turn.get("ts") or _now_iso(),
    }
    path = _session_path(persona_dir, session_id)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return session_id


def list_active_sessions(persona_dir: Path) -> list[str]:
    """Return session_ids of all active buffer files."""
    d = persona_dir / "active_conversations"
    if not d.exists():
        return []
    return [p.stem for p in d.iterdir() if p.is_file() and p.suffix == ".jsonl"]


def read_session(persona_dir: Path, session_id: str) -> list[dict]:
    """Return the turns from a session buffer, skipping malformed lines."""
    path = _session_path(persona_dir, session_id)
    return read_jsonl_skipping_corrupt(path)


def session_silence_minutes(turns: list[dict]) -> float:
    """Minutes elapsed since the last turn's ts.

    Returns 0.0 if turns is empty or the last ts cannot be parsed.
    """
    if not turns:
        return 0.0
    try:
        raw_ts = turns[-1]["ts"]
        last = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
    except (KeyError, ValueError, TypeError):
        return 0.0
    delta = datetime.now(UTC) - last
    return delta.total_seconds() / 60.0


def delete_session_buffer(persona_dir: Path, session_id: str) -> None:
    """Unlink the buffer file. Idempotent — no-op when file is missing."""
    path = _session_path(persona_dir, session_id)
    path.unlink(missing_ok=True)
