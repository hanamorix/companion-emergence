"""SP-4 BUFFER + CLOSE stages — per-persona session JSONL files.

One file per session: <persona_dir>/active_conversations/<session_id>.jsonl
Each line is a JSON-encoded turn record:
  {session_id, speaker, text, ts}

The append-only JSONL design is faithful to OG nell_conversation_ingest.py.
Reads tolerate corrupt lines via read_jsonl_skipping_corrupt.
"""

from __future__ import annotations

import json
import os
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


def _archived_conversations_dir(persona_dir: Path) -> Path:
    """Sibling of active_conversations/ — the lossless, append-only, NEVER
    size-rotated archive of compacted raw turns + faded summaries. Kept a
    sibling (not nested) so list_active_sessions / snapshot_stale_sessions
    never glob it as a live session."""
    d = persona_dir / "archived_conversations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _validate_session_id(session_id: str) -> None:
    """Raise ValueError unless session_id matches _SESSION_ID_RE.

    Shared by _session_path and _cursor_path so the grammar can evolve in
    one place. The isinstance check is kept for callers that ignore type
    hints — _SESSION_ID_RE.fullmatch on a non-str raises TypeError, which
    we'd rather surface as the same ValueError.
    """
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(f"invalid session_id {session_id!r} — must match [A-Za-z0-9_-]{{1,64}}")


def _session_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/active_conversations/<session_id>.jsonl.

    Validates session_id against _SESSION_ID_RE so a path-traversal string
    (e.g., '../../etc/passwd') or empty string can't escape the buffer dir.
    The HTTP bridge constrains session_id to UUID at the request-model
    layer, but other callers (chat engine, future MCP, tests) inherit
    this validation defensively.
    """
    _validate_session_id(session_id)
    return _active_conversations_dir(persona_dir) / f"{session_id}.jsonl"


def ingest_turn(persona_dir: Path, turn: dict) -> str:
    """Append one turn to the session buffer. Returns session_id.

    Required keys in ``turn``: speaker, text.
    Optional:
        session_id  — if absent, a new UUID-based session id is generated.
        ts          — ISO-8601 timestamp; defaults to now (UTC).
        image_shas  — list of sha-strings for images attached to this
                      turn. References content stored under
                      ``<persona_dir>/images/<sha>.<ext>``. Recorded on
                      the JSONL line so downstream stages (extract,
                      commit) can surface images to memory metadata.

    Buffer path: <persona_dir>/active_conversations/<session_id>.jsonl
    """
    session_id: str = turn.get("session_id") or f"sess_{uuid.uuid4().hex[:8]}"
    record: dict = {
        "session_id": session_id,
        "speaker": turn.get("speaker", "unknown"),
        "text": (turn.get("text") or "").strip(),
        "ts": turn.get("ts") or _now_iso(),
    }
    image_shas = turn.get("image_shas")
    if image_shas:
        # Defensive copy so callers passing tuples or generators end up
        # with a JSON-serialisable list on the record.
        record["image_shas"] = list(image_shas)
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


def _cursor_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/active_conversations/<session_id>.cursor.

    Same session_id validation as _session_path so the cursor file lands
    inside the active_conversations dir, never traversed.
    """
    _validate_session_id(session_id)
    return _active_conversations_dir(persona_dir) / f"{session_id}.cursor"


def read_cursor(persona_dir: Path, session_id: str) -> str | None:
    """Return the cursor ts (ISO string) for a session.

    Returns None when the cursor file is missing, empty, or its content
    doesn't parse as an ISO-8601 timestamp. Callers treat None as
    "extract from the beginning."
    """
    path = _cursor_path(persona_dir, session_id)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return text
    except ValueError:
        return None


def write_cursor(persona_dir: Path, session_id: str, ts: str) -> None:
    """Atomically write the cursor file. Raises ValueError on bad ts."""
    datetime.fromisoformat(ts.replace("Z", "+00:00"))
    path = _cursor_path(persona_dir, session_id)
    tmp = path.with_suffix(".cursor.tmp")
    tmp.write_text(ts, encoding="utf-8")
    os.replace(tmp, path)


def delete_cursor(persona_dir: Path, session_id: str) -> None:
    """Idempotent unlink of the cursor file."""
    path = _cursor_path(persona_dir, session_id)
    path.unlink(missing_ok=True)


def _backoff_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/active_conversations/<session_id>.backoff.

    Sibling sidecar to the cursor file used by extract_session_snapshot
    to record consecutive extraction failures and pause retries when a
    buffer is wedged (F-011). Lives next to the cursor rather than
    folded into it so the cursor stays a single-line atomic ISO ts.
    """
    _validate_session_id(session_id)
    return _active_conversations_dir(persona_dir) / f"{session_id}.backoff"


def read_backoff(persona_dir: Path, session_id: str) -> dict | None:
    """Return the backoff state ({failures: int, first_failure_at: str}) or
    None if the file is missing, empty, or malformed."""
    path = _backoff_path(persona_dir, session_id)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        if "failures" not in data or "first_failure_at" not in data:
            return None
        # Validate ts parses
        datetime.fromisoformat(str(data["first_failure_at"]).replace("Z", "+00:00"))
        return data
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def write_backoff(persona_dir: Path, session_id: str, failures: int, first_failure_at: str) -> None:
    """Atomically write the backoff state. Raises ValueError on bad ts."""
    datetime.fromisoformat(first_failure_at.replace("Z", "+00:00"))  # validate
    path = _backoff_path(persona_dir, session_id)
    tmp = path.with_suffix(".backoff.tmp")
    tmp.write_text(
        json.dumps({"failures": failures, "first_failure_at": first_failure_at}),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def delete_backoff(persona_dir: Path, session_id: str) -> None:
    """Idempotent unlink of the backoff sidecar."""
    path = _backoff_path(persona_dir, session_id)
    path.unlink(missing_ok=True)


def rewrite_session_atomic(persona_dir: Path, session_id: str, turns: list[dict]) -> None:
    """Atomically replace the active buffer with ``turns`` (verbatim).

    Unlike ``ingest_turn`` this writes each record EXACTLY as given — no
    ``.strip()``, no key rebuild — so nested fields (e.g. a summary row's
    ``compaction`` provenance object) and exact byte content survive. Used by
    compaction to install ``[summary, *retained_tail]``. Temp file + os.replace
    mirrors ``write_cursor`` so a reader never sees a torn buffer.
    """
    path = _session_path(persona_dir, session_id)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in turns:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _archive_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/archived_conversations/<session_id>.jsonl."""
    _validate_session_id(session_id)
    return _archived_conversations_dir(persona_dir) / f"{session_id}.jsonl"


def append_archive(persona_dir: Path, session_id: str, records: list[dict]) -> int:
    """Append ``records`` to the conversation archive, fsync, return byte count.

    Append-only and NEVER size-rotated (a truncating rotation would destroy the
    provenance chain and no multi-segment reader exists). The returned byte
    count lets the caller verify the write landed before mutating the live
    buffer (the archive-before-rewrite atomicity contract).
    """
    if not records:
        return 0
    path = _archive_path(persona_dir, session_id)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    data = payload.encode("utf-8")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    return len(data)


def read_archive(persona_dir: Path, session_id: str) -> list[dict]:
    """Return archived records (raw turns + faded summaries), skipping corrupt lines."""
    return read_jsonl_skipping_corrupt(_archive_path(persona_dir, session_id))


def _compacting_lock_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/active_conversations/<session_id>.compacting.

    A `.jsonl`-suffixed glob (list_active_sessions) never picks this up, and
    _validate_session_id keeps it inside the dir."""
    _validate_session_id(session_id)
    return _active_conversations_dir(persona_dir) / f"{session_id}.compacting"


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists (signal 0 probe)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def acquire_compaction_lock(
    persona_dir: Path, session_id: str, *, stale_s: float = 600.0
) -> bool:
    """Try to acquire the per-session compaction lock. Return True on success.

    Re-entrancy guard for the daily tick vs a manual tool call. A leftover lock
    from a crash must not disable compaction forever, so a held lock is REAPED
    when the recorded pid is dead (primary signal) OR its mtime is older than
    ``stale_s`` (backstop for pid reuse). A live, recent lock → return False
    (skip). The caller releases via release_compaction_lock in a finally.
    """
    path = _compacting_lock_path(persona_dir, session_id)
    payload = json.dumps({"pid": os.getpid(), "ts": _now_iso()})
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        return True
    except FileExistsError:
        # Held — decide whether it is stale and reapable.
        reap = False
        try:
            holder = json.loads(path.read_text(encoding="utf-8"))
            holder_pid = int(holder.get("pid", -1))
        except (OSError, ValueError, TypeError):
            holder_pid = -1
        if not _pid_alive(holder_pid):
            reap = True
        else:
            try:
                age = datetime.now(UTC).timestamp() - path.stat().st_mtime
                if age > stale_s:
                    reap = True
            except OSError:
                reap = False
        if not reap:
            return False
        # Reap and re-acquire. unlink+re-create (not a plain overwrite) so two
        # racers can't both believe they won — the O_EXCL re-create arbitrates.
        try:
            path.unlink(missing_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            return True
        except FileExistsError:
            return False


def release_compaction_lock(persona_dir: Path, session_id: str) -> None:
    """Idempotent unlink of the compaction lock sidecar."""
    _compacting_lock_path(persona_dir, session_id).unlink(missing_ok=True)


def read_session_after(persona_dir: Path, session_id: str, after_ts: str | None) -> list[dict]:
    """Return turns whose ts > after_ts.

    after_ts=None returns all turns. Malformed after_ts also returns all
    turns (logged at caller layer). Turns whose ts is missing or unparseable
    are skipped silently.
    """
    turns = read_session(persona_dir, session_id)
    if after_ts is None:
        return turns
    try:
        cutoff = datetime.fromisoformat(after_ts.replace("Z", "+00:00"))
    except ValueError:
        return turns
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=UTC)
    out: list[dict] = []
    for t in turns:
        raw = t.get("ts")
        if not raw:
            continue
        try:
            t_dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if t_dt.tzinfo is None:
            t_dt = t_dt.replace(tzinfo=UTC)
        if t_dt > cutoff:
            out.append(t)
    return out
