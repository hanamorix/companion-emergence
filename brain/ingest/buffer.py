"""SP-4 BUFFER + CLOSE stages — per-persona session JSONL files.

One file per session: <persona_dir>/active_conversations/<session_id>.jsonl
Each line is a JSON-encoded turn record:
  {session_id, speaker, text, ts}

The append-only JSONL design is faithful to OG nell_conversation_ingest.py.
Reads tolerate corrupt lines via brain.health.json_io.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from brain.health.json_io import (
    Recovery,
    append_jsonl,
    encode_jsonl,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from brain.utils.atomic_write import fsync_dir, save_with_backup_text

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
    append_jsonl(_session_path(persona_dir, session_id), record)
    return session_id


def list_active_sessions(persona_dir: Path) -> list[str]:
    """Return session_ids of all active buffer files."""
    d = persona_dir / "active_conversations"
    if not d.exists():
        return []
    return [p.stem for p in d.iterdir() if p.is_file() and p.suffix == ".jsonl"]


def read_session(persona_dir: Path, session_id: str) -> list[dict]:
    """Return the turns from a session buffer, skipping malformed lines."""
    # RAISE: compaction rewrites the buffer from this read and ingest advances its cursor on it, so a failed
    # read must raise (as it always did), never look like an empty session.
    return read_jsonl(_session_path(persona_dir, session_id), recover=Recovery.RAISE)


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


_UNLINK_RETRIES = 8
_UNLINK_RETRY_SLEEP_S = 0.05


def _unlink_with_retry(path: Path) -> None:
    """Unlink with a short bounded retry on transient PermissionError.

    Windows cannot delete a file another handle holds open (WinError 32) —
    a just-flushed writer, a not-yet-GC'd reader generator, or the CI
    runner's antivirus/indexer briefly pins freshly-written files. On POSIX
    the retry never triggers. Re-raises after the budget so a *real*
    permission problem still surfaces (fail-loud, matching the ingest
    pipeline's error counters)."""
    for attempt in range(_UNLINK_RETRIES):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == _UNLINK_RETRIES - 1:
                raise
            time.sleep(_UNLINK_RETRY_SLEEP_S * (attempt + 1))


def delete_session_buffer(persona_dir: Path, session_id: str) -> None:
    """Unlink the buffer file. Idempotent — no-op when file is missing."""
    path = _session_path(persona_dir, session_id)
    _unlink_with_retry(path)


def _cursor_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/active_conversations/<session_id>.cursor.

    Same session_id validation as _session_path so the cursor file lands
    inside the active_conversations dir, never traversed.
    """
    _validate_session_id(session_id)
    return _active_conversations_dir(persona_dir) / f"{session_id}.cursor"


def extract_lock_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/active_conversations/<session_id>.extract, the path whose ``file_lock``
    sidecar serialises extraction of one session (#477). The ``.jsonl`` glob of
    ``list_active_sessions`` never picks it up; ``_validate_session_id`` keeps it inside the dir."""
    _validate_session_id(session_id)
    return _active_conversations_dir(persona_dir) / f"{session_id}.extract"


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
    # The shared atomic text writer (#477): a unique temp per save, a bounded retry of a
    # transient PermissionError on the replace, fsync, and the temp removed on failure.
    save_with_backup_text(path, ts, keep_backups=False)


def delete_cursor(persona_dir: Path, session_id: str) -> None:
    """Idempotent unlink of the cursor file (retry — see _unlink_with_retry)."""
    path = _cursor_path(persona_dir, session_id)
    _unlink_with_retry(path)


def _rolled_to_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/active_conversations/<session_id>.rolled_to — the successor
    pointer written by a session rollover. A ``.jsonl`` glob never picks it up, so it
    is not seen as a live session; it survives the old buffer's deletion so a
    continuous client still holding the old sid is redirected to the successor."""
    _validate_session_id(session_id)
    return _active_conversations_dir(persona_dir) / f"{session_id}.rolled_to"


def write_rolled_to(persona_dir: Path, session_id: str, successor_id: str) -> None:
    """Atomically write the successor pointer (through the shared atomic writer). Written
    under the compaction lock BEFORE the old buffer is deleted, so the old sid never
    resolves to nothing."""
    _validate_session_id(session_id)
    _validate_session_id(successor_id)
    write_json(
        _rolled_to_path(persona_dir, session_id),
        {"successor": successor_id, "at": _now_iso()},
        indent=None,
        fsync=True,
    )


def read_rolled_to(persona_dir: Path, session_id: str) -> str | None:
    """Return the successor sid this session rolled over to, or None if absent/malformed."""
    data = read_json(_rolled_to_path(persona_dir, session_id), default=dict, expect=dict)
    succ = data.get("successor")
    return succ if isinstance(succ, str) and succ else None


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
    data = read_json(_backoff_path(persona_dir, session_id), default=lambda: None, expect=dict)
    if data is None or "failures" not in data or "first_failure_at" not in data:
        return None
    try:
        # Validate ts parses
        datetime.fromisoformat(str(data["first_failure_at"]).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return data


def write_backoff(persona_dir: Path, session_id: str, failures: int, first_failure_at: str) -> None:
    """Atomically write the backoff state. Raises ValueError on bad ts."""
    datetime.fromisoformat(first_failure_at.replace("Z", "+00:00"))  # validate
    write_json(
        _backoff_path(persona_dir, session_id),
        {"failures": failures, "first_failure_at": first_failure_at},
        indent=None,
        fsync=False,
    )


def delete_backoff(persona_dir: Path, session_id: str) -> None:
    """Idempotent unlink of the backoff sidecar (retry — see _unlink_with_retry)."""
    path = _backoff_path(persona_dir, session_id)
    _unlink_with_retry(path)


def rewrite_session_atomic(persona_dir: Path, session_id: str, turns: list[dict]) -> None:
    """Atomically replace the active buffer with ``turns`` (verbatim).

    Unlike ``ingest_turn`` this writes each record EXACTLY as given — no
    ``.strip()``, no key rebuild — so nested fields (e.g. a summary row's
    ``compaction`` provenance object) and exact byte content survive. Used by
    compaction to install ``[summary, *retained_tail]``. Temp file + os.replace
    mirrors ``write_cursor`` so a reader never sees a torn buffer.

    The replace gets the same bounded retry as ``_unlink_with_retry`` (#198):
    Windows refuses ``os.replace`` over a file another handle holds open — a
    ``GET /chat/history`` streaming this buffer is the realistic case, and it
    finishes in milliseconds. On POSIX the retry never triggers. After the
    budget the error re-raises (the old buffer is intact) and the temp file is
    removed so a repeated collision cannot litter ``active_conversations``.
    """
    write_jsonl(_session_path(persona_dir, session_id), turns, fsync=True)


# Archive rotation (1d). A single unbounded ``<sid>.jsonl`` grows without limit on
# a long-lived session; segmenting bounds each file. A hard BYTE cap is chosen over
# per-day so a bursty day cannot make one segment huge. The provenance chain (faded
# summaries + superseded raw turns, in order) is preserved across segments by the
# segment-aware reader.
_ARCHIVE_SEGMENT_MAX_BYTES = 5 * 1024 * 1024


def _archive_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve the LEGACY single-file archive path <persona>/archived_conversations/
    <session_id>.jsonl. Retained for back-compat: a pre-segmentation archive still
    reads. New writes roll into ``<session_id>.<NNN>.jsonl`` segments."""
    _validate_session_id(session_id)
    return _archived_conversations_dir(persona_dir) / f"{session_id}.jsonl"


def _archive_segments(persona_dir: Path, session_id: str) -> list[tuple[int, Path]]:
    """Return existing ``<sid>.<NNN>.jsonl`` segments as (num, path), numeric-sorted."""
    _validate_session_id(session_id)
    d = _archived_conversations_dir(persona_dir)
    prefix = f"{session_id}."
    suffix = ".jsonl"
    out: list[tuple[int, Path]] = []
    for p in d.glob(f"{session_id}.*.jsonl"):
        mid = p.name[len(prefix) : -len(suffix)]
        if mid.isdigit():
            out.append((int(mid), p))
    out.sort(key=lambda t: t[0])
    return out


def _archive_read_order(persona_dir: Path, session_id: str) -> list[Path]:
    """All archive files for a sid in provenance order: the legacy single file (if
    present, the oldest) then every rolling segment, numeric-ordered."""
    paths: list[Path] = []
    legacy = _archive_path(persona_dir, session_id)
    if legacy.exists():
        paths.append(legacy)
    paths.extend(p for _, p in _archive_segments(persona_dir, session_id))
    return paths


def append_archive(persona_dir: Path, session_id: str, records: list[dict]) -> int:
    """Append ``records`` to the conversation archive, fsync, return byte count.

    Append-only, rolled into size-capped segments ``<sid>.<NNN>.jsonl`` (never a
    truncating rotation — that would destroy the provenance chain). When the active
    segment would exceed ``_ARCHIVE_SEGMENT_MAX_BYTES`` a new segment is created and
    its containing directory fsynced (crash-mid-roll durability, C11). The returned
    byte count lets the caller verify the write landed before mutating the live
    buffer (the archive-before-rewrite atomicity contract).
    """
    if not records:
        return 0
    _validate_session_id(session_id)
    d = _archived_conversations_dir(persona_dir)
    data = encode_jsonl(records)

    # Choose the active target: the highest existing segment, else a legacy single
    # file if one exists, else a fresh segment 000.
    segments = _archive_segments(persona_dir, session_id)
    if segments:
        active_path = segments[-1][1]
    else:
        legacy = d / f"{session_id}.jsonl"
        active_path = legacy if legacy.exists() else d / f"{session_id}.000.jsonl"

    is_new_segment = not active_path.exists()
    if not is_new_segment:
        try:
            cur_size = active_path.stat().st_size
        except OSError:
            cur_size = 0
        # Roll only a non-empty segment (a huge single record still lands somewhere).
        if cur_size > 0 and cur_size + len(data) > _ARCHIVE_SEGMENT_MAX_BYTES:
            next_num = (segments[-1][0] + 1) if segments else 0
            active_path = d / f"{session_id}.{next_num:03d}.jsonl"
            is_new_segment = True

    written = append_jsonl(active_path, data, fsync=True)
    if is_new_segment:
        fsync_dir(d)
    return written


def read_archive(persona_dir: Path, session_id: str) -> list[dict]:
    """Return archived records (raw turns + faded summaries) across ALL segments in
    provenance order, skipping corrupt lines (tolerates a torn trailing line / a
    zero-length trailing segment left by a crash mid-roll)."""
    out: list[dict] = []
    for p in _archive_read_order(persona_dir, session_id):
        out.extend(read_jsonl(p, recover=Recovery.RAISE))
    return out


# --- Archive commit marker (Bug 3: idempotent archive-under-crash-retry) -----
# A durable sidecar proving a specific archive-append batch completed, so a
# crash-retry compaction pass can skip a redundant re-append ONLY when the
# marker PROVES the identical batch already landed — never by inferring from
# turn identity (that risks silently dropping a byte-identical duplicate turn).
# ``append_archive`` itself is left untouched: its append-only/never-dedup
# contract (that ``read_archive`` relies on) stays intact. A non-``.jsonl`` name
# so the session/segment globs never pick it up.


def _archive_marker_path(persona_dir: Path, session_id: str) -> Path:
    _validate_session_id(session_id)
    return _archived_conversations_dir(persona_dir) / f"{session_id}.archived"


def read_archive_marker(persona_dir: Path, session_id: str) -> str | None:
    """Return the last completed archive-batch key for this session, or None
    when the sidecar is missing, empty, or unreadable."""
    path = _archive_marker_path(persona_dir, session_id)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def write_archive_marker(persona_dir: Path, session_id: str, key: str) -> None:
    """Atomically write the archive commit-marker through the shared atomic text writer
    (unique temp + retried replace + fsync), as ``write_cursor`` does, and fsync the directory
    for a new marker — this marker must be durable
    across a crash landing immediately after the archive append's own fsync."""
    path = _archive_marker_path(persona_dir, session_id)
    is_new = not path.exists()
    save_with_backup_text(path, key, keep_backups=False, fsync=True)
    if is_new:
        # fsync the containing dir so the new sidecar's directory entry is durable
        # (crash-mid-create), matching append_archive's new-segment durability.
        fsync_dir(path.parent)


def _compacting_lock_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/active_conversations/<session_id>.compacting.

    A `.jsonl`-suffixed glob (list_active_sessions) never picks this up, and
    _validate_session_id keeps it inside the dir."""
    _validate_session_id(session_id)
    return _active_conversations_dir(persona_dir) / f"{session_id}.compacting"


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
    from brain import process_identity

    payload = json.dumps({"pid": os.getpid(), "ts": _now_iso(), "token": process_identity.own_token()})  # json-io: ok -- O_EXCL lock-file payload: a lock protocol, not a state file
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        return True
    except FileExistsError:
        # Held — decide whether it is stale and reapable.
        reap = False
        holder = read_json(path, default=dict, expect=dict)  # empty/torn payload (or unreadable): not alive
        try:
            holder_pid = int(holder.get("pid", -1))
        except (TypeError, ValueError):
            holder_pid = -1
        holder_token = holder.get("token") if isinstance(holder.get("token"), str) else None
        if holder_pid <= 0 or not process_identity.holder_alive(holder_pid, holder_token):
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
