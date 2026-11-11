"""SP-4 BUFFER + CLOSE stages — per-persona session JSONL files.

One file per session: <persona_dir>/active_conversations/<session_id>.jsonl
Each line is a JSON-encoded turn record:
  {session_id, speaker, text, ts}

The append-only JSONL design is faithful to OG nell_conversation_ingest.py.
Reads tolerate corrupt lines via brain.health.json_io.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.json_io import (
    Recovery,
    append_jsonl,
    encode_jsonl,
    iter_jsonl,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from brain.health.jsonl_reader import UnloadableJsonlLineError
from brain.utils.atomic_write import fsync_dir, save_with_backup_text
from brain.utils.file_lock import file_lock
from brain.utils.file_ops import delete_file, list_dir, stat_path
from brain.utils.log_throttle import warn_throttled
from brain.utils.time import lenient_utc

logger = logging.getLogger(__name__)

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


@contextmanager
def session_append_lock(persona_dir: Path, session_id: str) -> Iterator[None]:
    """Hold the buffer's append lock: the lock ``ingest_turn`` takes (through ``append_jsonl``) for this session.

    A caller that reads the buffer and then removes it (the session rollover) holds this across the read,
    the archive write and the removal, so a row appended in between waits and cannot be removed unseen.
    Not re-entrant: an ``ingest_turn`` of the same session inside the block, on any thread of this process,
    waits for the block to end. Lock order: take the chat registry lock (``brain.chat.session.registry_lock``)
    first when a caller needs both, and never take the registry lock while holding this one.

    When the sidecar lock cannot be taken (``OSError``: a filesystem that refuses the lock) a warning is logged
    and the block runs unlocked, as ``append_jsonl`` does, since the appenders are unlocked there too and
    refusing would stop the caller for good. An exception raised by the block propagates once."""
    stack = ExitStack()
    try:
        stack.enter_context(file_lock(_session_path(persona_dir, session_id)))
    except OSError as exc:
        logger.warning(
            "could not take the append lock for session %s (%s); continuing without it", session_id, exc
        )
    with stack:
        yield


def list_active_sessions(persona_dir: Path) -> list[str]:
    """Return session_ids of all active buffer files."""
    d = persona_dir / "active_conversations"
    if not d.exists():
        return []
    return [p.stem for p in d.iterdir() if p.is_file() and p.suffix == ".jsonl"]


def read_session(persona_dir: Path, session_id: str) -> list[dict]:
    """Return the turns from a session buffer, skipping malformed lines (a line the JSON parser cannot load stops the read)."""
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


def delete_session_buffer(persona_dir: Path, session_id: str) -> None:
    """Delete the buffer file. Idempotent — no-op when file is missing; a transient sharing violation is retried."""
    path = _session_path(persona_dir, session_id)
    delete_file(path)


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
    """Idempotent delete of the cursor file (retried: ``file_ops.delete_file``)."""
    path = _cursor_path(persona_dir, session_id)
    delete_file(path)


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
    """Return the backoff state ({failures: int, first_failure_at: str}, plus ``failing_end_ts`` when the failing chunk's end
    was recorded) or None if the file is missing, empty, or malformed."""
    data = read_json(_backoff_path(persona_dir, session_id), default=lambda: None, expect=dict)
    if data is None or "failures" not in data or "first_failure_at" not in data:
        return None
    try:
        # Validate ts parses
        datetime.fromisoformat(str(data["first_failure_at"]).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return data


def write_backoff(
    persona_dir: Path, session_id: str, failures: int, first_failure_at: str, failing_end_ts: str | None = None
) -> None:
    """Atomically write the backoff state. Raises ValueError on bad ts.

    ``failing_end_ts``: the end (the ``cursor_ts``) of the chunk whose extraction failed, so that setting it aside in
    ``poison/`` takes only the rows that attempt carried. Optional: the key is left out when unknown, and a reader that does
    not know it (an older brain) ignores it.
    """
    datetime.fromisoformat(first_failure_at.replace("Z", "+00:00"))  # validate
    state = {"failures": failures, "first_failure_at": first_failure_at}
    if failing_end_ts is not None:
        state["failing_end_ts"] = failing_end_ts
    write_json(_backoff_path(persona_dir, session_id), state, indent=None, fsync=False)


def delete_backoff(persona_dir: Path, session_id: str) -> None:
    """Idempotent delete of the backoff sidecar (retried: ``file_ops.delete_file``)."""
    path = _backoff_path(persona_dir, session_id)
    delete_file(path)


def rewrite_session_atomic(persona_dir: Path, session_id: str, turns: list[dict]) -> None:
    """Atomically replace the active buffer with ``turns`` (verbatim).

    Unlike ``ingest_turn`` this writes each record EXACTLY as given — no
    ``.strip()``, no key rebuild — so nested fields (e.g. a summary row's
    ``compaction`` provenance object) and exact byte content survive. Used by
    compaction to install ``[summary, *retained_tail]``. Temp file + os.replace
    mirrors ``write_cursor`` so a reader never sees a torn buffer.

    The replace gets the shared bounded retry (``file_ops.retry_transient``, #198):
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
        if mid.isascii() and mid.isdigit():  # str.isdigit() also accepts "²", which int() rejects
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


# --- The time of the user's last turn -----------------------------------------------------------------------------
# The one place it is computed (``days_since_human`` and the user-presence silence both call it). It reads the
# conversation files and nothing else: no memory, no presence file. A reader must not write, so neither step creates a
# folder (``_active_conversations_dir`` and ``_archived_conversations_dir`` both do).

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _latest_user_ts_in_file(path: Path) -> datetime | None:
    """The latest ``ts`` of a ``speaker == "user"`` row of one conversation file, read whole, or None.

    A row whose ``ts`` does not parse is skipped with a warning, like a bad line. A line the JSON parser cannot load
    (``UnloadableJsonlLineError``, #646) ends the read of THIS file with a warning: the rows before it count, the rows after
    it are not seen. A file that cannot be read reads as empty (``iter_jsonl`` warns)."""
    latest: datetime | None = None
    try:
        for row in iter_jsonl(path):
            if row.get("speaker") != "user":
                continue
            ts = lenient_utc(row.get("ts"))
            if ts is None:
                warn_throttled(
                    logger, "last_user_turn_ts:bad_ts", "last_user_turn_ts: a user row of %s has no usable ts", path.name,
                    exc_info=False,
                )
                continue
            if latest is None or ts > latest:
                latest = ts
    except UnloadableJsonlLineError:
        warn_throttled(
            logger, "last_user_turn_ts:unloadable",
            "last_user_turn_ts: %s has a line the JSON parser cannot load; the rest of that file is not read", path.name,
        )
    return latest


def _latest_user_ts_in_active_files(persona_dir: Path) -> datetime | None:
    try:
        session_ids = list_active_sessions(persona_dir)
    except OSError:
        warn_throttled(
            logger, "last_user_turn_ts:unlistable_active", "last_user_turn_ts: the active conversations folder cannot be listed"
        )
        return None
    latest: datetime | None = None
    for session_id in session_ids:
        try:
            path = _session_path(persona_dir, session_id)  # the folder exists (it was just listed): nothing is created
        except ValueError:
            continue  # a file name that is not a session id is not a session buffer
        ts = _latest_user_ts_in_file(path)
        if ts is not None and (latest is None or ts > latest):
            latest = ts
    return latest


def _latest_user_ts_in_session_archive(persona_dir: Path, session_id: str) -> datetime | None:
    """The latest user ``ts`` of one archived session: its files from the last backwards, stopping at the first that holds one."""
    try:
        files = _archive_read_order(persona_dir, session_id)
    except OSError:
        warn_throttled(
            logger, "last_user_turn_ts:unlistable_session", "last_user_turn_ts: the archived files of session %s cannot be listed",
            session_id,
        )
        return None
    for path in reversed(files):
        ts = _latest_user_ts_in_file(path)
        if ts is not None:
            return ts
    return None


def _latest_user_ts_in_archive(persona_dir: Path) -> datetime | None:
    # Not ``_archived_conversations_dir``: it creates the folder. A missing folder lists as [].
    folder = persona_dir / "archived_conversations"
    try:
        names = list_dir(folder)
    except OSError:
        warn_throttled(
            logger, "last_user_turn_ts:unlistable_archive", "last_user_turn_ts: the archived conversations folder cannot be listed"
        )
        return None
    # The session ids come from the names (``<sid>.jsonl`` legacy, ``<sid>.<NNN>.jsonl`` segments; a session id holds no
    # dot). Each session's last file is read off this one listing: its largest numbered segment, else the legacy file.
    last_file: dict[str, tuple[int, str]] = {}
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        session_id, _, segment = name[: -len(".jsonl")].partition(".")
        if not _SESSION_ID_RE.fullmatch(session_id):
            continue
        if not segment:
            rank = -1  # the legacy file comes before every numbered segment
        elif segment.isascii() and segment.isdigit():
            rank = int(segment)
        else:
            continue
        if session_id not in last_file or (rank, name) > last_file[session_id]:
            last_file[session_id] = (rank, name)
    # Session ids are not in time order, so the sessions are visited newest file first by modification time (a file's
    # last row cannot be later than the time the file was last written).
    visits: list[tuple[datetime, str]] = []
    for session_id, (_, name) in last_file.items():
        try:
            stat = stat_path(folder / name)
        except OSError:
            warn_throttled(
                logger, "last_user_turn_ts:stat", "last_user_turn_ts: the archived file %s cannot be inspected; skipped", name
            )
            continue
        if stat is None or not stat.is_file:  # gone since the listing, or not a file
            continue
        visits.append((_EPOCH + timedelta(microseconds=stat.mtime_ns // 1000), session_id))
    visits.sort(reverse=True)
    best: datetime | None = None
    for mtime, session_id in visits:
        if best is not None and best >= mtime:
            break  # no later session can hold a turn newer than the best found, so no further file is opened
        ts = _latest_user_ts_in_session_archive(persona_dir, session_id)
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def last_user_turn_ts(persona_dir: Path) -> datetime | None:
    """The ``ts`` (aware, UTC) of the user's last turn, from the conversation files, or None when no file holds one.

    Active step: every ``active_conversations/*.jsonl`` read whole (a file holds about the newest forty raw turns), the
    latest ``speaker == "user"`` row's ``ts`` across them. Archive step, only when no active file holds a user turn (after a
    full fold the successor holds only a summary row): the archived sessions visited newest file first, each read from its
    last file backwards to the first that holds a user turn, the walk ending as soon as the best ``ts`` found is not older
    than the next session's last-file modification time, so the read is bounded with no window constant. The answer is the
    latest ``ts`` across the sessions visited. A naive ``ts`` means UTC. Reads only: no folder is created.
    """
    latest = _latest_user_ts_in_active_files(persona_dir)
    if latest is not None:
        return latest
    return _latest_user_ts_in_archive(persona_dir)


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


def turn_identity(t: dict) -> tuple:
    """A stable identity for a buffer turn, used to match archived turns against
    a re-read buffer (race-safe rewrite). (ts, speaker, text) is unique enough;
    a collision means two byte-identical turns in the same second, where keeping
    or dropping one is harmless."""
    return (t.get("ts"), t.get("speaker"), t.get("text"))


def _archive_batch_key(records: list[dict], new_gen: int) -> str:
    """A stable identity for one archive-append batch: sha256 over ``new_gen`` and
    the ORDERED ``(ts, speaker, text)`` identity of every record — never any
    generated fold text. Two invocations that recompute the identical batch (a
    crash-retry over an unchanged buffer) hash to the SAME key; a later,
    genuinely-different batch (different ``new_gen`` or different records) never
    collides (Bug 3)."""
    h = hashlib.sha256()
    h.update(str(new_gen).encode("utf-8"))
    for r in records:
        h.update(repr(turn_identity(r)).encode("utf-8"))
    return h.hexdigest()


def archive_once(persona_dir: Path, session_id: str, records: list[dict], *, new_gen: int) -> int:
    """The one archive-with-dedup write: archive ``records`` exactly once for this batch, proven by a
    durable commit marker (Bug 3). Compaction, the cascade and the session rollover archive the turns they
    remove from a conversation buffer through this, before they remove anything.

    ``new_gen`` is the generation the batch belongs to (the summary generation being written, or the one
    the rollover carries); it is part of the batch key. Skips the append ONLY when the marker key matches
    — i.e. the marker PROVES a completed prior append of this exact batch — never by inferring from turn
    identity, so two byte-identical raw turns are always both archived (or both re-archived on a true
    partial-write retry), never silently collapsed (L2). Returns the byte count written, or ``1`` as a
    skip sentinel (a proven-already-archived batch; still ``> 0`` for the caller's write-verify). An
    empty batch writes nothing and returns 0. A crash between the append and the marker write repeats
    the batch on the retry (repeats, never a missing row)."""
    if not records:
        return 0
    key = _archive_batch_key(records, new_gen)
    if read_archive_marker(persona_dir, session_id) == key:
        return 1  # proven already archived this pass -> skip append (no dup, no loss)
    n = append_archive(persona_dir, session_id, records)
    write_archive_marker(persona_dir, session_id, key)
    return n


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
