# brain/files/pending.py
"""Pending-write store — one JSON per request under persona_dir/pending_writes/."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

from brain.health.json_io import read_json, write_json
from brain.utils.file_lock import file_lock
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

_DIR = "pending_writes"
_TTL_HOURS = 24.0
_MAX_PENDING = 10


def _dir(persona_dir: Path) -> Path:
    d = persona_dir / _DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lock(persona_dir: Path):
    """One store-wide cross-process lock (a single sidecar, not one per record).

    Serialises every read-modify-replace of a record across the bridge, the supervisor thread
    and the MCP subprocess, so `mark(expect=...)` is a real compare-and-set (#346). Volume is a
    handful of records, so one lock costs nothing.
    """
    return file_lock(_dir(persona_dir) / ".records")


def _new_id(resolved_path: str, content: str, now: datetime) -> str:
    h = hashlib.sha256(f"{resolved_path}{now.isoformat()}{content[:64]}".encode()).hexdigest()
    return h[:12]


def is_expired(rec: dict, now: datetime) -> bool:
    """True if a pending record is older than the TTL (whether or not the sweep has run yet)."""
    try:
        return now - datetime.fromisoformat(rec["proposed_at"]) > timedelta(hours=_TTL_HOURS)
    except (ValueError, KeyError, TypeError):
        return False


def count_pending(persona_dir: Path, *, now: datetime) -> int:
    return len(list_pending(persona_dir, now=now))


def create(persona_dir: Path, *, op: str, resolved_path: str, content: str,
           now: datetime, making_id: str | None = None) -> str:
    rid = _new_id(resolved_path, content, now)
    rec = {"id": rid, "op": op, "resolved_path": resolved_path, "content": content,
           "content_sha": hashlib.sha256(content.encode()).hexdigest(),
           "proposed_at": now.isoformat(), "status": "pending", "making_id": making_id}
    with _lock(persona_dir):
        write_json(_dir(persona_dir) / f"{rid}.json", rec, indent=None, fsync=False)
    return rid


def _read(p: Path) -> dict | None:
    """One record, or None. A missing record is normal; an unreadable or non-record one is reported once
    per interval by brain.health.json_io (#367), since it drops out of every listing. Writers replace the
    file atomically, so a reader never sees a half-written record; on Windows the module retries a transient
    PermissionError while a replace runs before it reports."""
    return read_json(p, default=lambda: None, expect=dict)


def get(persona_dir: Path, rid: str) -> dict | None:
    return _read(_dir(persona_dir) / f"{rid}.json")


def _all(persona_dir: Path) -> list[dict]:
    return [r for f in _dir(persona_dir).glob("*.json") if (r := _read(f)) is not None]


def _age(r: dict, now: datetime) -> timedelta | None:
    """How old a record is, or None. A record without a usable ``proposed_at``
    drops out of every listing and is never expired, so it is reported (#429)."""
    try:
        return now - datetime.fromisoformat(r["proposed_at"])
    except (ValueError, KeyError, TypeError):  # TypeError: null, or a tz-naive stamp
        warn_throttled(logger, f"pending:{r.get('id')}", "pending write %s has no usable proposed_at; skipped",
                       r.get("id"), exc_info=False)
        return None


def list_pending(persona_dir: Path, *, now: datetime) -> list[dict]:
    fresh = []
    for r in _all(persona_dir):
        if r.get("status") != "pending":
            continue
        if (age := _age(r, now)) is None:
            continue
        if age <= timedelta(hours=_TTL_HOURS):
            fresh.append(r)
    return sorted(fresh, key=lambda r: r["proposed_at"], reverse=True)


def find_duplicate(persona_dir: Path, *, op: str, resolved_path: str,
                   content_sha: str, now: datetime) -> str | None:
    """Id of a fresh pending record proposing this exact write, or None.

    The record id is not a dedupe key — _new_id mixes in `now`, so two
    identical proposals moments apart mint different ids and both persist
    (#93/#101). The dedupe key is (op, resolved_path, content_sha).

    Matches against list_pending, which already filters to status="pending"
    AND within _TTL_HOURS: a record still marked pending but past the TTL has
    merely not been swept yet, and must not suppress a legitimate proposal.
    """
    for r in list_pending(persona_dir, now=now):
        if (r.get("op") == op
                and r.get("resolved_path") == resolved_path
                and r.get("content_sha") == content_sha):
            return r["id"]
    return None


def list_by_status(persona_dir: Path, status: str) -> list[dict]:
    """Every record currently in `status`, oldest-unspecified order (callers sort)."""
    return [r for r in _all(persona_dir) if r.get("status") == status]


def mark(persona_dir: Path, rid: str, *, status: str, expect: str | None = None,
         **extra) -> bool:
    """Set the record's status (and merge `extra` fields). True if it took, False if not.

    Returning a bool rather than failing mutely is load-bearing (#101): get()
    swallows OSError/ValueError, so a transient read failure made this a silent
    no-op. commit_write then wrote the file while the record stayed 'pending',
    and its status guard — the thing making the commit idempotent — stopped
    guarding, so a retry appended the block a second time.

    `expect` makes this a compare-and-set (#344): the status is re-read HERE and
    nothing is written unless it equals `expect`. The read and the replace run under
    one store-wide cross-process lock (#346), so two claimers cannot both win and the
    shared `.tmp` can't tear. Readers (`get`/`_all`) stay lock-free: on POSIX `replace` is atomic;
    on Windows it can raise PermissionError while a lock-free reader has the file open. Every
    caller already treats a failed mark as 'did not take' (bool / OSError handled), so that recovers.
    """
    with _lock(persona_dir):
        rec = get(persona_dir, rid)
        if rec is None:
            return False
        if expect is not None and rec.get("status") != expect:
            return False
        rec["status"] = status
        rec.update(extra)
        write_json(_dir(persona_dir) / f"{rid}.json", rec, indent=None, fsync=False)
        return True


def sweep_expired(persona_dir: Path, *, now: datetime) -> int:
    n = 0
    for r in _all(persona_dir):
        if r.get("status") != "pending":
            continue
        if (age := _age(r, now)) is None:
            continue
        if age > timedelta(hours=_TTL_HOURS):
            # expect='pending': a card claimed since the read above is live, not expired (#344).
            if mark(persona_dir, r["id"], status="expired", expect="pending"):
                n += 1
    return n
