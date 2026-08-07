# brain/files/pending.py
"""Pending-write store — one JSON per request under persona_dir/pending_writes/."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR = "pending_writes"
_TTL_HOURS = 24.0
_MAX_PENDING = 10


def _dir(persona_dir: Path) -> Path:
    d = persona_dir / _DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _new_id(resolved_path: str, content: str, now: datetime) -> str:
    h = hashlib.sha256(f"{resolved_path}{now.isoformat()}{content[:64]}".encode()).hexdigest()
    return h[:12]


def count_pending(persona_dir: Path, *, now: datetime) -> int:
    return len(list_pending(persona_dir, now=now))


def create(persona_dir: Path, *, op: str, resolved_path: str, content: str,
           now: datetime, making_id: str | None = None) -> str:
    rid = _new_id(resolved_path, content, now)
    rec = {"id": rid, "op": op, "resolved_path": resolved_path, "content": content,
           "content_sha": hashlib.sha256(content.encode()).hexdigest(),
           "proposed_at": now.isoformat(), "status": "pending", "making_id": making_id}
    p = _dir(persona_dir) / f"{rid}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(p)
    return rid


def get(persona_dir: Path, rid: str) -> dict | None:
    p = _dir(persona_dir) / f"{rid}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _all(persona_dir: Path) -> list[dict]:
    out = []
    for f in _dir(persona_dir).glob("*.json"):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def list_pending(persona_dir: Path, *, now: datetime) -> list[dict]:
    fresh = []
    for r in _all(persona_dir):
        if r.get("status") != "pending":
            continue
        try:
            age = now - datetime.fromisoformat(r["proposed_at"])
        except (ValueError, KeyError):
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


def mark(persona_dir: Path, rid: str, *, status: str) -> None:
    rec = get(persona_dir, rid)
    if rec is None:
        return
    rec["status"] = status
    p = _dir(persona_dir) / f"{rid}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(p)


def sweep_expired(persona_dir: Path, *, now: datetime) -> int:
    n = 0
    for r in _all(persona_dir):
        if r.get("status") != "pending":
            continue
        try:
            age = now - datetime.fromisoformat(r["proposed_at"])
        except (ValueError, KeyError):
            continue
        if age > timedelta(hours=_TTL_HOURS):
            mark(persona_dir, r["id"], status="expired")
            n += 1
    return n
