"""brain.files.commit — perform an approved write (guard re-run) or decline.
Wires a file_write memory + feed event either way."""
from __future__ import annotations

import logging
from pathlib import Path

from brain.files import pending
from brain.files.audit import audit
from brain.files.write_guard import check_size, check_write_target

logger = logging.getLogger(__name__)


def _wire_memory(store, *, path: str, outcome: str) -> None:
    from brain.memory.store import Memory

    if outcome == "committed":
        content = f"you let me write to {path}"
    else:
        content = f"you declined my write to {path}"  # no file content stored
    try:
        store.create(
            Memory.create_new(
                content=content,
                memory_type="file_write",
                domain="interior",
                tags=["file_write", outcome],
            )
        )
    except Exception:
        logger.exception("file_write wire-back memory failed")


def commit_write(persona_dir: Path, rid: str, *, store) -> dict:
    rec = pending.get(persona_dir, rid)
    if rec is None or rec.get("status") != "pending":
        return {"ok": False, "error": "not a pending write"}
    op, content = rec["op"], rec["content"]
    # TOCTOU: re-run the guard on the resolved path RIGHT NOW.
    g = check_write_target(rec["resolved_path"], op=op, persona_dir=persona_dir)
    if not g.ok or not check_size(content, op=op, resolved=g.resolved).ok:
        err = g.error or "size check failed"
        pending.mark(persona_dir, rid, status="refused")
        audit(
            persona_dir,
            event="commit_refused",
            id=rid,
            op=op,
            path=rec["resolved_path"],
            error=err,
        )
        return {"ok": False, "error": err}
    try:
        g.resolved.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if op == "create" else "a"
        with g.resolved.open(mode, encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        pending.mark(persona_dir, rid, status="error")
        audit(persona_dir, event="error", id=rid, op=op, path=str(g.resolved), error=str(exc))
        return {"ok": False, "error": str(exc)}
    pending.mark(persona_dir, rid, status="committed")
    audit(
        persona_dir,
        event="commit",
        id=rid,
        op=op,
        path=str(g.resolved),
        content_sha=rec["content_sha"],
        outcome="committed",
    )
    _wire_memory(store, path=str(g.resolved), outcome="committed")
    return {"ok": True, "path": str(g.resolved)}


def decline_write(persona_dir: Path, rid: str, *, store) -> dict:
    rec = pending.get(persona_dir, rid)
    if rec is None or rec.get("status") != "pending":
        return {"ok": False, "error": "not a pending write"}
    pending.mark(persona_dir, rid, status="declined")
    audit(
        persona_dir,
        event="decline",
        id=rid,
        op=rec["op"],
        path=rec["resolved_path"],
        outcome="declined",
    )
    _wire_memory(store, path=rec["resolved_path"], outcome="declined")
    return {"ok": True}
