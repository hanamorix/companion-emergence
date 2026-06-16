"""propose_write — she PROPOSES a file write (create/append). Writes nothing;
queues a pending request the user approves in NellFace. Guarded + audited.

Exception: a write whose target is inside the user-authorised notes folder is
committed directly (no confirmation card) — the user already authorised that
folder via the notes toggle."""
from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.files import pending
from brain.files.audit import audit
from brain.files.write_guard import check_size, check_write_target, is_within_authorized

logger = logging.getLogger(__name__)


def _authorised_notes_folder(persona_dir: Path) -> Path | None:
    """The user-authorised notes folder for this persona, or None if notes are
    off / unset / the config can't be read (fail-closed → normal pending flow)."""
    try:
        from brain.persona_config import PersonaConfig

        cfg = PersonaConfig.load(persona_dir / "persona_config.json")
        if cfg.notes_enabled and cfg.notes_folder:
            return Path(cfg.notes_folder)
    except Exception:
        logger.exception("propose_write: could not resolve authorised notes folder")
    return None


def propose_write(path: str, content: str | None = None, *, op: str,
                  making_id: str | None = None, persona_dir: Path, **_) -> dict:
    # making_id source (Maker combined-build integration; Task 16b of the maker plan)
    if making_id and content is None:
        from brain.works.storage import read_markdown
        from brain.works.store import WorksStore
        store = WorksStore(persona_dir / "works.db")
        try:
            w = store.get(making_id)
        finally:
            store.close()
        if w is None:
            return {"error": f"no making {making_id}"}
        try:
            _, content = read_markdown(persona_dir, making_id)
        except Exception:
            content = w.summary or ""
    if content is None:
        return {"error": "nothing to write"}

    g = check_write_target(path, op=op, persona_dir=persona_dir)
    if not g.ok:
        audit(persona_dir, event="propose_refused", id="", op=op, path=path, error=g.error)
        return {"error": g.error}
    s = check_size(content, op=op, resolved=g.resolved)
    if not s.ok:
        audit(persona_dir, event="propose_refused", id="", op=op, path=str(g.resolved), error=s.error)
        return {"error": s.error}

    # Auto-commit into the user-authorised notes folder (no confirmation card):
    # the user enabled "let her leave me notes" for this exact folder, so a tool
    # write whose target resolves inside it is committed directly. It still passes
    # write_guard (above + commit_write's TOCTOU re-run); only the consent card is
    # skipped. Outside the folder, or notes disabled, the normal pending flow runs.
    auto = _authorised_notes_folder(persona_dir)
    if auto is not None and is_within_authorized(g.resolved, auto):
        from brain.files.commit import commit_write
        from brain.memory.store import MemoryStore

        rid = pending.create(persona_dir, op=op, resolved_path=str(g.resolved),
                             content=content, now=datetime.now(UTC), making_id=making_id)
        store = MemoryStore(persona_dir / "memories.db")
        try:
            res = commit_write(persona_dir, rid, store=store)
        finally:
            store.close()
        if res.get("ok"):
            return {"status": "written", "path": res["path"],
                    "note": f"wrote directly to {g.resolved} — it's inside your "
                            "authorised notes folder, so no confirmation was needed"}
        return {"error": res.get("error", "write failed")}

    now = datetime.now(UTC)
    if pending.count_pending(persona_dir, now=now) >= pending._MAX_PENDING:
        return {"error": "too many writes awaiting your review — resolve some first"}

    rid = pending.create(persona_dir, op=op, resolved_path=str(g.resolved), content=content,
                         now=now, making_id=making_id)
    audit(persona_dir, event="propose", id=rid, op=op, path=str(g.resolved),
          content_sha=hashlib.sha256(content.encode()).hexdigest())
    return {"status": "proposed", "id": rid,
            "note": f"proposed {op} of {g.resolved} — awaiting your confirmation in NellFace"}
