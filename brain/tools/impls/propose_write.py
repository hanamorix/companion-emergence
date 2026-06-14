"""propose_write — she PROPOSES a file write (create/append). Writes nothing;
queues a pending request the user approves in NellFace. Guarded + audited."""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from brain.files import pending
from brain.files.audit import audit
from brain.files.write_guard import check_size, check_write_target


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

    now = datetime.now(UTC)
    if pending.count_pending(persona_dir, now=now) >= pending._MAX_PENDING:
        return {"error": "too many writes awaiting your review — resolve some first"}

    rid = pending.create(persona_dir, op=op, resolved_path=str(g.resolved), content=content,
                         now=now, making_id=making_id)
    audit(persona_dir, event="propose", id=rid, op=op, path=str(g.resolved),
          content_sha=hashlib.sha256(content.encode()).hexdigest())
    return {"status": "proposed", "id": rid,
            "note": f"proposed {op} of {g.resolved} — awaiting your confirmation in NellFace"}
