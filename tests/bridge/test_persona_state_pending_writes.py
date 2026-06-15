from datetime import UTC, datetime

from brain.bridge.persona_state import _build_pending_writes
from brain.files import pending


def test_pending_writes_surfaced_excludes_resolved(tmp_path):
    a = pending.create(
        tmp_path, op="create", resolved_path="/x/a.md", content="A" * 50, now=datetime.now(UTC)
    )
    b = pending.create(
        tmp_path, op="append", resolved_path="/x/b.md", content="B", now=datetime.now(UTC)
    )
    pending.mark(tmp_path, b, status="committed")
    out = _build_pending_writes(tmp_path)
    ids = {r["id"] for r in out}
    assert a in ids
    assert b not in ids
    assert out[0]["op"] in ("create", "append")
    assert "preview" in out[0]  # truncated content preview
