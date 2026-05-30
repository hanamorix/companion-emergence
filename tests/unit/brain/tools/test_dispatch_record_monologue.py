"""record_monologue dispatch returns a noop tool result — real capture happens upstream."""
from __future__ import annotations

from pathlib import Path


def test_dispatch_record_monologue_returns_ok(tmp_path: Path):
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools.dispatch import dispatch

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        result = dispatch(
            "record_monologue",
            {"monologue": "thinking about Loopy", "feed_digest": "she searched and felt fond"},
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
        assert result == {"ok": True}
    finally:
        store.close()
        hebbian.close()
