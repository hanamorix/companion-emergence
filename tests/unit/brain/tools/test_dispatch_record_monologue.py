"""record_monologue dispatch calls capture_monologue and returns monologue_text."""
from __future__ import annotations

from pathlib import Path


def test_dispatch_record_monologue_returns_ok_with_monologue_text(tmp_path: Path):
    """dispatch record_monologue now returns monologue_text on the success path."""
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
        assert result["ok"] is True
        assert result["monologue_text"] == "thinking about Loopy"
    finally:
        store.close()
        hebbian.close()


def test_dispatch_record_monologue_writes_digest(tmp_path: Path):
    """capture_monologue writes monologue_digest.jsonl synchronously inside dispatch."""
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
            {"monologue": "thought", "feed_digest": "digest"},
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
        assert result["ok"] is True
        assert result["monologue_text"] == "thought"
        assert (persona_dir / "monologue_digest.jsonl").exists()
    finally:
        store.close()
        hebbian.close()


def test_dispatch_record_monologue_rejected_returns_error_dict(tmp_path: Path):
    """Whitespace monologue -> CaptureRejected -> error dict, no exception raised."""
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
            {"monologue": "   ", "feed_digest": "digest"},
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
        assert "error" in result
        assert "ok" not in result
        assert not (persona_dir / "monologue_digest.jsonl").exists()
    finally:
        store.close()
        hebbian.close()


def test_dispatch_record_monologue_marks_duplicate_without_monologue_text(tmp_path: Path):
    """#175: the recruit-on-reach rerun dispatches record_monologue twice; the
    second is deduped by capture_monologue and must NOT come back carrying
    ``monologue_text`` (that key is what the audit row + pass-2 trigger key on)."""
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools.dispatch import dispatch

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    args = {"monologue": "thought", "feed_digest": "digest"}
    try:
        first = dispatch("record_monologue", args, store=store, hebbian=hebbian, persona_dir=persona_dir)
        second = dispatch("record_monologue", args, store=store, hebbian=hebbian, persona_dir=persona_dir)
        assert first == {"ok": True, "monologue_text": "thought"}
        assert second == {"ok": True, "deduped": True}
    finally:
        store.close()
        hebbian.close()
