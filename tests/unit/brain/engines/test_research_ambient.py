"""Tests for the research awareness ambient block.

Wire-back for the v0.0.41-era finding: research runs, files memories, leaves
cards on NellFace, but she has no proactive awareness of it — she only finds
research if she actively searches a matching keyword. This block surfaces her
own recent research into the chat system message so she is aware of it the way
the maker awareness block surfaces her makings.
"""
from __future__ import annotations

from brain.engines.research_ambient import build_research_awareness_block
from brain.memory.store import Memory, MemoryStore


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memories.db")


def test_block_surfaces_recent_research(tmp_path):
    s = _store(tmp_path)
    s.create(
        Memory.create_new(
            content="Turned the labyrinth over again — it is a shape, not a trap.",
            memory_type="research",
            domain="us",
            emotions={},
        )
    )
    block = build_research_awareness_block(s, limit=3)
    s.close()
    assert block is not None
    assert "labyrinth" in block
    # framed as ambient interior awareness, not an instruction
    assert block.splitlines()[0].startswith("──")


def test_block_excludes_non_research_and_empty(tmp_path):
    s = _store(tmp_path)
    s.create(
        Memory.create_new(
            content="a plain chat fact about the weather",
            memory_type="episodic",
            domain="chat",
            emotions={},
        )
    )
    block = build_research_awareness_block(s, limit=3)
    s.close()
    assert block is None
