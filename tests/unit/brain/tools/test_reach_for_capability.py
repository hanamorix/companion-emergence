"""Tests for brain/tools/impls/reach_for_capability.py — agency safety-valve."""
from __future__ import annotations

from pathlib import Path

from brain.tools.impls.reach_for_capability import reach_for_capability


def test_returns_requested_capability(tmp_path: Path) -> None:
    out = reach_for_capability(capability="memory", persona_dir=tmp_path)
    assert out["recruited"] == "memory" and out["ok"] is True


def test_registered_and_dispatchable(tmp_path: Path) -> None:
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools import NELL_TOOL_NAMES, dispatch
    from brain.tools.schemas import SCHEMAS

    assert "reach_for_capability" in NELL_TOOL_NAMES
    assert "reach_for_capability" in SCHEMAS
    out = dispatch(
        "reach_for_capability",
        {"capability": "files"},
        store=MemoryStore(":memory:"),
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=tmp_path,
    )
    assert out["recruited"] == "files"
