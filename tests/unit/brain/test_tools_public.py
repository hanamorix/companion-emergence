"""Tests for brain.tools public surface — NELL_TOOL_NAMES export."""

from __future__ import annotations


def test_nell_tool_names_exported() -> None:
    from brain.tools import NELL_TOOL_NAMES

    assert isinstance(NELL_TOOL_NAMES, tuple)
    assert len(NELL_TOOL_NAMES) > 0
    # Canonical tools from spec §1 (brain core)
    assert "search_memories" in NELL_TOOL_NAMES
    assert "get_emotional_state" in NELL_TOOL_NAMES
    assert "get_soul" in NELL_TOOL_NAMES
    assert "get_personality" in NELL_TOOL_NAMES
    assert "get_body_state" in NELL_TOOL_NAMES
    assert "boot" in NELL_TOOL_NAMES
    assert "add_journal" in NELL_TOOL_NAMES
    assert "add_memory" in NELL_TOOL_NAMES
    assert "crystallize_soul" in NELL_TOOL_NAMES
    # works tools added in Task 3
    assert "save_work" in NELL_TOOL_NAMES
    assert "list_works" in NELL_TOOL_NAMES
    assert "search_works" in NELL_TOOL_NAMES
    assert "read_work" in NELL_TOOL_NAMES


def test_tool_loop_imports_from_brain_tools() -> None:
    """tool_loop must use the public name — no private fallback."""
    from brain.chat import tool_loop
    from brain.tools import NELL_TOOL_NAMES

    # build_tools_list iterates the same names — easiest assertion is to
    # call it and confirm shape matches NELL_TOOL_NAMES
    tools = tool_loop.build_tools_list()
    names_in_tools = {t["function"]["name"] for t in tools}
    # SCHEMAS gates which names actually appear; intersect with NELL_TOOL_NAMES
    from brain.tools.schemas import SCHEMAS

    expected = {n for n in NELL_TOOL_NAMES if n in SCHEMAS}
    assert names_in_tools == expected
