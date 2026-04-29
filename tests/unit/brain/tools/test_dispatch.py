"""Tests for brain/tools/dispatch.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.tools.dispatch import _DISPATCH, ToolDispatchError, dispatch


def _make_ctx(tmp_path: Path) -> dict:
    """Build a minimal dispatch context with in-memory store + hebbian."""
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    return {"store": store, "hebbian": hebbian, "persona_dir": tmp_path}


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_unknown_tool_raises_dispatch_error(tmp_path: Path) -> None:
    """Unknown tool name raises ToolDispatchError."""
    ctx = _make_ctx(tmp_path)
    with pytest.raises(ToolDispatchError, match="unknown tool"):
        dispatch("no_such_tool", {}, **ctx)


def test_missing_required_arg_raises_dispatch_error(tmp_path: Path) -> None:
    """Missing required arg for add_memory raises ToolDispatchError."""
    ctx = _make_ctx(tmp_path)
    # add_memory requires: content, memory_type, domain, emotions
    with pytest.raises(ToolDispatchError, match="missing required argument"):
        dispatch("add_memory", {"content": "hello"}, **ctx)


def test_wrong_type_emotions_raises_dispatch_error(tmp_path: Path) -> None:
    """emotions not a dict raises ToolDispatchError on add_memory."""
    ctx = _make_ctx(tmp_path)
    with pytest.raises(ToolDispatchError, match="must be a dict"):
        dispatch(
            "add_memory",
            {
                "content": "test",
                "memory_type": "event",
                "domain": "self",
                "emotions": "love:9",  # string instead of dict
            },
            **ctx,
        )


def test_add_journal_missing_content_raises_dispatch_error(tmp_path: Path) -> None:
    """add_journal with missing content raises ToolDispatchError."""
    ctx = _make_ctx(tmp_path)
    with pytest.raises(ToolDispatchError, match="missing required argument"):
        dispatch("add_journal", {}, **ctx)


# ---------------------------------------------------------------------------
# Successful dispatch
# ---------------------------------------------------------------------------


def test_dispatch_add_journal_returns_dict(tmp_path: Path) -> None:
    """Successful dispatch returns the impl's dict."""
    ctx = _make_ctx(tmp_path)
    result = dispatch("add_journal", {"content": "test entry"}, **ctx)
    assert isinstance(result, dict)
    assert "created_id" in result
    assert result["memory_type"] == "journal_entry"


def test_dispatch_get_emotional_state_returns_dict(tmp_path: Path) -> None:
    """get_emotional_state dispatches and returns structured result."""
    ctx = _make_ctx(tmp_path)
    result = dispatch("get_emotional_state", {}, **ctx)
    assert isinstance(result, dict)
    assert "dominant" in result
    assert "top_5" in result


def test_dispatch_boot_returns_composition(tmp_path: Path) -> None:
    """boot returns a dict with all 5 composition keys."""
    ctx = _make_ctx(tmp_path)
    result = dispatch("boot", {}, **ctx)
    assert isinstance(result, dict)
    assert "emotional_state" in result
    assert "personality" in result
    assert "soul" in result
    assert "body_state" in result
    assert "context_prose" in result


def test_dispatch_get_soul_returns_real_shape(tmp_path: Path) -> None:
    """get_soul returns real shape (SP-5 live)."""
    ctx = _make_ctx(tmp_path)
    result = dispatch("get_soul", {}, **ctx)
    assert result["loaded"] is True
    assert "crystallizations" in result
    assert "count" in result


def test_dispatch_crystallize_soul_creates_crystallization(tmp_path: Path) -> None:
    """crystallize_soul creates a real crystallization (SP-5 live)."""
    ctx = _make_ctx(tmp_path)
    result = dispatch(
        "crystallize_soul",
        {
            "moment": "a quiet moment",
            "love_type": "romantic",
            "why_it_matters": "it was real",
        },
        **ctx,
    )
    assert result["created"] is True
    assert "id" in result
    assert result["love_type"] == "romantic"


# ---------------------------------------------------------------------------
# All 9 tools smoke-test
# ---------------------------------------------------------------------------


def test_all_nine_tools_dispatch_without_crash(tmp_path: Path) -> None:
    """All 9 registered tool names dispatch without raising."""
    ctx = _make_ctx(tmp_path)

    # Map of tool → minimal valid arguments
    minimal_args: dict[str, dict] = {
        "get_emotional_state": {},
        "get_personality": {},
        "get_body_state": {},
        "search_memories": {"query": "test"},
        "add_journal": {"content": "journal entry"},
        "add_memory": {
            "content": "significant moment",
            "memory_type": "event",
            "domain": "self",
            "emotions": {"love": 10, "joy": 8},
        },
        "boot": {},
        "get_soul": {},
        "crystallize_soul": {
            "moment": "a moment",
            "love_type": "craft",
            "why_it_matters": "it mattered",
        },
    }

    for tool_name in _DISPATCH:
        args = minimal_args.get(tool_name, {})
        result = dispatch(tool_name, args, **ctx)
        assert isinstance(result, dict), f"{tool_name} did not return a dict"
