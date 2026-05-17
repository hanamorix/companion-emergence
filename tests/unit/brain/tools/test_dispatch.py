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


def test_unknown_tool_error_does_not_enumerate_known_tools(tmp_path: Path) -> None:
    """M-10: dispatch's unknown-tool error must NOT include the full list of
    known tool names. The error round-trips into the LLM's next context via
    tool_loop, leaking the canonical tool surface as training signal."""
    ctx = _make_ctx(tmp_path)
    try:
        dispatch("no_such_tool", {}, **ctx)
    except ToolDispatchError as exc:
        msg = str(exc)
        # Some sentinel tool names that should NOT appear:
        for known in ("search_memories", "get_emotional_state", "save_work", "boot"):
            assert known not in msg, f"leaked known tool name {known!r} in error"


def test_get_body_state_does_not_mutate_caller_arguments(tmp_path: Path) -> None:
    """M-4: dispatch's session_hours float coercion must not mutate the
    caller's arguments dict. The chat tool loop logs `arguments` straight
    into the invocations record; mutating it bleeds the float coercion
    into the audit trail."""
    ctx = _make_ctx(tmp_path)
    args = {"session_hours": "1.5"}
    args_id = id(args)
    args_snapshot = dict(args)

    # The dispatch may succeed or raise depending on impl — what matters is
    # that the caller's dict is unchanged either way.
    try:
        dispatch("get_body_state", args, **ctx)
    except Exception:  # noqa: BLE001
        pass

    assert id(args) == args_id, "caller's dict identity changed"
    assert args == args_snapshot, f"caller's dict mutated from {args_snapshot} to {args}"
    # Explicitly: session_hours stays a string in the caller's record
    assert args["session_hours"] == "1.5"
    assert isinstance(args["session_hours"], str)


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
# get_body_state — dispatcher session_hours injection
# (Bug surfaced 2026-05-17: MCP tool path returned session_hours=0.0,
#  energy=7, exhaustion=0 even hours into a session, because the
#  dispatcher claimed to inject session_hours but never did.)
# ---------------------------------------------------------------------------


def _seed_active_buffer(persona_dir: Path, *, hours_old: float) -> None:
    """Plant an active_conversations buffer whose first entry is `hours_old` ago."""
    import json
    from datetime import UTC, datetime, timedelta

    conv_dir = persona_dir / "active_conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC) - timedelta(hours=hours_old)
    entry = {
        "timestamp": started_at.isoformat(),
        "role": "user",
        "content": "session opener",
    }
    (conv_dir / "test-session.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")


def test_get_body_state_dispatcher_injects_session_hours_from_active_buffer(
    tmp_path: Path,
) -> None:
    """When the LLM calls get_body_state with no session_hours, the dispatcher
    must compute it from the active conversation buffer — matching what the
    UI's /persona/state path already does. Before this fix the MCP tool path
    returned session_hours=0.0 (fresh-persona defaults) even hours into a
    session, so the brain's self-read disagreed with what the panel showed."""
    ctx = _make_ctx(tmp_path)
    _seed_active_buffer(tmp_path, hours_old=3.0)

    result = dispatch("get_body_state", {}, **ctx)

    # Returned body's session_hours must reflect the live buffer age, not
    # the function-default 0.0. Allow ±0.1h tolerance (test runtime, ISO
    # parsing rounding).
    assert result["session_hours"] >= 2.9, (
        f"dispatcher failed to inject session_hours; got "
        f"session_hours={result['session_hours']!r} "
        f"(expected ~3.0 from 3h-old active buffer)"
    )


def test_get_body_state_caller_session_hours_wins_over_dispatcher_injection(
    tmp_path: Path,
) -> None:
    """Explicit caller-provided session_hours is NOT overwritten by the
    dispatcher's injection. Honors any CLI path or tool-loop caller that
    knows its own session age."""
    ctx = _make_ctx(tmp_path)
    _seed_active_buffer(tmp_path, hours_old=3.0)

    result = dispatch("get_body_state", {"session_hours": 1.5}, **ctx)

    assert result["session_hours"] == 1.5, (
        f"caller-provided session_hours overridden by dispatcher; got "
        f"session_hours={result['session_hours']!r} (expected 1.5)"
    )


def test_get_body_state_no_active_buffer_keeps_session_hours_zero(
    tmp_path: Path,
) -> None:
    """No active_conversations dir → injection yields 0.0 (no change to the
    fresh-persona behaviour). Guards against regression on CLI / fresh
    install path."""
    ctx = _make_ctx(tmp_path)
    # Deliberately do NOT plant an active buffer.

    result = dispatch("get_body_state", {}, **ctx)

    assert result["session_hours"] == 0.0


# ---------------------------------------------------------------------------
# All dispatched tools smoke-test
# ---------------------------------------------------------------------------


def test_all_dispatched_tools_dispatch_without_crash(tmp_path: Path) -> None:
    """All registered tool names dispatch without raising."""
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
        "save_work": {
            "title": "smoke",
            "type": "idea",
            "content": "smoke-test content",
        },
        "list_works": {},
        "search_works": {"query": "smoke"},
        "read_work": {"id": "zzzzzzzzzzzz"},
    }

    for tool_name in _DISPATCH:
        args = minimal_args.get(tool_name, {})
        result = dispatch(tool_name, args, **ctx)
        assert isinstance(result, dict) or isinstance(result, list), (
            f"{tool_name} did not return a dict or list"
        )
