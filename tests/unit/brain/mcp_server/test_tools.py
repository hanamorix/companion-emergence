"""Tests for brain.mcp_server.tools — MCP tool registration adapter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    """Minimal persona dir for tests."""
    d = tmp_path / "persona"
    d.mkdir()
    return d


@pytest.fixture()
def fake_stores() -> tuple[MagicMock, MagicMock]:
    return MagicMock(name="MemoryStore"), MagicMock(name="HebbianMatrix")


def test_register_tools_advertises_all_nine(persona_dir: Path, fake_stores) -> None:
    """list_tools() should advertise every schema in NELL_TOOL_NAMES."""
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools
    from brain.tools import NELL_TOOL_NAMES
    from brain.tools.schemas import SCHEMAS

    store, hebbian = fake_stores
    server = Server("brain-tools")
    register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)

    # Pull the list_tools handler the server registered
    list_handler = server.request_handlers[
        __import__("mcp.types", fromlist=["ListToolsRequest"]).ListToolsRequest
    ]
    result = asyncio.run(list_handler(MagicMock()))
    advertised = {t.name for t in result.root.tools}
    expected = {n for n in NELL_TOOL_NAMES if n in SCHEMAS}
    assert advertised == expected


def test_register_tools_dispatches_and_logs_success(persona_dir: Path, fake_stores) -> None:
    """call_tool() must call dispatch() and write an audit log line."""
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools

    store, hebbian = fake_stores
    server = Server("brain-tools")

    with patch("brain.mcp_server.tools.dispatch", return_value={"ok": True}) as mock_dispatch:
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        result = asyncio.run(
            call_handler(_call_request("search_memories", {"query": "x"}))
        )

    # Dispatch was invoked with the right args + injections
    mock_dispatch.assert_called_once_with(
        "search_memories",
        {"query": "x"},
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
    )
    # Result content is a JSON-encoded dispatch return
    text = result.root.content[0].text
    assert json.loads(text) == {"ok": True}
    # Audit log was written
    log_path = persona_dir / "tool_invocations.log.jsonl"
    rec = json.loads(log_path.read_text())
    assert rec["name"] == "search_memories"
    assert rec["arguments"] == {"query": "x"}
    assert rec["error"] is None


def test_register_tools_dispatches_and_logs_error(persona_dir: Path, fake_stores) -> None:
    """When dispatch raises, return {"error": ...} and log with error field."""
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools

    store, hebbian = fake_stores
    server = Server("brain-tools")

    with patch("brain.mcp_server.tools.dispatch", side_effect=RuntimeError("boom")):
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        # Use search_memories with valid args so SDK input validation passes;
        # dispatch is mocked to raise regardless of which tool is called.
        result = asyncio.run(
            call_handler(_call_request("search_memories", {"query": "test"}))
        )

    text = result.root.content[0].text
    assert json.loads(text) == {"error": "boom"}
    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text())
    assert rec["name"] == "search_memories"
    assert rec["error"] == "boom"


def test_register_tools_unknown_tool_returns_error(persona_dir: Path, fake_stores) -> None:
    """Unknown tool names dispatch through the same error path."""
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools
    from brain.tools.dispatch import ToolDispatchError

    store, hebbian = fake_stores
    server = Server("brain-tools")

    with patch(
        "brain.mcp_server.tools.dispatch",
        side_effect=ToolDispatchError("unknown tool: 'banana'"),
    ):
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        result = asyncio.run(call_handler(_call_request("banana", {})))

    text = result.root.content[0].text
    assert "unknown tool" in json.loads(text)["error"]


def test_register_tools_summary_truncated(persona_dir: Path, fake_stores) -> None:
    """A huge dispatch result should still produce a 140-char summary in the log."""
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools

    store, hebbian = fake_stores
    server = Server("brain-tools")

    big_result = {"hits": ["x" * 50 for _ in range(20)]}
    with patch("brain.mcp_server.tools.dispatch", return_value=big_result):
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        asyncio.run(call_handler(_call_request("search_memories", {"query": "x"})))

    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text())
    assert len(rec["result_summary"]) <= 141  # 140 + "…"


# ── helpers ───────────────────────────────────────────────────────────────────


def _get_call_handler(server):
    """Pull the call_tool handler off the server's request map."""
    from mcp.types import CallToolRequest

    return server.request_handlers[CallToolRequest]


def _call_request(name: str, arguments: dict):
    """Build a CallToolRequest in the shape the SDK passes to the handler."""
    from mcp.types import CallToolRequest, CallToolRequestParams

    return CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=name, arguments=arguments),
    )
