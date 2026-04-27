"""MCP tool registration adapter.

For each name in brain.tools.NELL_TOOL_NAMES, register an MCP tool on the
given Server that dispatches to brain.tools.dispatch.dispatch() and audit-
logs the invocation. Tool logic is not duplicated — every tool routes
through the same dispatch the chat engine already uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.mcp_server.audit import log_invocation
from brain.tools import NELL_TOOL_NAMES
from brain.tools.dispatch import dispatch
from brain.tools.schemas import SCHEMAS

_RESULT_SUMMARY_MAX_CHARS = 140


def register_tools(
    server: Server,
    *,
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
) -> None:
    """Register each brain-tool with the MCP server.

    Closures capture store/hebbian/persona_dir so each invocation passes
    them through dispatch unchanged. The server itself is mutated in place;
    the function returns None.
    """

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=SCHEMAS[name].get("description", ""),
                inputSchema=SCHEMAS[name].get("parameters", {"type": "object"}),
            )
            for name in NELL_TOOL_NAMES
            if name in SCHEMAS
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            result = dispatch(
                name,
                arguments,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            payload = json.dumps(result, default=str, ensure_ascii=False)
            log_invocation(
                persona_dir,
                name=name,
                arguments=arguments,
                result_summary=_summarize(payload),
            )
            return [TextContent(type="text", text=payload)]
        except Exception as exc:  # noqa: BLE001 — broad catch is intentional
            err_payload = json.dumps({"error": str(exc)})
            log_invocation(
                persona_dir,
                name=name,
                arguments=arguments,
                result_summary=f"error: {exc}",
                error=str(exc),
            )
            return [TextContent(type="text", text=err_payload)]


def _summarize(payload: str) -> str:
    """Single-line preview matching tool_loop._summarize_result behaviour."""
    s = payload.replace("\n", " ").strip()
    if len(s) <= _RESULT_SUMMARY_MAX_CHARS:
        return s
    return s[:_RESULT_SUMMARY_MAX_CHARS] + "…"
