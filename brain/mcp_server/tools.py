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
from mcp.types import ImageContent, TextContent, Tool

from brain.mcp_server.audit import log_invocation
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools import NELL_TOOL_NAMES
from brain.tools.dispatch import dispatch
from brain.tools.schemas import build_schemas

# Must match brain.mcp_server.audit._RESULT_SUMMARY_MAX_CHARS — both
# files truncate at the same boundary so the audit log preview length
# stays consistent regardless of which truncation triggered first.
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

    companion_name = persona_dir.name
    schemas = build_schemas(companion_name)

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=schemas[name].get("description", ""),
                inputSchema=schemas[name].get("parameters", {"type": "object"}),
            )
            for name in NELL_TOOL_NAMES
            if name in schemas
        ]

    @server.call_tool()
    async def _call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent | ImageContent]:
        try:
            result = dispatch(
                name,
                arguments,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
            # Viewable-image result: read_file (and any future image-returning
            # tool) signals an image with a structured `image` key. Emit an MCP
            # ImageContent block so the model actually SEES the pixels under the
            # disallowed-builtins posture (P0 spike mechanism). The audit line
            # summarizes the image compactly — NEVER the base64 (red-team G5 /
            # C15).
            if isinstance(result, dict) and isinstance(result.get("image"), dict):
                img = result["image"]
                media_type = str(img.get("media_type", ""))
                data_b64 = str(img.get("data_b64", ""))
                size_bytes = img.get("size_bytes")
                summary = (
                    f"{media_type} {size_bytes}B" if size_bytes is not None else media_type
                )
                log_invocation(
                    persona_dir,
                    name=name,
                    arguments=arguments,
                    result_summary=_summarize(summary),
                    monologue_text=None,
                )
                return [ImageContent(type="image", data=data_b64, mimeType=media_type)]
            payload = json.dumps(result, default=str, ensure_ascii=False)
            monologue_text: str | None = (
                result.get("monologue_text") if isinstance(result, dict) else None
            )
            log_invocation(
                persona_dir,
                name=name,
                arguments=arguments,
                result_summary=_summarize(payload),
                monologue_text=monologue_text,
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
