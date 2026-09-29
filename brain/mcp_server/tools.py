"""MCP tool registration adapter.

For each name in brain.tools.NELL_TOOL_NAMES, register an MCP tool on the
given Server that dispatches to brain.tools.dispatch.dispatch() and audit-
logs the invocation. Tool logic is not duplicated — every tool routes
through the same dispatch the chat engine already uses.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server import Server
from mcp.types import ImageContent, TextContent, Tool

from brain import dev_constants
from brain.bridge import state_file
from brain.mcp_server.audit import log_invocation
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools import NELL_TOOL_NAMES
from brain.tools.dispatch import dispatch
from brain.tools.schemas import build_schemas

# Tools dispatched to the running bridge over its local HTTP API instead of
# in-process (INC-4, S9/S10): the per-turn MCP tool process never builds the
# embedder/reranker/vector matrix for these — the bridge already holds them
# as long-lived singletons (S11). No fallback: a bridge failure is an error
# result, not a local model load (S10).
_BRIDGE_ROUTED_TOOLS = frozenset({"search_memories"})

# Must match brain.mcp_server.audit._RESULT_SUMMARY_MAX_CHARS — both
# files truncate at the same boundary so the audit log preview length
# stays consistent regardless of which truncation triggered first.
_RESULT_SUMMARY_MAX_CHARS = 140


def _search_via_bridge(persona_dir: Path, arguments: dict[str, Any]) -> dict:
    """Forward a search_memories call to the running bridge over HTTP (INC-4).

    Reads bridge.json for the port + bearer token (the same auth /health
    uses, S54) and posts to POST /tools/search_memories. No fallback: any
    failure here becomes an error result — nothing in this process ever
    builds the embedder/reranker/vector matrix (S10).

    Outcomes:
      - bridge.json missing / no recorded port / connection refused/reset →
        {"error": "bridge unreachable"}.
      - call exceeds SEARCH_BRIDGE_TIMEOUT_S → {"error": "bridge timeout"}
        (distinct from "unreachable", S39).
      - non-200 response → {"error": "bridge error <status>"}.
      - 200 → the bridge's JSON body, unchanged (byte-identical to an
        in-bridge dispatch() call for the same args, C1b).

    Any OTHER exception here (e.g. a malformed 200 body from resp.json())
    is not caught locally — it propagates to _call_tool's own outer
    try/except, which turns it into an {"error": ...} result and an
    outcome="error" audit row. Fail-soft (S10) still holds end to end;
    it's just handled one frame up, not inside this function.
    """
    state = state_file.read(persona_dir)
    if state is None or state.port is None:
        return {"error": "bridge unreachable"}
    headers = {"Authorization": f"Bearer {state.auth_token}"} if state.auth_token else {}
    try:
        resp = httpx.post(
            f"http://127.0.0.1:{state.port}/tools/search_memories",
            json=arguments,
            headers=headers,
            timeout=dev_constants.SEARCH_BRIDGE_TIMEOUT_S,
        )
    except httpx.TimeoutException:
        return {"error": "bridge timeout"}
    except httpx.HTTPError:
        return {"error": "bridge unreachable"}
    if resp.status_code == 200:
        return resp.json()
    return {"error": f"bridge error {resp.status_code}"}


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
            # #80: session_id crosses the parent->claude-CLI->mcp_server process
            # boundary via env var (NELL_MCP_SESSION_ID, set by
            # brain_tools_mcp_entry — mirrors NELL_MCP_AUDIT_REQUEST_ID below).
            # Harmless for every tool outside dispatch()'s _PROVIDER_TOOLS set,
            # which is the only consumer of this kwarg.
            session_id = os.environ.get("NELL_MCP_SESSION_ID") or None
            if name in _BRIDGE_ROUTED_TOOLS:
                result = _search_via_bridge(persona_dir, arguments)
            else:
                result = dispatch(
                    name,
                    arguments,
                    store=store,
                    hebbian=hebbian,
                    persona_dir=persona_dir,
                    session_id=session_id,
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
                # Surface the content-addressed rel_path (a hash, never base64)
                # so it rides the audit -> dispatched_invocations channel back to
                # the engine, which persists it into the durable buffer. This is
                # the only structured tool->engine back-channel on the cli path.
                stored = result.get("stored_image")
                stored_image_path = (
                    stored.get("rel_path") if isinstance(stored, dict) else None
                )
                log_invocation(
                    persona_dir,
                    name=name,
                    arguments=arguments,
                    result_summary=_summarize(summary),
                    monologue_text=None,
                    stored_image_path=stored_image_path,
                )
                return [ImageContent(type="image", data=data_b64, mimeType=media_type)]
            payload = json.dumps(result, default=str, ensure_ascii=False)
            monologue_text: str | None = (
                result.get("monologue_text") if isinstance(result, dict) else None
            )
            # #96/#102: a tool that RETURNS {"error": ...} is a refusal, not a
            # success — write_guard denials come back this way rather than
            # raising, so a refused propose_write was indistinguishable from a
            # committed one in the invocation record. Mirrors tool_loop.py's
            # in-process outcome semantics exactly.
            refusal_error = result.get("error") if isinstance(result, dict) else None
            # #175: a record_monologue the capture layer deduped is still a
            # dispatch that happened — audit it, but marked, so the log no
            # longer shows two "ok" captures per turn.
            deduped = isinstance(result, dict) and result.get("deduped") is True
            log_invocation(
                persona_dir,
                name=name,
                arguments=arguments,
                result_summary=_summarize(payload),
                error=refusal_error,
                outcome="refused" if refusal_error else ("deduped" if deduped else "ok"),
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
                outcome="error",
            )
            return [TextContent(type="text", text=err_payload)]


def _summarize(payload: str) -> str:
    """Single-line preview matching tool_loop._summarize_result behaviour."""
    s = payload.replace("\n", " ").strip()
    if len(s) <= _RESULT_SUMMARY_MAX_CHARS:
        return s
    return s[:_RESULT_SUMMARY_MAX_CHARS] + "…"
