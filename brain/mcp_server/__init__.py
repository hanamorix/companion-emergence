"""brain.mcp_server — MCP server exposing brain-tools to Claude.

Spawned as `python -m brain.mcp_server --persona-dir <path>` by
ClaudeCliProvider via --mcp-config. Lifecycle is per-chat-call: claude
spawns this process when it resolves the mcp config; the process runs
until claude closes the stdio connection.

Public surface:
    run_server(persona_dir: Path) — entry; runs until stdio closes
    register_tools                 — re-exported from .tools for testing

Spec: docs/superpowers/specs/2026-04-27-mcp-config-path-for-brain-tools-design.md
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server

from brain.mcp_server.tools import register_tools
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

__all__ = ["run_server", "register_tools"]


def run_server(persona_dir: Path) -> None:
    """Run the MCP server stdio loop until the client disconnects.

    Raises FileNotFoundError if persona_dir does not exist.
    """
    if not persona_dir.is_dir():
        raise FileNotFoundError(f"persona_dir does not exist: {persona_dir}")
    asyncio.run(_run(persona_dir))


async def _run(persona_dir: Path) -> None:
    server = Server("brain-tools")
    store = MemoryStore(db_path=persona_dir / "memories.db")
    hebbian = HebbianMatrix(db_path=persona_dir / "hebbian.db")
    try:
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        try:
            hebbian.close()
        finally:
            store.close()
