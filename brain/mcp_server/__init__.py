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
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server

from brain.mcp_server.tools import register_tools
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

__all__ = ["run_server", "register_tools"]

logger = logging.getLogger(__name__)


def run_server(persona_dir: Path, *, tool_prefix: str = "") -> None:
    """Run the MCP server stdio loop until the client disconnects.

    Raises FileNotFoundError if persona_dir does not exist.
    """
    if not persona_dir.is_dir():
        raise FileNotFoundError(f"persona_dir does not exist: {persona_dir}")
    asyncio.run(_run(persona_dir, tool_prefix))


async def _run(persona_dir: Path, tool_prefix: str = "") -> None:
    _load_vocabulary(persona_dir)
    server = Server("brain-tools")
    store = MemoryStore(db_path=persona_dir / "memories.db")
    hebbian = HebbianMatrix(db_path=persona_dir / "hebbian.db")
    try:
        register_tools(
            server, persona_dir=persona_dir, store=store, hebbian=hebbian, tool_prefix=tool_prefix
        )
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


def _load_vocabulary(persona_dir: Path) -> None:
    """Register the persona's own emotion names in this process before any tool runs (#689).

    The brain tools (record_monologue, add_memory, the emotion reads) run in this per-turn process, and the registry is
    per process; without this every name outside the baseline is unknown here. Read-only: nothing is written on a
    turn's path, and a corrupt file is left as it is (logged, baseline names only) for the bridge's start-up load to
    repair, since that load has the store and rebuilds from memories. Fail-soft: the tools must still serve.
    """
    try:
        from brain.emotion.persona_loader import load_persona_vocabulary

        load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", read_only=True)
    except Exception:  # noqa: BLE001
        logger.warning("mcp server: could not load the persona vocabulary; tools see only the baseline names", exc_info=True)
