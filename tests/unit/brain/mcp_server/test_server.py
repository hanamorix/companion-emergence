"""Tests for brain.mcp_server.run_server + __main__."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def _seed_persona(tmp_path: Path) -> Path:
    """Initialize a minimal valid persona directory."""
    d = tmp_path / "persona"
    d.mkdir()
    MemoryStore(db_path=d / "memories.db").close()
    HebbianMatrix(db_path=d / "hebbian.db").close()
    return d


def test_run_server_missing_persona_dir_raises(tmp_path: Path) -> None:
    from brain.mcp_server import run_server

    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="persona_dir does not exist"):
        run_server(missing)


def test_run_server_opens_and_closes_stores(tmp_path: Path) -> None:
    """run_server should open MemoryStore + HebbianMatrix + close on exit,
    even when the stdio loop returns immediately (mocked)."""
    persona = _seed_persona(tmp_path)

    from brain.mcp_server import run_server

    captured: dict = {}

    def _capture_register(server, *, persona_dir, store, hebbian):
        captured["store"] = store
        captured["hebbian"] = hebbian
        captured["persona_dir"] = persona_dir

    # Patch the stdio_server context manager to immediately exit
    class _FakeStdio:
        async def __aenter__(self):
            return (MagicMock(), MagicMock())

        async def __aexit__(self, *_):
            return None

    async def _fake_run(self, *_args, **_kwargs):
        return None

    with (
        patch("brain.mcp_server.register_tools", side_effect=_capture_register),
        patch("brain.mcp_server.stdio_server", lambda: _FakeStdio()),
        patch("mcp.server.Server.run", _fake_run),
    ):
        run_server(persona)

    # Stores were captured (and the run completed without raising on close)
    assert captured["persona_dir"] == persona
    # Stores are typed instances, not None
    assert captured["store"] is not None
    assert captured["hebbian"] is not None


def test_main_entry_runs(tmp_path: Path) -> None:
    """`python -m brain.mcp_server --persona-dir <path>` should accept the
    flag and invoke run_server. We use subprocess + a side-effect stub to
    confirm the dispatch without actually running stdio."""
    persona = _seed_persona(tmp_path)

    # Use subprocess so we exercise the real argparse path. The MCP server
    # would normally hang on stdio_server() — we kill it after a moment.
    proc = subprocess.Popen(
        [sys.executable, "-m", "brain.mcp_server", "--persona-dir", str(persona)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Passing empty bytes causes communicate() to close stdin (EOF),
        # which the stdio MCP server detects and exits cleanly.
        stdout, stderr = proc.communicate(input=b"", timeout=5)
        # Exit code 0 means the entry point parsed args and the server
        # ran cleanly. Non-zero means a crash before stdio_server returned.
        assert proc.returncode == 0, f"Server exited {proc.returncode}.\nstderr:\n{stderr.decode()}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)


def test_main_entry_missing_flag_exits_nonzero() -> None:
    """argparse should reject a call without --persona-dir."""
    proc = subprocess.run(
        [sys.executable, "-m", "brain.mcp_server"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert proc.returncode != 0
    assert "--persona-dir" in proc.stderr
