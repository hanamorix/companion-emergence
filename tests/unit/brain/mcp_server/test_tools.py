"""Tests for brain.mcp_server.tools — MCP tool registration adapter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_register_tools_advertises_all_dispatched(persona_dir: Path, fake_stores) -> None:
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


def test_register_tools_dispatches_and_logs_success(
    persona_dir: Path, fake_stores, monkeypatch
) -> None:
    """call_tool() must call dispatch() and write an audit log line."""
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools

    monkeypatch.delenv("NELL_MCP_SESSION_ID", raising=False)
    store, hebbian = fake_stores
    server = Server("brain-tools")

    with patch("brain.mcp_server.tools.dispatch", return_value={"ok": True}) as mock_dispatch:
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        result = asyncio.run(call_handler(_call_request("search_memories", {"query": "x"})))

    # Dispatch was invoked with the right args + injections. #80: session_id is
    # now always passed (None here — no NELL_MCP_SESSION_ID in the environment)
    # — harmless for every tool outside dispatch()'s _PROVIDER_TOOLS set.
    mock_dispatch.assert_called_once_with(
        "search_memories",
        {"query": "x"},
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
        session_id=None,
    )
    # Result content is a JSON-encoded dispatch return
    text = result.root.content[0].text
    assert json.loads(text) == {"ok": True}
    # Audit log was written
    log_path = persona_dir / "tool_invocations.log.jsonl"
    rec = json.loads(log_path.read_text(encoding="utf-8"))
    assert rec["name"] == "search_memories"
    # Audit 2026-05-07 P3-3: 'query' is now redacted in default mode.
    assert rec["arguments"] == {"query": "[REDACTED]"}
    assert rec["error"] is None
    # #96/#102: a normal return is outcome="ok".
    assert rec["outcome"] == "ok"


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
        result = asyncio.run(call_handler(_call_request("search_memories", {"query": "test"})))

    text = result.root.content[0].text
    assert json.loads(text) == {"error": "boom"}
    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["name"] == "search_memories"
    assert rec["error"] == "boom"
    # #96/#102: a raised exception is outcome="error".
    assert rec["outcome"] == "error"


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

    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert len(rec["result_summary"]) <= 141  # 140 + "…"


def test_register_tools_logs_refused_outcome_for_guard_denial(
    persona_dir: Path, fake_stores
) -> None:
    """#96/#102: write_guard denials RETURN {"error": ...} rather than raising,
    so a refused propose_write was indistinguishable from a committed one in
    the invocation record. Goes through the real dispatch/propose_write/
    write_guard chain (not mocked) — a guard-denied path is a realistic case.
    """
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools

    store, hebbian = fake_stores
    server = Server("brain-tools")
    register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
    call_handler = _get_call_handler(server)

    result = asyncio.run(
        call_handler(
            _call_request(
                "propose_write",
                {"path": "/etc/passwd", "op": "create", "content": "x"},
            )
        )
    )

    text = result.root.content[0].text
    payload = json.loads(text)
    assert "error" in payload  # write_guard denial, returned not raised

    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["name"] == "propose_write"
    assert rec["outcome"] == "refused"
    assert rec["error"]  # non-null/non-empty — the record must not drop it


def test_register_tools_emits_image_content_for_image_result(persona_dir: Path, fake_stores) -> None:
    """P0 image-tool-route: a dispatch result carrying an ``image`` key is
    emitted as an MCP ImageContent block (base64 + mimeType), NOT TextContent —
    this is what lets the model SEE the shared image. The audit summary records
    a compact ``image/<mt> <N>B`` line, never the base64 (C15)."""
    from mcp.server import Server
    from mcp.types import ImageContent

    from brain.mcp_server.tools import register_tools

    store, hebbian = fake_stores
    server = Server("brain-tools")

    b64 = "aGVsbG8="  # "hello"
    img_result = {"path": "/p/x.png", "image": {"media_type": "image/png", "data_b64": b64, "size_bytes": 5}}
    with patch("brain.mcp_server.tools.dispatch", return_value=img_result):
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        result = asyncio.run(call_handler(_call_request("read_file", {"path": "/p/x.png"})))

    block = result.root.content[0]
    assert isinstance(block, ImageContent)
    assert block.data == b64
    assert block.mimeType == "image/png"
    # C15 — the audit summary is a compact image line, never the base64 payload.
    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["name"] == "read_file"
    assert b64 not in rec["result_summary"]
    assert "image/png" in rec["result_summary"] and "5B" in rec["result_summary"]


def test_register_tools_audits_stored_image_path_not_base64(persona_dir: Path, fake_stores) -> None:
    """image-path-persist C3/C10 — when a read_file image result carries a
    ``stored_image`` handle, the MCP audit record surfaces its content-addressed
    ``stored_image_path`` (a hash, for durable-buffer binding) and NEVER the
    base64 image bytes."""
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools

    store, hebbian = fake_stores
    server = Server("brain-tools")

    b64 = "aGVsbG8="  # "hello"
    rel = "images/" + ("a" * 64) + ".png"
    img_result = {
        "path": "/p/x.png",
        "image": {"media_type": "image/png", "data_b64": b64, "size_bytes": 5},
        "stored_image": {"sha": "a" * 64, "media_type": "image/png", "rel_path": rel},
    }
    with patch("brain.mcp_server.tools.dispatch", return_value=img_result):
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        asyncio.run(call_handler(_call_request("read_file", {"path": "/p/x.png"})))

    line = (persona_dir / "tool_invocations.log.jsonl").read_text(encoding="utf-8")
    rec = json.loads(line)
    assert rec["stored_image_path"] == rel
    assert b64 not in line  # no base64 anywhere in the audit record


# ── #80 — compact_history reachable via the REAL MCP handler ──────────────────
#
# Drives server.request_handlers[CallToolRequest] directly (the same mechanism
# the dragonfly repro used) with a REAL (unmocked) dispatch() — proving the
# tool reaches actual compaction logic, not a stub, when NELL_MCP_SESSION_ID is
# present in the environment the way brain_tools_mcp_entry()/_call_tool wire it.


def test_call_tool_compact_history_via_env_session_id_reaches_real_dispatch(
    persona_dir: Path, fake_stores, monkeypatch
) -> None:
    """C1: with NELL_MCP_SESSION_ID set (as the real MCP subprocess spawn sets
    it, per brain_tools_mcp_entry), a real compact_history call through the
    real registered handler must NOT return the #80 provider/session_id error
    — a genuine no-op (reason: cursor_none, on this fresh persona_dir with no
    ingest cursor) is a valid pass, proving real compaction logic was reached."""
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools

    monkeypatch.setenv("NELL_MCP_SESSION_ID", "sess-mcp-live")
    store, hebbian = fake_stores
    server = Server("brain-tools")
    register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
    call_handler = _get_call_handler(server)

    result = asyncio.run(call_handler(_call_request("compact_history", {"age_hours": 1})))

    text = result.root.content[0].text
    payload = json.loads(text)
    assert "error" not in payload, f"compact_history call failed: {payload}"
    assert payload["reason"] == "cursor_none"  # real no-op, not a stub
    assert payload["compacted"] is False


def test_call_tool_compact_history_without_session_id_env_var_fails_loud(
    persona_dir: Path, fake_stores, monkeypatch
) -> None:
    """C2/ST1.5f (this is the oracle's self-test — it reproduces the ORIGINAL
    #80 failure class on demand): with NO NELL_MCP_SESSION_ID in the
    environment, the real handler must still return a loud {"error": ...}
    naming session_id — dropping the provider requirement must not silently
    let an unresolvable-session call through, which would corrupt the wrong
    session's buffer."""
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools

    monkeypatch.delenv("NELL_MCP_SESSION_ID", raising=False)
    store, hebbian = fake_stores
    server = Server("brain-tools")
    register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
    call_handler = _get_call_handler(server)

    result = asyncio.run(call_handler(_call_request("compact_history", {"age_hours": 1})))

    text = result.root.content[0].text
    payload = json.loads(text)
    assert "error" in payload
    assert "session_id" in payload["error"]


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
