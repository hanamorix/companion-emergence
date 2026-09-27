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
    """call_tool() must call dispatch() and write an audit log line.

    Uses get_soul as the generic dispatch-routed example — search_memories
    is bridge-routed (INC-4, _BRIDGE_ROUTED_TOOLS) and no longer reaches
    dispatch() from this handler; its bridge-transport path has its own
    coverage in tests/bridge/test_search_via_bridge.py + this file's
    test_register_tools_search_memories_routes_via_bridge below.
    """
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools

    monkeypatch.delenv("NELL_MCP_SESSION_ID", raising=False)
    store, hebbian = fake_stores
    server = Server("brain-tools")

    with patch("brain.mcp_server.tools.dispatch", return_value={"ok": True}) as mock_dispatch:
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        result = asyncio.run(call_handler(_call_request("get_soul", {})))

    # Dispatch was invoked with the right args + injections. #80: session_id is
    # now always passed (None here — no NELL_MCP_SESSION_ID in the environment)
    # — harmless for every tool outside dispatch()'s _PROVIDER_TOOLS set.
    mock_dispatch.assert_called_once_with(
        "get_soul",
        {},
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
    assert rec["name"] == "get_soul"
    assert rec["arguments"] == {}
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
        # Use get_soul (dispatch-routed; search_memories is bridge-routed as
        # of INC-4 and would never reach this mocked dispatch).
        result = asyncio.run(call_handler(_call_request("get_soul", {})))

    text = result.root.content[0].text
    assert json.loads(text) == {"error": "boom"}
    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["name"] == "get_soul"
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
        # get_soul (dispatch-routed); search_memories is bridge-routed (INC-4).
        asyncio.run(call_handler(_call_request("get_soul", {})))

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


def test_register_tools_logs_deduped_outcome_for_duplicate_monologue(
    persona_dir: Path, fake_stores, monkeypatch
) -> None:
    """#175: a deduped record_monologue is audited as outcome="deduped", not a
    second "ok" row carrying monologue_text. The audit trail still records that
    the dispatch happened — it is marked, not suppressed."""
    from mcp.server import Server

    from brain.mcp_server.tools import register_tools

    monkeypatch.delenv("NELL_MCP_SESSION_ID", raising=False)
    store, hebbian = fake_stores
    server = Server("brain-tools")

    with patch("brain.mcp_server.tools.dispatch", return_value={"ok": True, "deduped": True}):
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        asyncio.run(call_handler(_call_request("record_monologue", {"monologue": "t", "feed_digest": "d"})))

    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["name"] == "record_monologue"
    assert rec["outcome"] == "deduped"
    assert not rec.get("monologue_text")


# ── INC-4 — search_memories routes to the bridge, not dispatch() ─────────────
#
# The MCP child never builds the embedder/reranker/vector matrix for
# search_memories (S9/S10): _search_via_bridge forwards over HTTP to the
# bridge's POST /tools/search_memories using bridge.json's port + bearer
# token. Fail-first: pre-change, search_memories went through the same
# dispatch() path as every other tool — these tests would have failed
# against that code (dispatch mocked, never called; _search_via_bridge
# didn't exist).


def _fake_bridge_state(port: int | None = 4321, auth_token: str | None = "tok-123"):
    from brain.bridge import state_file

    return state_file.BridgeState(
        persona="test-persona",
        pid=1234,
        port=port,
        started_at="2026-01-01T00:00:00+00:00",
        stopped_at=None,
        shutdown_clean=True,
        client_origin="cli",
        auth_token=auth_token,
    )


def test_search_via_bridge_success_posts_and_returns_body(
    persona_dir: Path, monkeypatch
) -> None:
    """Happy path: bridge.json read, correct URL/headers/timeout, body returned unchanged."""
    from brain import dev_constants
    from brain.mcp_server import tools as tools_mod

    monkeypatch.setattr(tools_mod.state_file, "read", lambda _pd: _fake_bridge_state())

    captured: dict = {}

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"memories": [{"id": "m1"}], "mode": "semantic"}

    def _fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResp()

    monkeypatch.setattr(tools_mod.httpx, "post", _fake_post)

    result = tools_mod._search_via_bridge(persona_dir, {"query": "x", "limit": 5})

    assert result == {"memories": [{"id": "m1"}], "mode": "semantic"}
    assert captured["url"] == "http://127.0.0.1:4321/tools/search_memories"
    assert captured["json"] == {"query": "x", "limit": 5}
    assert captured["headers"] == {"Authorization": "Bearer tok-123"}
    assert captured["timeout"] == dev_constants.SEARCH_BRIDGE_TIMEOUT_S


def test_search_via_bridge_no_state_file_is_unreachable(persona_dir: Path, monkeypatch) -> None:
    from brain.mcp_server import tools as tools_mod

    monkeypatch.setattr(tools_mod.state_file, "read", lambda _pd: None)
    result = tools_mod._search_via_bridge(persona_dir, {"query": "x"})
    assert result == {"error": "bridge unreachable"}


def test_search_via_bridge_no_port_is_unreachable(persona_dir: Path, monkeypatch) -> None:
    from brain.mcp_server import tools as tools_mod

    monkeypatch.setattr(tools_mod.state_file, "read", lambda _pd: _fake_bridge_state(port=None))
    result = tools_mod._search_via_bridge(persona_dir, {"query": "x"})
    assert result == {"error": "bridge unreachable"}


def test_search_via_bridge_connect_error_is_unreachable(persona_dir: Path, monkeypatch) -> None:
    from brain.mcp_server import tools as tools_mod

    monkeypatch.setattr(tools_mod.state_file, "read", lambda _pd: _fake_bridge_state())

    def _raise_connect(*_a, **_kw):
        raise tools_mod.httpx.ConnectError("refused")

    monkeypatch.setattr(tools_mod.httpx, "post", _raise_connect)
    result = tools_mod._search_via_bridge(persona_dir, {"query": "x"})
    assert result == {"error": "bridge unreachable"}


def test_search_via_bridge_timeout_is_distinct_error(persona_dir: Path, monkeypatch) -> None:
    """S39: timeout must be a DIFFERENT error string from unreachable."""
    from brain.mcp_server import tools as tools_mod

    monkeypatch.setattr(tools_mod.state_file, "read", lambda _pd: _fake_bridge_state())

    def _raise_timeout(*_a, **_kw):
        raise tools_mod.httpx.TimeoutException("too slow")

    monkeypatch.setattr(tools_mod.httpx, "post", _raise_timeout)
    result = tools_mod._search_via_bridge(persona_dir, {"query": "x"})
    assert result == {"error": "bridge timeout"}
    assert result != {"error": "bridge unreachable"}


def test_search_via_bridge_non_200_reports_status(persona_dir: Path, monkeypatch) -> None:
    from brain.mcp_server import tools as tools_mod

    monkeypatch.setattr(tools_mod.state_file, "read", lambda _pd: _fake_bridge_state())

    class _FakeResp:
        status_code = 401

        def json(self):  # pragma: no cover — not reached on non-200
            raise AssertionError("json() must not be called on a non-200 response")

    monkeypatch.setattr(tools_mod.httpx, "post", lambda *a, **kw: _FakeResp())
    result = tools_mod._search_via_bridge(persona_dir, {"query": "x"})
    assert result == {"error": "bridge error 401"}


def test_register_tools_search_memories_routes_via_bridge_not_dispatch(
    persona_dir: Path, fake_stores, monkeypatch
) -> None:
    """search_memories must NOT reach dispatch() — it's bridge-routed (INC-4).

    Fail-first: before this increment, search_memories went through the same
    dispatch() call every other tool does; this test asserts dispatch is
    never invoked and the bridge helper is used instead, with the audit log
    still written the same way as any other tool.
    """
    from mcp.server import Server

    from brain.mcp_server import tools as tools_mod
    from brain.mcp_server.tools import register_tools

    store, hebbian = fake_stores
    server = Server("brain-tools")

    with (
        patch.object(
            tools_mod, "_search_via_bridge", return_value={"memories": [], "mode": "lexical"}
        ) as mock_bridge,
        patch.object(tools_mod, "dispatch") as mock_dispatch,
    ):
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        result = asyncio.run(call_handler(_call_request("search_memories", {"query": "x"})))

    mock_dispatch.assert_not_called()
    mock_bridge.assert_called_once_with(persona_dir, {"query": "x"})
    text = result.root.content[0].text
    assert json.loads(text) == {"memories": [], "mode": "lexical"}
    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["name"] == "search_memories"
    assert rec["outcome"] == "ok"


def test_register_tools_search_memories_bridge_error_is_refused_outcome(
    persona_dir: Path, fake_stores
) -> None:
    """A bridge error result ({"error": ...}) is a refusal, same as any other
    tool's {"error": ...} return (#96/#102 semantics apply uniformly)."""
    from mcp.server import Server

    from brain.mcp_server import tools as tools_mod
    from brain.mcp_server.tools import register_tools

    store, hebbian = fake_stores
    server = Server("brain-tools")

    with patch.object(
        tools_mod, "_search_via_bridge", return_value={"error": "bridge unreachable"}
    ):
        register_tools(server, persona_dir=persona_dir, store=store, hebbian=hebbian)
        call_handler = _get_call_handler(server)
        result = asyncio.run(call_handler(_call_request("search_memories", {"query": "x"})))

    text = result.root.content[0].text
    assert json.loads(text) == {"error": "bridge unreachable"}
    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["name"] == "search_memories"
    assert rec["outcome"] == "refused"
    assert rec["error"] == "bridge unreachable"


def test_mcp_server_tools_module_never_imports_model_building_modules() -> None:
    """Structural half of C1c ("no provider constructed"): the MCP child's own
    module source never imports the modules that build the embedder,
    reranker, or vector matrix — those symbols are simply unreachable from
    this process for search_memories's bridge-routed path. This is a
    stronger guarantee than an RSS proxy (tests/bridge/test_search_via_bridge.py
    measures the empirical RSS bound on a real process; this pins the causal
    mechanism a real-process test can only observe indirectly)."""
    import brain.mcp_server.tools as tools_mod

    src = Path(tools_mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("brain.memory.embeddings", "brain.memory.reranker", "brain.memory.embedding_matrix"):
        assert forbidden not in src, (
            f"brain/mcp_server/tools.py must never import {forbidden!r} — that would let the "
            "MCP child build a model in-process for a bridge-routed tool"
        )
