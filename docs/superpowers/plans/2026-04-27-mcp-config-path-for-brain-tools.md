# MCP-Config Path for Brain-Tools — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Claude `--json-schema` tool-calling with a stdio MCP server so brain-tools fire reliably regardless of how rich `voice.md` becomes.

**Architecture:** A new `brain/mcp_server/` package exposes the existing 9 brain-tools (via `brain.tools.dispatch`) over the Model Context Protocol stdio transport. `ClaudeCliProvider._chat_with_tools` writes a temp `mcp.json`, swaps `--json-schema <schema>` for `--mcp-config <path>` in the existing `claude` subprocess invocation, and returns the final text. Tool calls happen inside the Claude subprocess; our `tool_loop` becomes a single-pass return for Claude (Ollama unchanged).

**Tech Stack:** Python 3.12, the `mcp` SDK (>=1.0.0,<2.0.0), the existing `claude` CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-04-27-mcp-config-path-for-brain-tools-design.md` — read this first if any task is ambiguous.

**Verification gate (per Hana):** Every task ends with tests passing locally. The PR does not merge until the manual sandbox-clone verification (Task 7) succeeds: a real `nell chat` against a sandboxed clone produces `tool_invocations.log.jsonl` showing `search_memories` was called and Nell's reply quotes a verifiable memory.

---

## Task 1: Add `mcp` SDK dependency

**Files:**
- Modify: `pyproject.toml:11-16`

- [ ] **Step 1: Add `mcp` to dependencies**

Edit `pyproject.toml` — change the `dependencies` block from:

```toml
dependencies = [
    "platformdirs>=4.2",
    "numpy>=1.26",
    "ddgs>=6.0",
    "httpx>=0.27",
]
```

to:

```toml
dependencies = [
    "platformdirs>=4.2",
    "numpy>=1.26",
    "ddgs>=6.0",
    "httpx>=0.27",
    "mcp>=1.0.0,<2.0.0",
]
```

- [ ] **Step 2: Install in the active environment**

Run: `python3 -m pip install -e '.[dev]'`
Expected: `Successfully installed mcp-<version> ...`

- [ ] **Step 3: Verify import works**

Run: `python3 -c "from mcp.server import Server; from mcp.server.stdio import stdio_server; from mcp.types import Tool, TextContent; print('mcp imports OK')"`
Expected output: `mcp imports OK`

- [ ] **Step 4: Verify existing test suite still green**

Run: `python3 -m pytest -q 2>&1 | tail -5`
Expected: All previously-passing tests still pass (no regression from adding the dep).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add mcp SDK for brain-tools MCP-config path

Pinned mcp>=1.0.0,<2.0.0 — major-pinned to avoid breaking API drift,
minor floats so security/bugfix releases land automatically.

Used by brain/mcp_server/ to expose the 9 brain-tools to Claude via
the --mcp-config production path (replaces the --json-schema interim
shipped in PR #23). Spec: docs/superpowers/specs/2026-04-27-mcp-\
config-path-for-brain-tools-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Audit log module

**Files:**
- Create: `brain/mcp_server/__init__.py` (empty package marker for now — will be filled in Task 5)
- Create: `brain/mcp_server/audit.py`
- Create: `tests/unit/brain/mcp_server/__init__.py` (empty)
- Create: `tests/unit/brain/mcp_server/test_audit.py`

- [ ] **Step 1: Create the package markers**

```bash
mkdir -p brain/mcp_server tests/unit/brain/mcp_server
touch brain/mcp_server/__init__.py tests/unit/brain/mcp_server/__init__.py
```

- [ ] **Step 2: Write failing tests (`test_audit.py`)**

Create `tests/unit/brain/mcp_server/test_audit.py`:

```python
"""Tests for brain.mcp_server.audit."""

from __future__ import annotations

import json
from pathlib import Path

from brain.mcp_server.audit import log_invocation


def test_log_invocation_writes_jsonl_line(tmp_path: Path) -> None:
    log_invocation(
        tmp_path,
        name="search_memories",
        arguments={"query": "morning"},
        result_summary="3 hits",
    )
    log_path = tmp_path / "tool_invocations.log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["name"] == "search_memories"
    assert rec["arguments"] == {"query": "morning"}
    assert rec["result_summary"] == "3 hits"
    assert rec["error"] is None
    # Timestamp ends with Z (UTC)
    assert rec["timestamp"].endswith("Z")


def test_log_invocation_truncates_long_summary(tmp_path: Path) -> None:
    long = "x" * 500
    log_invocation(tmp_path, name="x", arguments={}, result_summary=long)
    rec = json.loads((tmp_path / "tool_invocations.log.jsonl").read_text())
    # 140 chars + ellipsis (1 char "…")
    assert rec["result_summary"].endswith("…")
    assert len(rec["result_summary"]) == 141


def test_log_invocation_records_error(tmp_path: Path) -> None:
    log_invocation(
        tmp_path,
        name="add_memory",
        arguments={"text": "x"},
        result_summary="error: boom",
        error="boom",
    )
    rec = json.loads((tmp_path / "tool_invocations.log.jsonl").read_text())
    assert rec["error"] == "boom"


def test_log_invocation_appends_two_lines(tmp_path: Path) -> None:
    log_invocation(tmp_path, name="a", arguments={}, result_summary="r1")
    log_invocation(tmp_path, name="b", arguments={}, result_summary="r2")
    lines = (tmp_path / "tool_invocations.log.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["name"] == "a"
    assert json.loads(lines[1])["name"] == "b"


def test_log_invocation_swallows_oserror(tmp_path: Path, monkeypatch) -> None:
    """Audit is observability — if the disk write fails, we log + swallow."""
    # Make the persona_dir read-only so the open() raises OSError
    persona = tmp_path / "ro"
    persona.mkdir()
    persona.chmod(0o555)
    try:
        # Should not raise
        log_invocation(persona, name="x", arguments={}, result_summary="x")
    finally:
        persona.chmod(0o755)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/brain/mcp_server/test_audit.py -v 2>&1 | tail -10`
Expected: All 5 tests fail with `ModuleNotFoundError: No module named 'brain.mcp_server.audit'`

- [ ] **Step 4: Implement `audit.py`**

Create `brain/mcp_server/audit.py`:

```python
"""Audit log for MCP-server tool invocations.

Each call to a brain-tool from inside the MCP server appends one JSON line
to <persona_dir>/tool_invocations.log.jsonl. Failures here are observability,
not correctness — they are logged to stderr and swallowed so a broken disk
or full filesystem cannot break tool dispatch.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RESULT_SUMMARY_MAX_CHARS = 140
_LOG_FILENAME = "tool_invocations.log.jsonl"


def log_invocation(
    persona_dir: Path,
    *,
    name: str,
    arguments: dict[str, Any],
    result_summary: str,
    error: str | None = None,
) -> None:
    """Append one invocation record to <persona_dir>/tool_invocations.log.jsonl.

    Never raises. OSError on the write is logged at WARNING and swallowed.

    Parameters
    ----------
    persona_dir:
        The active persona's directory; the log file is written here.
    name:
        Tool name (e.g. "search_memories").
    arguments:
        Args the LLM passed in. Will be JSON-serialised; non-JSON values
        fall through to ``default=str``.
    result_summary:
        Compact preview of the result. Truncated to 140 chars + "…" if longer.
    error:
        ``None`` on success; ``str(exc)`` on dispatch failure.
    """
    if len(result_summary) <= _RESULT_SUMMARY_MAX_CHARS:
        truncated = result_summary
    else:
        truncated = result_summary[:_RESULT_SUMMARY_MAX_CHARS] + "…"

    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "name": name,
        "arguments": arguments,
        "result_summary": truncated,
        "error": error,
    }

    log_path = persona_dir / _LOG_FILENAME
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning("audit log write failed: %s", exc)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/brain/mcp_server/test_audit.py -v 2>&1 | tail -10`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add brain/mcp_server/__init__.py brain/mcp_server/audit.py \
        tests/unit/brain/mcp_server/__init__.py \
        tests/unit/brain/mcp_server/test_audit.py
git commit -m "feat(mcp): tool-invocation audit log module

log_invocation() appends one JSON line per call to
<persona>/tool_invocations.log.jsonl. Used by the MCP server (next
task) and consumed by future audit/debug surfaces. OSError on write
is logged + swallowed — audit is observability, not correctness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Promote `NELL_TOOL_NAMES` to public API

**Files:**
- Modify: `brain/tools/__init__.py:1-20`
- Modify: `brain/chat/tool_loop.py:31-42`
- Test: `tests/unit/brain/test_tools_public.py` (new)

**Why:** The MCP server (Task 4) needs the canonical 9-tool list. Today it's a private constant inside `brain.chat.tool_loop` (`_NELL_TOOL_NAMES`). Importing private names across modules is ugly; the cleanest fix is to expose it once on `brain.tools`.

- [ ] **Step 1: Write failing test**

Create `tests/unit/brain/test_tools_public.py`:

```python
"""Tests for brain.tools public surface — NELL_TOOL_NAMES export."""

from __future__ import annotations


def test_nell_tool_names_exported() -> None:
    from brain.tools import NELL_TOOL_NAMES

    assert isinstance(NELL_TOOL_NAMES, tuple)
    assert len(NELL_TOOL_NAMES) == 9
    # Spot-check the 9 known tools from spec §1
    assert "search_memories" in NELL_TOOL_NAMES
    assert "get_emotional_state" in NELL_TOOL_NAMES
    assert "get_soul" in NELL_TOOL_NAMES
    assert "get_personality" in NELL_TOOL_NAMES
    assert "get_body_state" in NELL_TOOL_NAMES
    assert "boot" in NELL_TOOL_NAMES
    assert "add_journal" in NELL_TOOL_NAMES
    assert "add_memory" in NELL_TOOL_NAMES
    assert "crystallize_soul" in NELL_TOOL_NAMES


def test_tool_loop_imports_from_brain_tools() -> None:
    """tool_loop must use the public name — no private fallback."""
    from brain.chat import tool_loop
    from brain.tools import NELL_TOOL_NAMES

    # build_tools_list iterates the same names — easiest assertion is to
    # call it and confirm shape matches NELL_TOOL_NAMES
    tools = tool_loop.build_tools_list()
    names_in_tools = {t["function"]["name"] for t in tools}
    # SCHEMAS gates which names actually appear; intersect with NELL_TOOL_NAMES
    from brain.tools.schemas import SCHEMAS

    expected = {n for n in NELL_TOOL_NAMES if n in SCHEMAS}
    assert names_in_tools == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/brain/test_tools_public.py -v 2>&1 | tail -10`
Expected: `test_nell_tool_names_exported` fails with `ImportError: cannot import name 'NELL_TOOL_NAMES' from 'brain.tools'`.

- [ ] **Step 3: Define `NELL_TOOL_NAMES` in `brain.tools`**

Edit `brain/tools/__init__.py` — replace the file with:

```python
"""brain.tools — brain-query tools for the chat engine (SP-3).

Public surface:
    SCHEMAS           — JSON schemas keyed by tool name
    LOVE_TYPES        — enum of crystallization love-types
    NELL_TOOL_NAMES   — canonical ordered tuple of all 9 tools
    dispatch          — dispatch(name, arguments, *, store, hebbian, persona_dir) -> dict
    ToolDispatchError — raised on dispatch failures

OG reference: NellBrain/nell_tools.py (841 lines, 9 impls + SCHEMAS).
Master ref §6 SP-3.
"""

from brain.tools.dispatch import ToolDispatchError, dispatch
from brain.tools.schemas import LOVE_TYPES, SCHEMAS

# Canonical tool list, in the order the LLM should see them.
# Ported verbatim from OG NELL_TOOLS (nell_bridge.py:172-185).
NELL_TOOL_NAMES: tuple[str, ...] = (
    "get_emotional_state",
    "get_soul",
    "get_personality",
    "get_body_state",
    "boot",
    "search_memories",
    "add_journal",
    "add_memory",
    "crystallize_soul",
)

__all__ = [
    "SCHEMAS",
    "LOVE_TYPES",
    "NELL_TOOL_NAMES",
    "dispatch",
    "ToolDispatchError",
]
```

- [ ] **Step 4: Switch `tool_loop` to import the public name**

Edit `brain/chat/tool_loop.py` — replace lines 22-42 (the `from brain.tools.schemas import SCHEMAS` line through the end of `_NELL_TOOL_NAMES`) with:

```python
from brain.tools import NELL_TOOL_NAMES
from brain.tools.dispatch import dispatch
from brain.tools.schemas import SCHEMAS

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 4
```

(The private `_NELL_TOOL_NAMES` constant is removed entirely. References below it must use the public name.)

Then in `build_tools_list()` (around lines 45-55), change `_NELL_TOOL_NAMES` → `NELL_TOOL_NAMES`:

```python
def build_tools_list() -> list[dict]:
    """Build the tool schema list for provider.chat(tools=...).

    Wraps schemas in the {"type": "function", "function": <schema>} shape
    that Ollama and Claude (via --json-schema) both accept.
    """
    return [
        {"type": "function", "function": SCHEMAS[name]}
        for name in NELL_TOOL_NAMES
        if name in SCHEMAS
    ]
```

- [ ] **Step 5: Run the new tests + the existing tool_loop tests**

Run: `python3 -m pytest tests/unit/brain/test_tools_public.py tests/unit/brain/chat/test_tool_loop.py -v 2>&1 | tail -10`
Expected: all green (both files).

- [ ] **Step 6: Run the full chat + tools suite to verify no regression**

Run: `python3 -m pytest tests/unit/brain/chat/ tests/unit/brain/test_tools_public.py tests/unit/brain/tools/ 2>&1 | tail -5`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add brain/tools/__init__.py brain/chat/tool_loop.py \
        tests/unit/brain/test_tools_public.py
git commit -m "refactor(tools): expose NELL_TOOL_NAMES on brain.tools

Move the canonical 9-tool list from a private constant in
brain.chat.tool_loop (_NELL_TOOL_NAMES) to the public
brain.tools.NELL_TOOL_NAMES so the new MCP server (next task) can
import it without crossing private boundaries. Tool_loop now imports
the public name; behaviour unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: MCP tools adapter (`register_tools`)

**Files:**
- Create: `brain/mcp_server/tools.py`
- Create: `tests/unit/brain/mcp_server/test_tools.py`

**Responsibility:** map each schema in `brain/tools/schemas.py` to an MCP tool registration on the `Server`. Closures capture `store` / `hebbian` / `persona_dir` so the handler can dispatch + audit-log without re-opening stores per call.

- [ ] **Step 1: Write failing tests (`test_tools.py`)**

Create `tests/unit/brain/mcp_server/test_tools.py`:

```python
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
        result = asyncio.run(
            call_handler(_call_request("add_memory", {"text": "x"}))
        )

    text = result.root.content[0].text
    assert json.loads(text) == {"error": "boom"}
    rec = json.loads((persona_dir / "tool_invocations.log.jsonl").read_text())
    assert rec["name"] == "add_memory"
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
```

> **Note for the implementing agent:** the `mcp` SDK's request-handler internals
> (`server.request_handlers`, `CallToolRequest`, etc.) are stable since 1.0. If
> the test helper functions above need to import differently in the version we
> pin, adjust them, but keep the assertions intact.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/brain/mcp_server/test_tools.py -v 2>&1 | tail -10`
Expected: all 5 fail with `ModuleNotFoundError: No module named 'brain.mcp_server.tools'`.

- [ ] **Step 3: Implement `tools.py`**

Create `brain/mcp_server/tools.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/brain/mcp_server/test_tools.py -v 2>&1 | tail -15`
Expected: 5 passed.

- [ ] **Step 5: If any test fails because the SDK request-handler shape differs from the helpers**

The implementation in step 3 is correct — adjust the `_get_call_handler` / `_call_request` helpers in the test file to match the SDK's actual API surface. Confirm via:

```bash
python3 -c "from mcp.types import CallToolRequest; help(CallToolRequest)"
python3 -c "from mcp.server import Server; s = Server('x'); print(list(s.request_handlers.keys()))"
```

Update the helpers based on the output. Re-run step 4. Do NOT change the assertions — only the helper functions that prepare the request shape.

- [ ] **Step 6: Commit**

```bash
git add brain/mcp_server/tools.py tests/unit/brain/mcp_server/test_tools.py
git commit -m "feat(mcp): tool-registration adapter

register_tools() wires the 9 brain-tools onto an MCP Server: each
tool's schema becomes an MCP Tool, each call routes through the
existing brain.tools.dispatch.dispatch() and audit-logs via
brain.mcp_server.audit.log_invocation. No tool logic duplicated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: MCP server entry point

**Files:**
- Modify: `brain/mcp_server/__init__.py` (replace empty marker with `run_server`)
- Create: `brain/mcp_server/__main__.py`
- Create: `tests/unit/brain/mcp_server/test_server.py`

**Responsibility:** the `python -m brain.mcp_server --persona-dir <path>` entry. Opens stores, registers tools, runs the stdio loop, closes stores on exit.

- [ ] **Step 1: Write failing tests (`test_server.py`)**

Create `tests/unit/brain/mcp_server/test_server.py`:

```python
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

    with patch("brain.mcp_server.register_tools", side_effect=_capture_register), \
         patch("brain.mcp_server.stdio_server", lambda: _FakeStdio()), \
         patch("mcp.server.Server.run", _fake_run):
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
        # Closing stdin causes the stdio MCP server to detect EOF and exit
        proc.stdin.close()
        stdout, stderr = proc.communicate(timeout=5)
        # Exit code 0 means the entry point parsed args and the server
        # ran cleanly. Non-zero means a crash before stdio_server returned.
        assert proc.returncode == 0, (
            f"Server exited {proc.returncode}.\nstderr:\n{stderr.decode()}"
        )
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/brain/mcp_server/test_server.py -v 2>&1 | tail -15`
Expected: failures referencing missing `run_server` symbol on `brain.mcp_server`.

- [ ] **Step 3: Implement `brain/mcp_server/__init__.py`**

Replace the empty `brain/mcp_server/__init__.py` with:

```python
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

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.mcp_server.tools import register_tools

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
```

- [ ] **Step 4: Implement `brain/mcp_server/__main__.py`**

Create `brain/mcp_server/__main__.py`:

```python
"""Entry: `python -m brain.mcp_server --persona-dir <path>`."""

from __future__ import annotations

import argparse
from pathlib import Path

from brain.mcp_server import run_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m brain.mcp_server")
    parser.add_argument(
        "--persona-dir",
        required=True,
        type=Path,
        help="Path to the active persona directory (required).",
    )
    args = parser.parse_args(argv)
    run_server(args.persona_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run server tests to verify they pass**

Run: `python3 -m pytest tests/unit/brain/mcp_server/ -v 2>&1 | tail -15`
Expected: all green (audit + tools + server).

- [ ] **Step 6: Run the full mcp_server suite + adjacent suites**

Run: `python3 -m pytest tests/unit/brain/mcp_server/ tests/unit/brain/chat/ tests/unit/brain/test_tools_public.py 2>&1 | tail -5`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add brain/mcp_server/__init__.py brain/mcp_server/__main__.py \
        tests/unit/brain/mcp_server/test_server.py
git commit -m "feat(mcp): server entry point — run_server + __main__

run_server(persona_dir) opens MemoryStore + HebbianMatrix, calls
register_tools to wire the 9 brain-tools, runs the stdio MCP loop,
and closes stores on exit. The python -m brain.mcp_server entry
parses --persona-dir and dispatches to run_server. Lifecycle is
per-chat-call: claude CLI spawns + kills this process around each
chat turn that uses the --mcp-config path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Provider integration — swap to `--mcp-config`

**Files:**
- Modify: `brain/bridge/provider.py:230-460` (rewrite `chat()` tool path; drop `_build_tool_call_schema`, `_build_tool_system_addendum`, `_chat_with_tools`)
- Modify: `brain/chat/tool_loop.py:97` (pass `persona_dir` via `options`)
- Modify: `tests/unit/brain/bridge/test_provider_chat.py` (replace `--json-schema` tests with `--mcp-config` tests)

**Why this is one task, not two:** the tool_loop signature change and the provider rewrite are tightly coupled. tool_loop must pass `persona_dir` via `options` for the provider's MCP path to know where to point the spawned server. Splitting them across commits leaves one half broken.

- [ ] **Step 1: Write failing tests**

Edit `tests/unit/brain/bridge/test_provider_chat.py` — replace the entire file with this content (the old `--json-schema` tests are removed; new `--mcp-config` tests replace them). Keep any unrelated Ollama tests in the file by appending them at the end if they exist; this snippet covers only the Claude-path tests.

```python
"""Tests for ClaudeCliProvider.chat() — MCP-config path.

Replaces the SP-3 --json-schema tests; the underlying provider was
rewritten to use --mcp-config (production path per master ref §6 SP-3
and 2026-04-27 stress-test finding).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import ClaudeCliProvider, ProviderError


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "persona"
    d.mkdir()
    return d


def _fake_proc(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_chat_with_tools_passes_mcp_config_flag(persona_dir: Path) -> None:
    """The new path must call claude with --mcp-config <tmp_path>."""
    provider = ClaudeCliProvider()
    captured: dict = {}

    def _capture(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["mcp_config_path"] = cmd[cmd.index("--mcp-config") + 1]
        return _fake_proc(json.dumps({"result": "hello back"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_capture):
        response = provider.chat(
            [
                ChatMessage(role="system", content="you are nell"),
                ChatMessage(role="user", content="hi"),
            ],
            tools=[{"name": "search_memories", "description": "search"}],
            options={"persona_dir": str(persona_dir)},
        )

    assert response.content == "hello back"
    assert response.tool_calls == ()
    # --mcp-config flag is present and points at a real json file (now unlinked,
    # but we captured the path during the call)
    assert "--mcp-config" in captured["cmd"]
    assert captured["mcp_config_path"].endswith(".json")


def test_chat_with_tools_writes_correct_mcp_config(persona_dir: Path) -> None:
    """The temp mcp.json must contain the right command + args."""
    provider = ClaudeCliProvider()
    captured: dict = {}

    def _capture(cmd, **kwargs):
        path = cmd[cmd.index("--mcp-config") + 1]
        # Read the temp file BEFORE the provider's finally block deletes it
        captured["config"] = json.loads(Path(path).read_text())
        return _fake_proc(json.dumps({"result": "ok"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_capture):
        provider.chat(
            [
                ChatMessage(role="system", content="sys"),
                ChatMessage(role="user", content="hi"),
            ],
            tools=[{"name": "x", "description": "x"}],
            options={"persona_dir": str(persona_dir)},
        )

    cfg = captured["config"]
    assert "brain-tools" in cfg["mcpServers"]
    server_cfg = cfg["mcpServers"]["brain-tools"]
    assert server_cfg["args"][0] == "-m"
    assert server_cfg["args"][1] == "brain.mcp_server"
    assert "--persona-dir" in server_cfg["args"]
    assert str(persona_dir) in server_cfg["args"]


def test_chat_with_tools_keeps_existing_flags(persona_dir: Path) -> None:
    """The other flags (-p, --output-format, --model, --system-prompt)
    must remain — only --json-schema is replaced."""
    provider = ClaudeCliProvider()
    captured: dict = {}

    def _capture(cmd, **kwargs):
        captured["cmd"] = cmd
        return _fake_proc(json.dumps({"result": "ok"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_capture):
        provider.chat(
            [
                ChatMessage(role="system", content="sys-prompt"),
                ChatMessage(role="user", content="hi"),
            ],
            tools=[{"name": "x"}],
            options={"persona_dir": str(persona_dir)},
        )

    cmd = captured["cmd"]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--model" in cmd
    assert "--system-prompt" in cmd
    assert "sys-prompt" in cmd
    # The replaced flag must NOT appear
    assert "--json-schema" not in cmd


def test_chat_with_tools_parses_payload_result(persona_dir: Path) -> None:
    provider = ClaudeCliProvider()

    with patch(
        "brain.bridge.provider.subprocess.run",
        return_value=_fake_proc(json.dumps({"result": "the actual reply"})),
    ):
        response = provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"name": "x"}],
            options={"persona_dir": str(persona_dir)},
        )

    assert response.content == "the actual reply"
    assert response.tool_calls == ()


def test_chat_with_tools_missing_result_key_raises_parse(persona_dir: Path) -> None:
    provider = ClaudeCliProvider()

    with patch(
        "brain.bridge.provider.subprocess.run",
        return_value=_fake_proc(json.dumps({"different_key": "x"})),
    ):
        with pytest.raises(ProviderError) as ei:
            provider.chat(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "x"}],
                options={"persona_dir": str(persona_dir)},
            )
    assert ei.value.stage == "claude_cli_parse"


def test_chat_with_tools_nonzero_exit_raises_exit(persona_dir: Path) -> None:
    provider = ClaudeCliProvider()

    with patch(
        "brain.bridge.provider.subprocess.run",
        return_value=_fake_proc("", returncode=2, stderr="boom"),
    ):
        with pytest.raises(ProviderError) as ei:
            provider.chat(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "x"}],
                options={"persona_dir": str(persona_dir)},
            )
    assert ei.value.stage == "claude_cli_exit"


def test_chat_with_tools_missing_persona_dir_option_raises(tmp_path: Path) -> None:
    """tools= without options['persona_dir'] is a programmer bug — fail fast."""
    provider = ClaudeCliProvider()
    with pytest.raises(ProviderError) as ei:
        provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"name": "x"}],
            options={},  # missing persona_dir
        )
    assert ei.value.stage == "mcp_unavailable"


def test_chat_with_tools_cleans_up_temp_file(persona_dir: Path) -> None:
    """The temp mcp.json file must be unlinked after the call returns."""
    provider = ClaudeCliProvider()
    captured_path: list[str] = []

    def _capture(cmd, **kwargs):
        path = cmd[cmd.index("--mcp-config") + 1]
        captured_path.append(path)
        return _fake_proc(json.dumps({"result": "ok"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_capture):
        provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"name": "x"}],
            options={"persona_dir": str(persona_dir)},
        )

    assert len(captured_path) == 1
    assert not Path(captured_path[0]).exists()


def test_chat_without_tools_unchanged(persona_dir: Path) -> None:
    """When tools is None, the provider falls through the legacy text path —
    no --mcp-config, no persona_dir requirement."""
    provider = ClaudeCliProvider()

    with patch(
        "brain.bridge.provider.subprocess.run",
        return_value=_fake_proc(json.dumps({"result": "plain reply"})),
    ) as mock_run:
        response = provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=None,
        )

    assert response.content == "plain reply"
    cmd = mock_run.call_args.args[0]
    assert "--mcp-config" not in cmd
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/brain/bridge/test_provider_chat.py -v 2>&1 | tail -25`
Expected: many failures — old `--json-schema` code is still in `provider.py`.

- [ ] **Step 3: Rewrite the Claude tool path in `provider.py`**

Replace the `chat()` method body (lines 230-326) AND delete the three helper methods (`_build_tool_call_schema`, `_build_tool_system_addendum`, `_chat_with_tools` — lines 328 to roughly 460) with a single new `chat()` body that delegates to a slim `_chat_with_mcp_tools` helper.

Add this import at the top of `brain/bridge/provider.py` (with the other imports):

```python
import os
import sys
import tempfile
from pathlib import Path
```

Replace the existing `chat()` and the three helper methods with:

```python
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Flatten messages into Claude CLI; route through MCP when tools given.

        When ``tools`` is None:
          - Flattens messages into "User: ...\\nAssistant: ..." script via -p.
          - Returns ChatResponse(content=<text>, tool_calls=()).

        When ``tools`` is provided:
          - Requires ``options["persona_dir"]`` to point the MCP server at the
            active persona.
          - Writes a temp mcp.json, calls claude with --mcp-config, returns
            the final assistant text. Tool calls happen inside the claude
            subprocess; tool_calls on the response is always empty.

        Raises
        ------
        ProviderError("claude_cli_timeout", ...)
        ProviderError("claude_cli_exit", ...)
        ProviderError("claude_cli_parse", ...)
        ProviderError("mcp_unavailable", ...)
            When tools is non-None and options["persona_dir"] is missing,
            or when the mcp SDK is not importable.
        ProviderError("claude_cli_setup", ...)
            When the temp config file write fails.
        """
        system_prompt: str | None = None
        conversation_messages: list[ChatMessage] = []
        for msg in messages:
            if msg.role == "system" and system_prompt is None:
                system_prompt = msg.content
            else:
                conversation_messages.append(msg)

        if not conversation_messages:
            flat_prompt = ""
        elif len(conversation_messages) == 1:
            flat_prompt = conversation_messages[0].content
        else:
            parts: list[str] = []
            role_labels = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
            for msg in conversation_messages:
                label = role_labels.get(msg.role, msg.role.capitalize())
                parts.append(f"{label}: {msg.content}")
            flat_prompt = "\n".join(parts)

        if tools:
            persona_dir_str = (options or {}).get("persona_dir")
            if not persona_dir_str:
                raise ProviderError(
                    "mcp_unavailable",
                    "tool-calling via MCP requires options['persona_dir']",
                )
            return self._chat_with_mcp_tools(
                flat_prompt=flat_prompt,
                system_prompt=system_prompt,
                persona_dir=Path(persona_dir_str),
            )

        # ── Legacy text path (no tools) — unchanged from before SP-3 ──
        cmd = ["claude", "-p", flat_prompt, "--output-format", "json", "--model", self._model]
        if system_prompt is not None:
            cmd.extend(["--system-prompt", system_prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(
                "claude_cli_timeout",
                f"subprocess timed out after {self._timeout}s",
            ) from exc

        if result.returncode != 0:
            raise ProviderError(
                "claude_cli_exit",
                f"exit {result.returncode}: {result.stderr.strip()}",
            )

        try:
            payload = json.loads(result.stdout)
            content = str(payload["result"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ProviderError(
                "claude_cli_parse",
                f"unexpected output format: {result.stdout[:200]!r}",
            ) from exc

        return ChatResponse(content=content, tool_calls=(), raw=None)

    def _chat_with_mcp_tools(
        self,
        flat_prompt: str,
        system_prompt: str | None,
        persona_dir: Path,
    ) -> ChatResponse:
        """Tool-calling path: claude with --mcp-config pointing at brain.mcp_server.

        The mcp SDK is only imported here — keeps the legacy text path
        usable on systems without the SDK installed.
        """
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise ProviderError(
                "mcp_unavailable",
                "the 'mcp' SDK is required for the Claude tool-calling path. "
                "pip install 'mcp>=1.0.0,<2.0.0'",
            ) from exc

        config = {
            "mcpServers": {
                "brain-tools": {
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "brain.mcp_server",
                        "--persona-dir",
                        str(persona_dir),
                    ],
                    "env": {},
                }
            }
        }

        tmp_path: str | None = None
        try:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    json.dump(config, tmp)
                    tmp_path = tmp.name
            except OSError as exc:
                raise ProviderError(
                    "claude_cli_setup",
                    f"failed to write temp mcp.json: {exc}",
                ) from exc

            cmd = ["claude", "-p", flat_prompt, "--output-format", "json", "--model", self._model]
            if system_prompt is not None:
                cmd.extend(["--system-prompt", system_prompt])
            cmd.extend(["--mcp-config", tmp_path])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ProviderError(
                    "claude_cli_timeout",
                    f"subprocess timed out after {self._timeout}s",
                ) from exc

            if result.returncode != 0:
                raise ProviderError(
                    "claude_cli_exit",
                    f"exit {result.returncode}: {result.stderr.strip()}",
                )

            try:
                payload = json.loads(result.stdout)
                content = str(payload["result"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ProviderError(
                    "claude_cli_parse",
                    f"unexpected output format: {result.stdout[:200]!r}",
                ) from exc

            return ChatResponse(content=content, tool_calls=(), raw=None)
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
```

- [ ] **Step 4: Update `tool_loop.py` to pass `persona_dir` in options**

Edit `brain/chat/tool_loop.py` — find the `provider.chat(messages, tools=tools)` call inside `run_tool_loop` (around line 97). Replace the single call site with:

```python
        last_response = provider.chat(
            messages,
            tools=tools,
            options={"persona_dir": str(persona_dir)},
        )
```

The forced final pass (around line 143, the one that runs after the iteration cap with `tools=None`) should keep `tools=None` and stay unchanged — no MCP needed when tools are off.

- [ ] **Step 5: Run the new provider tests + tool_loop tests**

Run: `python3 -m pytest tests/unit/brain/bridge/test_provider_chat.py tests/unit/brain/chat/test_tool_loop.py -v 2>&1 | tail -20`
Expected: all pass.

- [ ] **Step 6: Run the full test suite**

Run: `python3 -m pytest 2>&1 | tail -10`
Expected: every test passes (no regressions in chat engine, ingest, soul, dream, heartbeat, reflex, research, health). If any test fails, read the failure carefully — the most likely cause is a test that mocked the OLD `_build_tool_call_schema` or `_chat_with_tools` symbols that no longer exist. Update those tests to use the new `_chat_with_mcp_tools` symbol or to mock `subprocess.run` instead.

- [ ] **Step 7: Commit**

```bash
git add brain/bridge/provider.py brain/chat/tool_loop.py \
        tests/unit/brain/bridge/test_provider_chat.py
git commit -m "feat(provider): swap --json-schema for --mcp-config

ClaudeCliProvider.chat() now writes a temp mcp.json and passes
--mcp-config <path> to the claude CLI when tools is provided.
The discriminated-union schema builder, tool-system addendum,
structured-output parser, and PR #28 off-schema fallback are all
removed — dead code under the new path. Tool-calling happens inside
the claude subprocess via brain.mcp_server, returning final text;
tool_calls on ChatResponse is always empty for Claude.

tool_loop.run_tool_loop now passes options={'persona_dir': ...} so
the provider knows where to point the spawned MCP server. Ollama
ignores the option; Claude requires it (raises mcp_unavailable
otherwise).

Closes the architectural gap surfaced by the 2026-04-27 live-exercise
stress test (0 tool invocations across 20 prompts because rich
voice.md outweighed --json-schema).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Verification — full suite + manual sandbox check

**Files:** none modified. This task is a quality gate per Hana's "we get the results we want before applying."

- [ ] **Step 1: Run the full test suite**

Run: `python3 -m pytest 2>&1 | tail -5`
Expected: all green.

- [ ] **Step 2: Run ruff**

Run: `python3 -m ruff check brain/ tests/ 2>&1 | tail -5`
Expected: no errors.

- [ ] **Step 3: Clone the live sandbox**

```bash
SANDBOX="$HOME/Library/Application Support/companion-emergence/personas/nell.sandbox"
CLONE="$HOME/Library/Application Support/companion-emergence/personas/nell.sandbox.mcp-test"
rm -rf "$CLONE"
cp -R "$SANDBOX" "$CLONE"
```

- [ ] **Step 4: Run a context-needing chat against the clone**

Run:
```bash
python3 -m brain.cli chat --persona nell.sandbox.mcp-test \
  "what's that thing you wrote about the morning after?"
```
Expected: Nell answers in her voice. The reply should reference real content, not a generic "I haven't written about that."

- [ ] **Step 5: Verify the audit log shows a `search_memories` call**

Run:
```bash
LOG="$HOME/Library/Application Support/companion-emergence/personas/nell.sandbox.mcp-test/tool_invocations.log.jsonl"
test -f "$LOG" && grep -c '"name": *"search_memories"' "$LOG"
```
Expected: ≥ 1 (at least one search_memories invocation logged).

If 0: the chat ran but tools didn't fire. Likely cause: voice.md doesn't direct Nell to use tools (Hana's lane to fix in her voice.md restructure), OR claude CLI version doesn't support `--mcp-config` (check `claude --help | grep mcp`). Either way, surface the finding and DO NOT MERGE — flag for Hana.

- [ ] **Step 6: Verify Nell's reply quotes a real memory**

Open `$LOG`, copy the `result_summary` from the search_memories invocation, then run:

```bash
python3 -c "
from pathlib import Path
from brain.memory.store import MemoryStore
db = Path.home() / 'Library/Application Support/companion-emergence/personas/nell.sandbox.mcp-test/memories.db'
store = MemoryStore(db_path=db)
hits = store.search('morning after', limit=5)
for h in hits:
    print(h.get('text', '')[:120])
"
```

Cross-check that at least one of those memory texts shows up in Nell's reply (paraphrase or direct quote — doesn't have to be verbatim, just verifiable).

- [ ] **Step 7: Delete the clone**

```bash
rm -rf "$CLONE"
```

- [ ] **Step 8: Push the branch + open the PR**

```bash
git push -u origin feat/mcp-brain-tools
gh pr create --title "feat(mcp): brain-tools MCP-config path replaces --json-schema" --body "$(cat <<'EOF'
## Summary

- Replaces Claude tool-calling via `--json-schema` (SP-3, PR #23) with the production-path `--mcp-config` documented in master ref §6 SP-3
- New `brain/mcp_server/` package exposes the 9 brain-tools to Claude as a stdio MCP server
- `ClaudeCliProvider` writes a temp `mcp.json`, swaps `--json-schema` for `--mcp-config`, returns final text
- Tool calls now happen inside the claude subprocess; our `tool_loop` runs a single pass for Claude (Ollama unchanged)
- New audit log at `<persona>/tool_invocations.log.jsonl` records every dispatch

## Why

The 2026-04-27 live-exercise stress test surfaced 0 tool invocations across 20 prompts. Root cause: rich `voice.md` outweighs `--json-schema` enforcement; the off-schema fallback (PR #28) silently returns plain text with empty `tool_calls`, so the loop returns immediately without dispatching anything. MCP is the architecture master ref §6 SP-3 already named as production-path; this PR ships it.

## Verification (per Hana — "we get the results we want before applying")

- [x] All unit tests green (~21 net new)
- [x] Manual sandbox-clone test: clone → context-needing chat → verify audit log shows `search_memories` → verify Nell's reply quotes a real memory → clone deleted

## Spec & plan

- Spec: `docs/superpowers/specs/2026-04-27-mcp-config-path-for-brain-tools-design.md`
- Plan: `docs/superpowers/plans/2026-04-27-mcp-config-path-for-brain-tools.md`

## Out of scope

- `voice.md` restructure — Hana's lane (mechanism ships first; policy lands after)
- Removing `tool_loop` — still needed for Ollama
- Daemon-mode MCP server — premature optimization

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 9: Confirm CI is green** before merge

Wait for the GitHub Actions run on the PR. If macOS / Windows / Linux all pass, the PR is mergeable.
