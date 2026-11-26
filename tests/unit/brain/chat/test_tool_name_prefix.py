"""#417/#419: text the model reads names each brain tool by the name its backend registers.

On the Claude CLI the brain tools are `mcp__brain-tools__<name>`; on Ollama they are bare. The
prefix is decided once per call from the provider and inserted when each text is built (system
prompt, volatile tail, tool list, tool descriptions, tool results), never by rewriting the
finished prompt, so conversation history is left alone. voice.md keeps plain names on disk and
gets the prefix when it is loaded into the prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.chat.engine import respond
from brain.chat.session import create_session, reset_registry
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.tools import NELL_TOOL_NAMES
from brain.tools.naming import TOOL_PREFIX_SLOT, fill_tool_prefix, prefix_tool_names

MCP = "mcp__brain-tools__"
_NAMES = "|".join(sorted(NELL_TOOL_NAMES, key=len, reverse=True))
# A tool name not preceded by the prefix (and not part of a longer word).
_UNPREFIXED = re.compile(rf"(?<![\w-])(?<!{re.escape(MCP)})({_NAMES})(?!\w)")


# Below the add_memory gate, so its reply names add_journal.
_SMALL_MEMORY = {"content": "small", "memory_type": "event", "domain": "us", "emotions": {}}


def _unprefixed(text: str) -> list[str]:
    # "boot" is also an English word ("Full session boot"); only its quoted form is a tool name.
    return [m.group(0) for m in _UNPREFIXED.finditer(text) if m.group(0) != "boot"]


def test_fill_tool_prefix_inserts_the_prefix_at_every_slot() -> None:
    text = f"call {TOOL_PREFIX_SLOT}add_name, or {TOOL_PREFIX_SLOT}read_file"
    assert fill_tool_prefix(text, MCP) == f"call {MCP}add_name, or {MCP}read_file"
    assert fill_tool_prefix(text, "") == "call add_name, or read_file"


def test_prefix_tool_names_reaches_quoted_and_called_names_only() -> None:
    voice = (
        "When `get_soul` returns, call `record_monologue(monologue, feed_digest)`. "
        "A session boot is not a tool; neither is `WebSearch`."
    )
    out = prefix_tool_names(voice, MCP)
    assert f"`{MCP}get_soul`" in out
    assert f"`{MCP}record_monologue(" in out
    assert "session boot is" in out and "`WebSearch`" in out
    assert prefix_tool_names(out, MCP) == out, "never prefixes twice"
    assert prefix_tool_names(voice, "") == voice


def test_providers_declare_their_prefix() -> None:
    from brain.bridge.provider import ClaudeCliProvider, FakeProvider, OllamaProvider

    assert ClaudeCliProvider.tool_name_prefix == MCP
    assert OllamaProvider.tool_name_prefix == ""
    assert FakeProvider.tool_name_prefix == ""


@pytest.mark.parametrize("prefix", ["", MCP])
@pytest.mark.parametrize("companion", ["Nell", "Canary"])
def test_tool_descriptions_name_tools_with_the_prefix(prefix: str, companion: str) -> None:
    from brain.tools.schemas import build_schemas

    for name, schema in build_schemas(companion, tool_prefix=prefix).items():
        props = schema.get("parameters", {}).get("properties", {}).values()
        for text in [schema.get("description", ""), *(p.get("description", "") for p in props)]:
            assert TOOL_PREFIX_SLOT not in text, name
            if prefix:
                assert not _unprefixed(text), (name, _unprefixed(text))


class _Recording(LLMProvider):
    tool_name_prefix = MCP

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []
        self.options: dict[str, Any] = {}
        self.tools: list[dict] = []

    def name(self) -> str:
        return "recording"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "ok"

    def chat(self, messages, *, tools=None, options=None) -> ChatResponse:
        self.messages, self.options, self.tools = (
            list(messages),
            dict(options or {}),
            list(tools or []),
        )
        return ChatResponse(content="hello", tool_calls=())


def _respond(tmp_path: Path, provider: LLMProvider) -> None:
    """One real respond() turn with a memory the recall block will surface as a snippet."""
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop"}), encoding="utf-8"
    )
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    store.create(
        Memory.create_new(
            content="Hana mentioned Jordan once over coffee.",
            memory_type="event",
            domain="relationship",
            emotions={"love": 6.0},
            tags=[],
        )
    )
    reset_registry()
    try:
        respond(
            persona_dir,
            "Tell me what we said about Jordan last time.",
            store=store,
            hebbian=hebbian,
            provider=provider,
            session=create_session(persona_dir.name),
            voice_md_override="# Nell\n\nWhen `get_soul` returns crystallisations, they are yours.",
        )
    finally:
        reset_registry()
        hebbian.close()
        store.close()


def test_claude_cli_turn_names_every_tool_with_the_mcp_prefix(tmp_path: Path) -> None:
    provider = _Recording()
    _respond(tmp_path, provider)

    system = provider.messages[0].content
    volatile = provider.options["volatile_suffix"]
    for label, text in (("system", system), ("volatile", volatile)):
        assert TOOL_PREFIX_SLOT not in text, label
        assert not _unprefixed(text), (label, _unprefixed(text))
    assert f"`{MCP}get_soul`" in system  # voice.md, prefixed at load
    assert f"{MCP}read_full_memory" in system  # epistemic instruction
    assert f"`{MCP}boot`" in system  # tool inventory entry
    assert f"`{MCP}reach_for_capability`" in system  # reach valve
    assert f"`{MCP}record_monologue(" in system  # monologue frame
    assert f"{MCP}read_full_memory to reach" in volatile  # recall snippet invitation
    assert f"`{MCP}record_monologue` this turn" in volatile  # reply frame
    for tool in provider.tools:
        assert not _unprefixed(tool["function"]["description"]), tool["function"]["name"]


def test_ollama_turn_keeps_bare_names_and_leaves_no_slot(tmp_path: Path) -> None:
    provider = _Recording()
    provider.tool_name_prefix = ""
    _respond(tmp_path, provider)

    system = provider.messages[0].content
    texts = [system, provider.options["volatile_suffix"]]
    texts += [t["function"]["description"] for t in provider.tools]
    for text in texts:
        assert MCP not in text
        assert TOOL_PREFIX_SLOT not in text
    assert "`get_soul`" in system and "open it with read_full_memory" in system
    # The user's own words go out as typed: history is never rewritten.
    assert [m for m in provider.messages if m.role == "user"][-1].content == (
        "Tell me what we said about Jordan last time."
    )


@pytest.mark.parametrize("prefix", ["", MCP])
def test_in_process_tool_result_names_tools_with_the_provider_prefix(
    tmp_path: Path, prefix: str
) -> None:
    """A tool result that names another tool (the add_memory gate's reason) carries the prefix."""
    from brain.bridge.chat import ToolCall
    from brain.chat.tool_loop import run_tool_loop

    class _OneCall(_Recording):
        def chat(self, messages, *, tools=None, options=None) -> ChatResponse:
            self.messages = list(messages)
            if not any(m.role == "tool" for m in messages):
                call = ToolCall(id="c1", name="add_memory", arguments=_SMALL_MEMORY)
                return ChatResponse(content="", tool_calls=(call,))
            return ChatResponse(content="done", tool_calls=())

    provider = _OneCall()
    provider.tool_name_prefix = prefix
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    try:
        run_tool_loop(
            [ChatMessage(role="user", content="hi")],
            provider=provider,
            tools=None,
            store=store,
            hebbian=hebbian,
            persona_dir=tmp_path,
        )
    finally:
        hebbian.close()
        store.close()
    result = next(m.content for m in provider.messages if m.role == "tool")
    assert f"use {prefix}add_journal for" in result, result
    assert TOOL_PREFIX_SLOT not in result


def test_mcp_server_describes_and_answers_with_its_tool_prefix(tmp_path: Path) -> None:
    """The MCP child the CLI spawns serves descriptions and results in the CLI's names."""
    import asyncio
    from unittest.mock import MagicMock, patch

    from mcp.server import Server
    from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest

    from brain.mcp_server.tools import register_tools

    server = Server("brain-tools")
    gated = {
        "created": False,
        "reason": f"use {TOOL_PREFIX_SLOT}add_journal for low-weight content",
        "content": f"as written: {TOOL_PREFIX_SLOT}",
    }
    with patch("brain.mcp_server.tools.dispatch", return_value=gated):
        register_tools(
            server, persona_dir=tmp_path, store=MagicMock(), hebbian=MagicMock(), tool_prefix=MCP
        )
        listed = asyncio.run(server.request_handlers[ListToolsRequest](MagicMock()))
        call = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name="add_memory", arguments=_SMALL_MEMORY),
        )
        answered = asyncio.run(server.request_handlers[CallToolRequest](call))

    for tool in listed.root.tools:
        assert not _unprefixed(tool.description), (tool.name, _unprefixed(tool.description))
    text = answered.root.content[0].text
    assert f"use {MCP}add_journal for" in text
    assert f"as written: {TOOL_PREFIX_SLOT}" in text  # content is not the brain's wording


def test_cli_spawn_entry_hands_the_mcp_child_its_prefix(tmp_path: Path) -> None:
    from brain.bridge.provider import brain_tools_mcp_entry
    from brain.mcp_server.__main__ import main

    args = brain_tools_mcp_entry(tmp_path)["args"]
    assert args[args.index("--tool-prefix") + 1] == MCP
    with patch("brain.mcp_server.__main__.run_server") as run:
        main(args[args.index("brain.mcp_server") + 1 :])
    run.assert_called_once_with(tmp_path, tool_prefix=MCP)


def test_prompt_strings_name_tools_through_the_slot() -> None:
    """Canary: a tool named in prompt_strings.toml gets the prefix, or (voice text) is backticked
    so prefix_tool_names reaches it. A bare name would reach the CLI as a tool it cannot call."""
    import tomllib

    def strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict | list):
            for v in node.values() if isinstance(node, dict) else node:
                yield from strings(v)

    doc = tomllib.loads(
        (Path(__file__).resolve().parents[4] / "brain" / "prompt_strings.toml").read_text(
            encoding="utf-8"
        )
    )
    voice = doc["chat"].pop("voice")
    filled = [fill_tool_prefix(s, MCP) for s in strings([doc["chat"], doc["tools"]])]
    assert not [n for s in filled for n in _unprefixed(s)]
    assert not [n for s in strings(voice) for n in _unprefixed(prefix_tool_names(s, MCP))]


def test_the_app_streaming_proxy_passes_the_real_providers_prefix_through() -> None:
    """Every app chat turn goes through _StreamingProxy; it must not fall back to bare names."""
    from brain.bridge.provider import ClaudeCliProvider
    from brain.bridge.server import _StreamingProxy

    assert _StreamingProxy(ClaudeCliProvider(), None, None).tool_name_prefix == MCP


def test_tool_content_is_left_as_written_and_records_carry_no_slot(tmp_path: Path) -> None:
    """Only the texts the brain writes are filled: a file that itself contains the slot is read
    back as written, and the invocation record (logs, the WS summary) never shows the slot."""
    from brain.bridge.chat import ToolCall
    from brain.chat.tool_loop import run_tool_loop

    file_text = {"content": f"literal {TOOL_PREFIX_SLOT} in a user's file"}
    gated = {"created": False, "reason": f"use {TOOL_PREFIX_SLOT}add_journal for low-weight content"}

    class _TwoCalls(_Recording):
        def chat(self, messages, *, tools=None, options=None) -> ChatResponse:
            self.messages = list(messages)
            if not any(m.role == "tool" for m in messages):
                calls = (ToolCall(id="a", name="read_file", arguments={}), ToolCall(id="b", name="add_memory", arguments={}))
                return ChatResponse(content="", tool_calls=calls)
            return ChatResponse(content="done", tool_calls=())

    provider = _TwoCalls()
    results = {"read_file": file_text, "add_memory": gated}
    with patch("brain.chat.tool_loop.dispatch", side_effect=lambda name, *a, **k: results[name]):
        _, invocations = run_tool_loop(
            [ChatMessage(role="user", content="hi")],
            provider=provider,
            tools=None,
            store=MagicMock(),
            hebbian=MagicMock(),
            persona_dir=tmp_path,
        )
    read, gate = (m.content for m in provider.messages if m.role == "tool")
    assert f"literal {TOOL_PREFIX_SLOT} in a user's file" in read
    assert f"use {MCP}add_journal" in gate
    assert all(TOOL_PREFIX_SLOT not in inv["result_summary"] for inv in invocations[1:])
