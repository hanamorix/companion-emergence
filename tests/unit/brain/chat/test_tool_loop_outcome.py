"""#96 — a refused tool call must be distinguishable from a committed one.

write_guard RETURNS {"error": ...} rather than raising, so a refused
propose_write reached invocations looking identical to a success.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from brain.bridge.chat import ChatMessage, ChatResponse, ToolCall
from brain.bridge.provider import LLMProvider
from brain.chat.tool_loop import build_tools_list, run_tool_loop
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.chat_calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "stub generate"

    def name(self) -> str:
        return "scripted"

    def chat(self, messages, *, tools=None, options=None) -> ChatResponse:
        self.chat_calls.append({"messages": list(messages), "tools": tools})
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return ChatResponse(content="fallback", tool_calls=(), raw=None)


def _run(provider, tmp_path):
    return run_tool_loop(
        [ChatMessage(role="user", content="write that down")],
        provider=provider,
        tools=build_tools_list("Nell"),
        store=MemoryStore(":memory:"),
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=tmp_path,
        companion_name="Nell",
    )


def test_guard_refused_write_is_marked_refused(tmp_path: Path):
    """A propose_write the guard denies must carry outcome='refused'."""
    denied = ToolCall(
        id="c1",
        name="propose_write",
        arguments={"path": "/etc/passwd", "content": "x", "op": "append"},
    )
    provider = ScriptedProvider([
        ChatResponse(content="", tool_calls=(denied,), raw=None),
        ChatResponse(content="done", tool_calls=(), raw=None),
    ])
    _resp, invocations = _run(provider, tmp_path)

    write_invs = [i for i in invocations if i.get("name") == "propose_write"]
    assert len(write_invs) == 1, f"expected one propose_write record, got {invocations}"
    assert write_invs[0].get("outcome") == "refused", (
        f"guard-refused write must be marked refused, got {write_invs[0]}"
    )


def test_successful_call_is_marked_ok(tmp_path: Path):
    """A tool that returns normally carries outcome='ok'."""
    call = ToolCall(
        id="c1",
        name="record_monologue",
        arguments={"monologue": "a thought", "feed_digest": "she thought"},
    )
    provider = ScriptedProvider([
        ChatResponse(content="", tool_calls=(call,), raw=None),
        ChatResponse(content="done", tool_calls=(), raw=None),
    ])
    _resp, invocations = _run(provider, tmp_path)

    mono = [i for i in invocations if i.get("name") == "record_monologue"]
    assert len(mono) == 1
    assert mono[0].get("outcome") == "ok", f"expected ok, got {mono[0]}"


def test_unknown_tool_is_marked_error(tmp_path: Path):
    """A dispatch that raises carries outcome='error' and keeps record['error']."""
    call = ToolCall(id="c1", name="no_such_tool_at_all", arguments={})
    provider = ScriptedProvider([
        ChatResponse(content="", tool_calls=(call,), raw=None),
        ChatResponse(content="done", tool_calls=(), raw=None),
    ])
    _resp, invocations = _run(provider, tmp_path)

    rec = [i for i in invocations if i.get("name") == "no_such_tool_at_all"]
    assert len(rec) == 1
    assert rec[0].get("outcome") == "error", f"expected error, got {rec[0]}"
    assert rec[0].get("error"), "the pre-existing error field must still be set"
