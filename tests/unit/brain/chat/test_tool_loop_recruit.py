"""Task 2.3 — test build_tools_list(allowed=...) and recruit-on-reach."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.chat.tool_loop import build_tools_list, run_tool_loop
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools import NELL_TOOL_NAMES

# ---------------------------------------------------------------------------
# Stub provider for scripted sequential ChatResponse objects
# ---------------------------------------------------------------------------


class ScriptedProvider(LLMProvider):
    """Returns ChatResponses in order; records all chat() calls."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.chat_calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "stub generate"

    def name(self) -> str:
        return "scripted"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        self.chat_calls.append({"messages": list(messages), "tools": tools})
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return ChatResponse(content="fallback", tool_calls=(), raw=None)


def test_build_tools_list_filters_to_allowed():
    full = build_tools_list("Nell")
    slim = build_tools_list("Nell", allowed=["record_monologue", "reach_for_capability"])
    names = {t["function"]["name"] for t in slim}
    assert names == {"record_monologue", "reach_for_capability"}
    assert len(slim) < len(full)


def test_build_tools_list_default_is_full():
    assert len(build_tools_list("Nell")) == len(build_tools_list("Nell", allowed=None))


def test_recruit_on_reach_expands_and_reruns(tmp_path: Path) -> None:
    """When call #1 dispatched reach_for_capability, run_tool_loop must make
    a 2nd call with the FULL tool set and return that 2nd response's content."""
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    slim_allowed = ["record_monologue", "reach_for_capability"]

    response_1 = ChatResponse(
        content="I need more tools",
        tool_calls=(),
        dispatched_invocations=(
            {"name": "reach_for_capability", "arguments": {"capability": "memory"}},
        ),
        raw=None,
    )
    response_2 = ChatResponse(
        content="Now I can answer properly",
        tool_calls=(),
        dispatched_invocations=(),
        raw=None,
    )

    provider = ScriptedProvider([response_1, response_2])
    messages = [
        ChatMessage(role="system", content="You are Nell."),
        ChatMessage(role="user", content="Help me remember something."),
    ]
    slim_tools = build_tools_list("Nell", allowed=slim_allowed)

    final_resp, invocations = run_tool_loop(
        messages,
        provider=provider,
        tools=slim_tools,
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
        companion_name="Nell",
        recruited_allowed=slim_allowed,
    )

    # Exactly 2 provider.chat calls
    assert len(provider.chat_calls) == 2, f"Expected 2 chat calls, got {len(provider.chat_calls)}"

    # 2nd call used the FULL tool suite
    second_tool_names = {t["function"]["name"] for t in (provider.chat_calls[1]["tools"] or [])}
    assert set(NELL_TOOL_NAMES).issubset(second_tool_names), (
        f"2nd call tools missing: {set(NELL_TOOL_NAMES) - second_tool_names}"
    )

    # Final content is from call #2
    assert final_resp.content == "Now I can answer properly"

    # reach invocation appears in the returned list
    inv_names = [inv.get("name") for inv in invocations]
    assert "reach_for_capability" in inv_names


def test_no_reach_no_expansion(tmp_path: Path) -> None:
    """When call #1 has NO reach_for_capability, only 1 provider.chat call."""
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    slim_allowed = ["record_monologue", "reach_for_capability"]

    response_1 = ChatResponse(
        content="Simple answer",
        tool_calls=(),
        dispatched_invocations=(),
        raw=None,
    )

    provider = ScriptedProvider([response_1])
    messages = [
        ChatMessage(role="system", content="You are Nell."),
        ChatMessage(role="user", content="Hello."),
    ]
    slim_tools = build_tools_list("Nell", allowed=slim_allowed)

    final_resp, _ = run_tool_loop(
        messages,
        provider=provider,
        tools=slim_tools,
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
        companion_name="Nell",
        recruited_allowed=slim_allowed,
    )

    assert len(provider.chat_calls) == 1, f"Expected 1 chat call, got {len(provider.chat_calls)}"
    assert final_resp.content == "Simple answer"
