"""Tests for brain.chat.tool_loop — run_tool_loop() + build_tools_list()."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse, ToolCall
from brain.bridge.provider import FakeProvider, LLMProvider
from brain.chat.tool_loop import build_tools_list, run_tool_loop
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture()
def hebbian(tmp_path: Path) -> HebbianMatrix:
    h = HebbianMatrix(db_path=":memory:")
    yield h
    h.close()


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    return d


def _make_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="You are nell."),
        ChatMessage(role="user", content="Hello"),
    ]


# ── build_tools_list ──────────────────────────────────────────────────────────


def test_build_tools_list_produces_correct_shape() -> None:
    tools = build_tools_list()
    assert len(tools) > 0
    for t in tools:
        assert t["type"] == "function"
        assert "function" in t
        assert "name" in t["function"]
        assert "parameters" in t["function"]


def test_build_tools_list_contains_expected_tools() -> None:
    tools = build_tools_list()
    names = {t["function"]["name"] for t in tools}
    expected = {
        "get_emotional_state",
        "get_soul",
        "boot",
        "search_memories",
        "add_journal",
    }
    assert expected.issubset(names)


# ── run_tool_loop — no tool calls ─────────────────────────────────────────────


def test_run_tool_loop_no_tool_calls_returns_immediately(
    store: MemoryStore, hebbian: HebbianMatrix, persona_dir: Path
) -> None:
    """FakeProvider never produces tool calls → returns on first iteration."""
    messages = _make_messages()
    provider = FakeProvider()
    tools = build_tools_list()

    response, invocations = run_tool_loop(
        messages,
        provider=provider,
        tools=tools,
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
    )
    assert response.content.startswith("FAKE_CHAT")
    assert invocations == []


def test_run_tool_loop_with_none_tools_returns_immediately(
    store: MemoryStore, hebbian: HebbianMatrix, persona_dir: Path
) -> None:
    """tools=None → single call, no iteration."""
    messages = _make_messages()
    provider = FakeProvider()

    response, invocations = run_tool_loop(
        messages,
        provider=provider,
        tools=None,
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
    )
    assert response.content
    assert invocations == []


# ── run_tool_loop — tool dispatch ─────────────────────────────────────────────


class _ProviderWithOneToolCall(LLMProvider):
    """First call returns a tool_call; second call returns plain content."""

    def __init__(self) -> None:
        self._call_count = 0

    def name(self) -> str:
        return "fake-with-tool"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "not used"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        self._call_count += 1
        if self._call_count == 1:
            return ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="tc-001",
                        name="get_emotional_state",
                        arguments={},
                    ),
                ),
                raw=None,
            )
        return ChatResponse(content="Final answer after tool.", tool_calls=(), raw=None)


def test_run_tool_loop_dispatches_tool_call_and_continues(
    store: MemoryStore, hebbian: HebbianMatrix, persona_dir: Path
) -> None:
    """Provider returns one tool call, then content. Loop runs twice."""
    messages = _make_messages()
    provider = _ProviderWithOneToolCall()
    tools = build_tools_list()

    response, invocations = run_tool_loop(
        messages,
        provider=provider,
        tools=tools,
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
    )
    assert response.content == "Final answer after tool."
    assert len(invocations) == 1
    assert invocations[0]["name"] == "get_emotional_state"
    assert "result_summary" in invocations[0]


def test_run_tool_loop_preserves_invocation_order(
    store: MemoryStore, hebbian: HebbianMatrix, persona_dir: Path
) -> None:
    """Multiple tool calls in one turn → invocations listed in order."""

    class _MultiToolProvider(LLMProvider):
        def __init__(self) -> None:
            self._call_count = 0

        def name(self) -> str:
            return "multi-tool"

        def generate(self, prompt: str, *, system: str | None = None) -> str:
            return ""

        def chat(
            self,
            messages: list[ChatMessage],
            *,
            tools: list[dict[str, Any]] | None = None,
            options: dict[str, Any] | None = None,
        ) -> ChatResponse:
            self._call_count += 1
            if self._call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=(
                        ToolCall(id="tc-a", name="get_emotional_state", arguments={}),
                        ToolCall(id="tc-b", name="get_soul", arguments={}),
                    ),
                    raw=None,
                )
            return ChatResponse(content="Done.", tool_calls=(), raw=None)

    messages = _make_messages()
    provider = _MultiToolProvider()
    response, invocations = run_tool_loop(
        messages,
        provider=provider,
        tools=build_tools_list(),
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
    )
    assert len(invocations) == 2
    assert invocations[0]["name"] == "get_emotional_state"
    assert invocations[1]["name"] == "get_soul"


def test_run_tool_loop_tool_dispatch_error_sets_error_field(
    store: MemoryStore, hebbian: HebbianMatrix, persona_dir: Path
) -> None:
    """Unknown tool name → error field in invocation, loop still resolves."""

    class _BadToolProvider(LLMProvider):
        def __init__(self) -> None:
            self._call_count = 0

        def name(self) -> str:
            return "bad-tool"

        def generate(self, prompt: str, *, system: str | None = None) -> str:
            return ""

        def chat(
            self,
            messages: list[ChatMessage],
            *,
            tools: list[dict[str, Any]] | None = None,
            options: dict[str, Any] | None = None,
        ) -> ChatResponse:
            self._call_count += 1
            if self._call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=(ToolCall(id="tc-x", name="nonexistent_tool", arguments={}),),
                    raw=None,
                )
            return ChatResponse(content="Recovered.", tool_calls=(), raw=None)

    messages = _make_messages()
    provider = _BadToolProvider()
    response, invocations = run_tool_loop(
        messages,
        provider=provider,
        tools=build_tools_list(),
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
    )
    assert "error" in invocations[0]
    assert response.content == "Recovered."


def test_run_tool_loop_max_iterations_forces_no_tools_final_pass(
    store: MemoryStore, hebbian: HebbianMatrix, persona_dir: Path
) -> None:
    """Provider always returns tool_calls → hit cap → final pass with tools=None."""
    call_count = 0
    tools_on_final: list[Any] = []

    class _InfiniteToolProvider(LLMProvider):
        def name(self) -> str:
            return "infinite-tool"

        def generate(self, prompt: str, *, system: str | None = None) -> str:
            return ""

        def chat(
            self,
            messages: list[ChatMessage],
            *,
            tools: list[dict[str, Any]] | None = None,
            options: dict[str, Any] | None = None,
        ) -> ChatResponse:
            nonlocal call_count
            call_count += 1
            if tools is None:
                tools_on_final.append(True)
                return ChatResponse(content="Forced final.", tool_calls=(), raw=None)
            return ChatResponse(
                content="",
                tool_calls=(ToolCall(id=f"tc-{call_count}", name="get_soul", arguments={}),),
                raw=None,
            )

    messages = _make_messages()
    provider = _InfiniteToolProvider()
    response, invocations = run_tool_loop(
        messages,
        provider=provider,
        tools=build_tools_list(),
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
        max_iterations=2,
    )
    # Final forced call happened
    assert tools_on_final
    assert response.content == "Forced final."
    # Invocations capped to max_iterations calls
    assert len(invocations) == 2
