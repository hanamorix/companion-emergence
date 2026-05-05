"""SP-6 tool loop — provider.chat() → dispatch → repeat.

Ported from OG NellBrain/nell_bridge.py:run_tool_loop (lines 243-301).

Key differences from OG:
  - Uses brain.bridge.chat.ChatMessage / ChatResponse typed objects (not dicts)
  - Uses brain.tools.dispatch.dispatch() (not nell_tools.dispatch())
  - No model param — provider encapsulates the model choice per SP-1
  - tool_calls is a tuple[ToolCall, ...] on ChatResponse (not raw dicts)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools import NELL_TOOL_NAMES
from brain.tools.dispatch import dispatch
from brain.tools.schemas import SCHEMAS

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 4


def build_tools_list() -> list[dict]:
    """Build the tool schema list for provider.chat(tools=...).

    Wraps schemas in the {"type": "function", "function": <schema>} shape
    that Ollama accepts natively and the MCP server registers as tool
    descriptions for the Claude path.
    """
    return [
        {"type": "function", "function": SCHEMAS[name]}
        for name in NELL_TOOL_NAMES
        if name in SCHEMAS
    ]


def run_tool_loop(
    messages: list[ChatMessage],
    *,
    provider: LLMProvider,
    tools: list[dict] | None,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> tuple[ChatResponse, list[dict]]:
    """Loop: provider.chat() → if tool_calls, dispatch each → retry.

    Up to max_iterations. On final iteration if still requesting tool_calls,
    force one more pass with tools=None to obligate a content response.

    Parameters
    ----------
    messages:
        Current conversation history including the system message and
        the user's latest turn. Modified in-place as tool calls are appended.
    provider:
        LLMProvider instance — encapsulates model and API surface.
    tools:
        Tool schemas list (from build_tools_list()) or None to disable tools.
    store, hebbian, persona_dir:
        Injected into each tool dispatch call.
    max_iterations:
        Maximum tool-call cycles before forcing a content-only response.

    Returns
    -------
    (final_response, invocations)
      - final_response: ChatResponse with content set (tool_calls may be empty)
      - invocations: list of dicts, each: {name, arguments, result_summary, error?}
    """
    invocations: list[dict[str, Any]] = []
    last_response = ChatResponse(content="", tool_calls=(), raw=None)

    for _iteration in range(max_iterations):
        last_response = provider.chat(
            messages,
            tools=tools,
            options={"persona_dir": str(persona_dir)},
        )
        # Provider-dispatched invocations (claude-cli MCP path): tools
        # already ran inside the subprocess. Surface them for telemetry
        # without re-dispatching.
        if last_response.dispatched_invocations:
            invocations.extend(last_response.dispatched_invocations)
        if not last_response.tool_calls:
            return last_response, invocations

        # Append the assistant turn that contained the tool_calls so the
        # model can see its own request on the next iteration.
        assistant_turn = ChatMessage(
            role="assistant",
            content=last_response.content or "",
            tool_calls=last_response.tool_calls,
        )
        messages.append(assistant_turn)

        for tc in last_response.tool_calls:
            record: dict[str, Any] = {
                "name": tc.name,
                "arguments": tc.arguments,
            }
            try:
                result = dispatch(
                    tc.name,
                    tc.arguments,
                    store=store,
                    hebbian=hebbian,
                    persona_dir=persona_dir,
                )
                record["result_summary"] = _summarize_result(result)
                tool_content = json.dumps(result, default=str, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("tool dispatch error: %s — %s", tc.name, exc)
                record["error"] = str(exc)
                record["result_summary"] = f"error: {exc}"
                tool_content = json.dumps({"error": str(exc)})

            invocations.append(record)
            messages.append(
                ChatMessage(
                    role="tool",
                    content=tool_content,
                    tool_call_id=tc.id,
                )
            )

    # Hit iteration cap — force a final pass with no tools so the model
    # is obligated to produce a content response (per OG pattern).
    logger.warning("tool loop hit max_iterations=%d", max_iterations)
    last_response = provider.chat(messages, tools=None)
    return last_response, invocations


def _summarize_result(result: Any, max_chars: int = 140) -> str:
    """Compact single-line preview for invocation metadata."""
    try:
        s = json.dumps(result, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(result)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= max_chars else s[:max_chars] + "…"
