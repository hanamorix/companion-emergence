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
import threading
from pathlib import Path
from typing import Any

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.chat.extractor import apply_side_effects, extract_from_thinking
from brain.chat.monologue_capture import CaptureRejected, capture_monologue
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools import NELL_TOOL_NAMES
from brain.tools.dispatch import dispatch
from brain.tools.schemas import build_schemas

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 4


def _spawn_pass2(
    *,
    provider: LLMProvider,
    monologue_text: str,
    visible_reply: str,
    recent_user_msgs: tuple[str, ...],
    persona_dir: Path,
) -> None:
    """Fire-and-forget pass 2: Haiku reads the monologue, writes memory/emotion/soul/reflex_audit."""

    def _run() -> None:
        try:
            out = extract_from_thinking(
                provider=provider,
                thinking_blocks=(monologue_text,),
                visible_reply=visible_reply,
                recent_turn_context=recent_user_msgs,
            )
            apply_side_effects(out, persona_dir=persona_dir)
        except Exception:  # noqa: BLE001
            logger.exception("pass-2 monologue extraction failed")

    threading.Thread(target=_run, daemon=True, name="monologue-extractor").start()


def _find_monologue_text(invocations: list[dict]) -> str | None:
    """Scan invocations for a captured record_monologue entry."""
    return next(
        (
            inv.get("monologue_text")
            for inv in invocations
            if inv.get("name") == "record_monologue" and inv.get("monologue_text")
        ),
        None,
    )


def build_tools_list(companion_name: str = "Nell") -> list[dict]:
    """Build the tool schema list for provider.chat(tools=...).

    Wraps schemas in the {"type": "function", "function": <schema>} shape
    that Ollama accepts natively and the MCP server registers as tool
    descriptions for the Claude path.
    """
    schemas = build_schemas(companion_name)
    return [
        {"type": "function", "function": schemas[name]}
        for name in NELL_TOOL_NAMES
        if name in schemas
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
            monologue_text = _find_monologue_text(invocations)
            if monologue_text:
                _spawn_pass2(
                    provider=provider,
                    monologue_text=monologue_text,
                    visible_reply=last_response.content or "",
                    recent_user_msgs=tuple(
                        m.content_text() for m in messages if m.role == "user"
                    )[-2:],
                    persona_dir=persona_dir,
                )
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
            if tc.name == "record_monologue":
                try:
                    monologue_text = capture_monologue(
                        persona_dir=persona_dir,
                        monologue=tc.arguments.get("monologue", ""),
                        feed_digest=tc.arguments.get("feed_digest", ""),
                    )
                    record["result_summary"] = "captured"
                    record["monologue_text"] = monologue_text
                    tool_content = json.dumps({"ok": True})
                    invocations.append(record)
                    messages.append(
                        ChatMessage(role="tool", content=tool_content, tool_call_id=tc.id)
                    )
                    continue
                except CaptureRejected as exc:
                    record["error"] = str(exc)
                    record["result_summary"] = f"rejected: {exc}"
                    tool_content = json.dumps({"error": str(exc)})
                    invocations.append(record)
                    messages.append(
                        ChatMessage(role="tool", content=tool_content, tool_call_id=tc.id)
                    )
                    continue
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
    monologue_text = _find_monologue_text(invocations)
    if monologue_text:
        _spawn_pass2(
            provider=provider,
            monologue_text=monologue_text,
            visible_reply=last_response.content or "",
            recent_user_msgs=tuple(
                m.content_text() for m in messages if m.role == "user"
            )[-2:],
            persona_dir=persona_dir,
        )
    return last_response, invocations


def _summarize_result(result: Any, max_chars: int = 140) -> str:
    """Compact single-line preview for invocation metadata."""
    try:
        s = json.dumps(result, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(result)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= max_chars else s[:max_chars] + "…"
