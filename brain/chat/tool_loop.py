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
import traceback
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any

from brain.attunement.budget import consume_call as _attunement_consume_call
from brain.attunement.crystallise import check_crystallisations
from brain.attunement.detector import run_detector, should_run_detector
from brain.attunement.store import (
    BufferTurn,
    mark_addressed,
    merge_into_learned,
    write_current_read,
)
from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.chat.extractor import apply_side_effects, extract_from_thinking
from brain.chat.reflection_gate import should_reflect
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools import NELL_TOOL_NAMES
from brain.tools.dispatch import dispatch
from brain.tools.schemas import build_schemas

logger = logging.getLogger(__name__)

_pass2_counter = count(1)
_attunement_counter = count(1)

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
                monologue_blocks=(monologue_text,),
                visible_reply=visible_reply,
                recent_turn_context=recent_user_msgs,
            )
            apply_side_effects(out, persona_dir=persona_dir)
        except Exception:  # noqa: BLE001
            logger.exception("pass-2 monologue extraction failed")

    threading.Thread(
        target=_run,
        daemon=True,
        name=f"monologue-extractor-{next(_pass2_counter)}",
    ).start()


def _attunement_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_attunement_error(persona_dir: Path, turn_id: str, exc: BaseException) -> None:
    path = persona_dir / "attunement_errors.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": _attunement_now_iso(),
        "turn_id": turn_id,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _run_attunement_pass2(
    persona_dir: Path,
    turn_id: str,
    user_message: str,
    reply_text: str,
    buffer_slice: list[BufferTurn],
) -> None:
    try:
        if not _attunement_consume_call(persona_dir, now=datetime.now(UTC)):
            return  # budget exhausted; defer silently
        output = run_detector(
            buffer_slice=buffer_slice,
            reply_text=reply_text,
            companion_name=persona_dir.name,
            persona_dir=persona_dir,
        )
        write_current_read(persona_dir, output.current_read)
        merge_into_learned(
            persona_dir,
            output.pattern_candidates,
            buffer_slice,
            now_iso=_attunement_now_iso(),
        )
        mark_addressed(persona_dir, output.addressed_pattern_ids, now_iso=_attunement_now_iso())
        check_crystallisations(persona_dir, now_iso=_attunement_now_iso())
    except Exception as exc:  # noqa: BLE001 — error isolation by design
        _log_attunement_error(persona_dir, turn_id, exc)


def _spawn_pass2_attunement(
    persona_dir: Path,
    turn_id: str,
    user_message: str,
    reply_text: str,
    buffer_slice: list[BufferTurn],
) -> None:
    """Spawn the async attunement pass-2 daemon (mirrors monologue _spawn_pass2)."""
    if not should_run_detector(buffer_slice, user_message, reply_text):
        return
    thread = threading.Thread(
        target=_run_attunement_pass2,
        args=(persona_dir, turn_id, user_message, reply_text, buffer_slice),
        name=f"attunement-extractor-{next(_attunement_counter)}",
        daemon=True,
    )
    thread.start()


def _buffer_slice_from_messages(messages: list[ChatMessage]) -> list[BufferTurn]:
    """Build a BufferTurn slice from the user messages in the conversation.

    Uses the message index as a stable turn id — suitable for grounding validation
    within this call (the ids only need to match within the buffer_slice passed to
    the same run_detector call).
    """
    return [
        BufferTurn(id=f"msg-{i}", content=m.content_text())
        for i, m in enumerate(messages)
        if m.role == "user" and m.content_text().strip()
    ]


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


def build_tools_list(
    companion_name: str = "Nell",
    *,
    allowed: list[str] | None = None,
) -> list[dict]:
    """Build the tool schema list for provider.chat(tools=...).

    Wraps schemas in the {"type": "function", "function": <schema>} shape
    that Ollama accepts natively and the MCP server registers as tool
    descriptions for the Claude path.

    Parameters
    ----------
    companion_name:
        The companion's name — used to parameterise tool descriptions.
    allowed:
        Optional allowlist of tool names.  When provided, only tools whose
        names appear in *allowed* are included.  ``None`` (default) returns
        the full suite; existing callers are unaffected.
    """
    schemas = build_schemas(companion_name)
    names: tuple[str, ...] | list[str]
    if allowed is None:
        names = NELL_TOOL_NAMES
    else:
        allowed_set = set(allowed)
        names = [n for n in NELL_TOOL_NAMES if n in allowed_set]
    return [{"type": "function", "function": schemas[name]} for name in names if name in schemas]


def _maybe_recruit_and_rerun(
    last_response: ChatResponse,
    invocations: list[dict],
    *,
    messages: list[ChatMessage],
    provider: LLMProvider,
    persona_dir: Path,
    companion_name: str,
    recruited_allowed: list[str] | None,
) -> ChatResponse | None:
    """If the model reached for a withheld faculty, re-invoke ONCE with the full
    tool suite so it can complete the reach this turn.

    Returns the new ChatResponse, or None to keep the original.
    Fails safe: errors → None (keep original reply, never crash the turn).
    The expansion is bounded to one because the re-run result is returned
    directly — it is NOT fed back into the recruit-check loop.

    Side effect: extends `invocations` in place with any tool calls the
    re-invoke dispatched.
    """
    reached = any(inv.get("name") == "reach_for_capability" for inv in invocations)
    if not reached or recruited_allowed is None:
        return None
    # Only expand if we were actually running a SLIM set (not already the full suite).
    if set(recruited_allowed) >= set(NELL_TOOL_NAMES):
        return None
    try:
        tools = build_tools_list(companion_name, allowed=list(NELL_TOOL_NAMES))
        # Tell the model it already reached, so it uses the real tool now instead of re-reaching.
        rerun_messages = list(messages) + [
            ChatMessage(
                role="user",
                content=("[The faculty you reached for is now available to you this turn. "
                         "Call the specific tool you need now (e.g. search_memories, read_file) — "
                         "do not call reach_for_capability again.]"),
            )
        ]
        rerun = provider.chat(rerun_messages, tools=tools, options={"persona_dir": str(persona_dir)})
        if rerun.dispatched_invocations:
            invocations.extend(rerun.dispatched_invocations)
        return rerun
    except Exception:  # noqa: BLE001 — fail safe: keep original reply, never crash the turn
        logger.exception("recruit-on-reach re-invoke failed; keeping original reply")
        return None


def run_tool_loop(
    messages: list[ChatMessage],
    *,
    provider: LLMProvider,
    tools: list[dict] | None,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
    companion_name: str = "Nell",
    recruited_allowed: list[str] | None = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    signal=None,
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
            # ONE-SHOT recruit-on-reach: if the model called reach_for_capability
            # while running on a slim tool set, re-invoke once with the full suite
            # so it can complete the action this turn.  Bounded to one expansion;
            # fails safe (error → keep original reply, never crash the turn).
            recruited = _maybe_recruit_and_rerun(
                last_response,
                invocations,
                messages=messages,
                provider=provider,
                persona_dir=persona_dir,
                companion_name=companion_name,
                recruited_allowed=recruited_allowed,
            )
            if recruited is not None:
                last_response = recruited
            # Pass-2 spawns fire against the FINAL response (after any recruit re-invoke).
            turn_index = sum(1 for m in messages if m.role == "user")
            monologue_text = _find_monologue_text(invocations)
            if monologue_text and (
                signal is None
                or should_reflect(signal, persona_dir, kind="monologue", turn_index=turn_index)
            ):
                _spawn_pass2(
                    provider=provider,
                    monologue_text=monologue_text,
                    visible_reply=last_response.content or "",
                    recent_user_msgs=tuple(
                        m.content_text() for m in messages if m.role == "user"
                    )[-2:],
                    persona_dir=persona_dir,
                )
            if signal is None or should_reflect(
                signal, persona_dir, kind="attunement", turn_index=turn_index
            ):
                _spawn_pass2_attunement(
                    persona_dir,
                    turn_id=f"turn-{len(messages)}",
                    user_message=next(
                        (m.content_text() for m in reversed(messages) if m.role == "user"), ""
                    ),
                    reply_text=last_response.content or "",
                    buffer_slice=_buffer_slice_from_messages(messages),
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
            try:
                result = dispatch(
                    tc.name,
                    tc.arguments,
                    store=store,
                    hebbian=hebbian,
                    persona_dir=persona_dir,
                )
                record["result_summary"] = _summarize_result(result)
                # record_monologue returns {"ok": True, "monologue_text": ...} on
                # success so _find_monologue_text can locate it and spawn pass 2.
                if isinstance(result, dict) and result.get("monologue_text"):
                    record["monologue_text"] = result["monologue_text"]
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
    turn_index = sum(1 for m in messages if m.role == "user")
    monologue_text = _find_monologue_text(invocations)
    if monologue_text and (
        signal is None
        or should_reflect(signal, persona_dir, kind="monologue", turn_index=turn_index)
    ):
        _spawn_pass2(
            provider=provider,
            monologue_text=monologue_text,
            visible_reply=last_response.content or "",
            recent_user_msgs=tuple(
                m.content_text() for m in messages if m.role == "user"
            )[-2:],
            persona_dir=persona_dir,
        )
    if signal is None or should_reflect(
        signal, persona_dir, kind="attunement", turn_index=turn_index
    ):
        _spawn_pass2_attunement(
            persona_dir,
            turn_id=f"turn-{len(messages)}-cap",
            user_message=next(
                (m.content_text() for m in reversed(messages) if m.role == "user"), ""
            ),
            reply_text=last_response.content or "",
            buffer_slice=_buffer_slice_from_messages(messages),
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
