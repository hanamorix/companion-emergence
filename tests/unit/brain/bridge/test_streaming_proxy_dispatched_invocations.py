"""Streaming-path regression: _StreamingProxy.chat() must populate dispatched_invocations.

The bug we hit live in v0.0.26 final hotfix: the streaming path silently dropped
MCP audit-log entries, so run_tool_loop's _spawn_pass2 never fired and the
monologue feature was dead for all real users. This test pins the audit-log
read so no future change can silently regress.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path

from brain.bridge.chat import ChatMessage, ChatResponse, StreamDone, TextDelta
from brain.bridge.server import _StreamingProxy


class _FakeStreamingProvider:
    """Fake provider with chat_stream() yielding a known sequence."""

    def __init__(self, deltas: list[str], final_content: str = "") -> None:
        self._deltas = deltas
        self._final_content = final_content or "".join(deltas)

    def name(self) -> str:
        return "fake-streaming"

    def healthy(self) -> bool:
        return True

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return ""

    def complete(self, prompt: str) -> str:
        return ""

    def chat(self, messages, *, tools=None, options=None) -> ChatResponse:
        return ChatResponse(content=self._final_content, tool_calls=(), raw=None)

    def chat_stream(self, messages, *, tools=None, options=None) -> Iterable:
        for d in self._deltas:
            yield TextDelta(text=d)
        yield StreamDone(content=self._final_content)


def test_streaming_proxy_populates_dispatched_invocations_from_audit_log(tmp_path: Path):
    """When MCP audit log gains an entry during the stream, _StreamingProxy.chat()
    must return it on ChatResponse.dispatched_invocations.

    Setup:
    - Write a pre-existing entry to the audit log BEFORE _StreamingProxy.chat() runs.
      It MUST NOT appear in dispatched_invocations (offset must be captured before stream).
    - The fake provider's chat_stream() appends a NEW entry mid-stream.
      That entry MUST appear in dispatched_invocations.
    """
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    audit_log = persona_dir / "tool_invocations.log.jsonl"

    pre_existing = {"name": "search_memories", "arguments": {"query": "x"}, "result_summary": "noop"}
    new_entry = {
        "name": "record_monologue",
        "arguments": {"monologue": "thinking", "feed_digest": "she thought"},
        "result_summary": "captured",
        "monologue_text": "thinking",
    }

    # Pre-existing audit entry — written before the stream starts.
    audit_log.write_text(json.dumps(pre_existing) + "\n")

    class _AppendingProvider(_FakeStreamingProvider):
        def chat_stream(self, messages, *, tools=None, options=None):
            yield TextDelta(text="he")
            # Append the new entry mid-stream (simulates the MCP server writing
            # during tool dispatch inside the claude subprocess).
            with audit_log.open("a") as fh:
                fh.write(json.dumps(new_entry) + "\n")
            yield TextDelta(text="llo")
            yield StreamDone(content="hello")

    loop = asyncio.new_event_loop()

    async def _go() -> ChatResponse:
        chunk_q: asyncio.Queue = asyncio.Queue()
        provider = _AppendingProvider(["he", "llo"])
        proxy = _StreamingProxy(provider, chunk_q, asyncio.get_event_loop())
        return await asyncio.to_thread(
            proxy.chat,
            [ChatMessage(role="user", content="hi")],
            tools=None,
            options={"persona_dir": str(persona_dir)},
        )

    try:
        result = loop.run_until_complete(_go())
    finally:
        loop.close()

    # Critical assertions: the new entry MUST be in dispatched_invocations,
    # the pre-existing entry MUST NOT be.
    di = result.dispatched_invocations
    assert di, "dispatched_invocations was empty — streaming-path audit read regressed"
    names = [inv.get("name") for inv in di]
    assert "record_monologue" in names, f"record_monologue missing from {names}"
    assert "search_memories" not in names, (
        "pre-existing audit entry leaked through — offset capture is wrong"
    )
    # The monologue_text field must come through (this is what
    # _find_monologue_text reads to fire pass 2).
    rec = next(inv for inv in di if inv.get("name") == "record_monologue")
    assert rec.get("monologue_text") == "thinking"


def test_streaming_proxy_returns_empty_dispatched_when_options_missing(tmp_path: Path):
    """No persona_dir in options → no audit read attempted → empty dispatched.

    Defensive baseline: no crash, no exception.
    """
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    loop = asyncio.new_event_loop()

    async def _go() -> ChatResponse:
        chunk_q: asyncio.Queue = asyncio.Queue()
        provider = _FakeStreamingProvider(["hi"])
        proxy = _StreamingProxy(provider, chunk_q, asyncio.get_event_loop())
        return await asyncio.to_thread(
            proxy.chat,
            [ChatMessage(role="user", content="hi")],
            tools=None,
            options=None,  # NO persona_dir
        )

    try:
        result = loop.run_until_complete(_go())
    finally:
        loop.close()

    assert result.dispatched_invocations == ()
