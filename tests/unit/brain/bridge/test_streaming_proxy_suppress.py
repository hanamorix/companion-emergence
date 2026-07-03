"""_StreamingProxy suppress_stream + emit_text (bug #4 double-stream guard).

On a recruitable first pass run_tool_loop sets suppress_stream so the proxy
buffers the reply instead of pushing it to the WS queue; if no reach happens it
flushes the buffered reply once via emit_text. This keeps exactly one reply on
the WS (no pass-1 + rerun concatenation).
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterable

from brain.bridge.chat import ChatMessage, ChatResponse, StreamDone, TextDelta
from brain.bridge.server import _StreamingProxy


class _FakeStreamingProvider:
    def name(self) -> str:
        return "fake"

    def healthy(self) -> bool:
        return True

    def generate(self, prompt, *, system=None) -> str:
        return ""

    def complete(self, prompt) -> str:
        return ""

    def chat(self, messages, *, tools=None, options=None) -> ChatResponse:
        return ChatResponse(content="hello there", tool_calls=(), raw=None)

    def chat_stream(self, messages, *, tools=None, options=None) -> Iterable:
        yield TextDelta(text="hello ")
        yield TextDelta(text="there")
        yield StreamDone(content="hello there")


def _drain(q: asyncio.Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_suppress_stream_buffers_without_pushing_to_queue():
    loop = asyncio.new_event_loop()

    async def _go():
        q: asyncio.Queue = asyncio.Queue()
        proxy = _StreamingProxy(_FakeStreamingProvider(), q, asyncio.get_event_loop())
        resp = await asyncio.to_thread(
            proxy.chat,
            [ChatMessage(role="user", content="hi")],
            tools=None,
            options={"suppress_stream": True},
        )
        return resp, _drain(q)

    try:
        resp, pushed = loop.run_until_complete(_go())
    finally:
        loop.close()
    assert resp.content == "hello there", "content must still be buffered/returned"
    assert pushed == [], f"suppress_stream must push nothing to the WS, got {pushed}"


def test_emit_text_pushes_word_chunks():
    loop = asyncio.new_event_loop()

    async def _go():
        q: asyncio.Queue = asyncio.Queue()
        proxy = _StreamingProxy(_FakeStreamingProvider(), q, asyncio.get_event_loop())
        await asyncio.to_thread(proxy.emit_text, "flush me")
        return _drain(q)

    try:
        pushed = loop.run_until_complete(_go())
    finally:
        loop.close()
    assert "".join(pushed) == "flush me", f"emit_text must word-chunk the text, got {pushed}"
