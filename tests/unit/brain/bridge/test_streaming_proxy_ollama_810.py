"""An Ollama turn over the /stream WebSocket goes through _StreamingProxy.chat (#810).

OllamaProvider.chat_stream takes no ``tools`` and yields plain ``str`` chunks, not
the TextDelta/StreamDone events the proxy collects. Calling it from the proxy
raised ``TypeError`` before any request, and even without that the reply came
back empty. The proxy must send Ollama turns through the blocking ``chat()`` —
the only Ollama path that carries tools out and tool calls back — and word-chunk
the reply onto the WS queue.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import OllamaProvider
from brain.bridge.server import _StreamingProxy

_TOOLS = [
    {
        "type": "function",
        "function": {"name": "search_memories", "description": "d", "parameters": {"type": "object"}},
    }
]


def _drain(q: asyncio.Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _run_proxy(body: dict, *, options: dict | None = None):
    captured: list[dict] = []

    def _fake_post(url, *, json=None, timeout=None):  # noqa: A002 - mirrors httpx.post's keyword
        captured.append(json)
        return httpx.Response(200, text=_dumps(body), request=httpx.Request("POST", url))

    def _no_stream(*_a, **_k):
        raise AssertionError("the proxy must not use Ollama's text-only chat_stream for a tool turn")

    loop = asyncio.new_event_loop()

    async def _go():
        q: asyncio.Queue = asyncio.Queue()
        proxy = _StreamingProxy(OllamaProvider(), q, asyncio.get_event_loop())
        with patch("httpx.post", side_effect=_fake_post), patch("httpx.stream", side_effect=_no_stream):
            resp = await asyncio.to_thread(
                proxy.chat,
                [ChatMessage(role="user", content="hi")],
                tools=_TOOLS,
                options=options,
            )
        await asyncio.sleep(0)  # run the put_nowait callbacks the worker thread queued
        return resp, _drain(q)

    try:
        resp, pushed = loop.run_until_complete(_go())
    finally:
        loop.close()
    return resp, pushed, captured


def _dumps(obj: dict) -> str:
    return json.dumps(obj)


def test_ollama_turn_reaches_ollama_with_tools_and_streams_the_reply():
    resp, pushed, captured = _run_proxy(
        {"message": {"role": "assistant", "content": "hello there, love"}, "done": True}
    )

    assert len(captured) == 1, "exactly one request must reach Ollama"
    assert captured[0]["tools"] == _TOOLS, "the turn's tools must reach Ollama"
    assert resp.content == "hello there, love"
    assert "".join(pushed) == "hello there, love", f"reply must be queued as chunks, got {pushed}"
    assert len(pushed) > 1, "the reply should be word-chunked, not one blob"


def test_ollama_tool_call_comes_back_through_the_proxy():
    resp, pushed, captured = _run_proxy(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "search_memories", "arguments": {"query": "tea"}}}],
            },
            "done": True,
        }
    )

    assert captured[0]["tools"] == _TOOLS
    assert [tc.name for tc in resp.tool_calls] == ["search_memories"]
    assert resp.tool_calls[0].arguments == {"query": "tea"}
    assert pushed == []
