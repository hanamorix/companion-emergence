"""Regression: _StreamingProxy must queue StreamDone.content when no TextDeltas arrived.

Bug surfaced in v0.0.15-alpha.2 (reported on v0.0.16): chat bubbles render
empty during live streaming when the underlying provider's chat_stream()
yields only StreamDone (no per-token TextDelta events). This happens when
Claude CLI's --output-format stream-json returns the reply in a single
result frame — common for extended-thinking mode, fast single-block
responses, or the EOF fallback path in ClaudeCliProvider.chat_stream
(brain/bridge/provider.py:731-733).

Before the fix, _StreamingProxy captured ev.content for the ChatResponse
return value but never put it on chunk_q. The WS handler then sent zero
reply_chunk frames; the frontend bubble stayed empty (text=""), only the
timestamp showed. Reopening NellFace pulled the persisted text via the
history endpoint, masking the live-arrival regression.
"""

from __future__ import annotations

import asyncio
from typing import Any

from brain.bridge.chat import ChatMessage, StreamDone
from brain.bridge.server import _StreamingProxy


class _FakeDoneOnlyProvider:
    """Provider whose chat_stream yields only StreamDone — no TextDeltas.

    Mimics the Claude CLI path where text arrives only in the `result`
    frame or via the assistant-snapshot EOF fallback.
    """

    def name(self) -> str:
        return "fake-done-only"

    def healthy(self) -> bool:
        return True

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ):
        yield StreamDone(content="hello world", metadata={})


async def _drive_proxy(provider: Any) -> tuple[Any, list[str]]:
    """Mirror the production WS handler: drive proxy.chat() in a thread,
    drain reply_chunk frames from the queue, finish on the None sentinel
    (which the production handler emits in its finally block).
    """
    q: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    proxy = _StreamingProxy(provider, q, loop)

    async def _runner() -> Any:
        try:
            return await asyncio.to_thread(
                proxy.chat,
                [ChatMessage(role="user", content="hi")],
            )
        finally:
            # Same sentinel pattern as brain/bridge/server.py:1807.
            q.put_nowait(None)

    task = asyncio.create_task(_runner())

    chunks: list[str] = []
    while True:
        item = await q.get()
        if item is None:
            break
        chunks.append(item)

    resp = await task
    return resp, chunks


def test_streaming_proxy_queues_content_when_no_text_deltas() -> None:
    """A chat_stream that yields only StreamDone must still put content on chunk_q."""
    resp, chunks = asyncio.run(_drive_proxy(_FakeDoneOnlyProvider()))

    assert chunks == ["hello world"], (
        f"expected ['hello world'] but got {chunks!r}. "
        "When chat_stream yields only StreamDone (no preceding TextDeltas), "
        "_StreamingProxy must also queue ev.content so the WS sends at least "
        "one reply_chunk frame — otherwise the chat bubble renders empty."
    )
    # The ChatResponse return value still gets the same content, unchanged.
    assert resp.content == "hello world"


class _FakeProgressiveProvider:
    """Provider that yields TextDeltas then a StreamDone — the normal path."""

    def name(self) -> str:
        return "fake-progressive"

    def healthy(self) -> bool:
        return True

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ):
        from brain.bridge.chat import TextDelta

        for word in ["hello", " ", "world"]:
            yield TextDelta(text=word)
        # StreamDone with content matching the deltas — should NOT
        # double-queue, because chunks is already populated.
        yield StreamDone(content="hello world", metadata={})


def test_streaming_proxy_does_not_double_queue_when_deltas_arrived() -> None:
    """Progressive-streaming path: TextDeltas queued individually, StreamDone is a no-op for the queue."""
    resp, chunks = asyncio.run(_drive_proxy(_FakeProgressiveProvider()))

    assert chunks == ["hello", " ", "world"], (
        f"expected per-token chunks ['hello', ' ', 'world'] but got {chunks!r}. "
        "When TextDeltas arrived during streaming, StreamDone.content must "
        "NOT be queued a second time — that would render the reply twice."
    )
    assert resp.content == "hello world"
