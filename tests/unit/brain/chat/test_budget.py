"""Tests for brain.chat.budget — prompt-size guard."""

from __future__ import annotations

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.chat.budget import apply_budget


class _StubProvider(LLMProvider):
    """Minimal LLMProvider stub. apply_budget only calls generate()."""

    def __init__(self, response: str = "[summary of earlier conversation]") -> None:
        self.response = response
        self.calls: list[str] = []

    def name(self) -> str:
        return "stub"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(prompt)
        return self.response

    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content=self.response, tool_calls=())


class _ExplodingProvider(LLMProvider):
    """Provider whose generate() always raises."""

    def name(self) -> str:
        return "exploding"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("model down")

    def chat(self, messages, *, tools=None, options=None):
        raise RuntimeError("model down")


def test_apply_budget_below_threshold_is_passthrough() -> None:
    msgs = [
        ChatMessage(role="system", content="be Nell"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]
    out = apply_budget(
        msgs, max_tokens=1_000, preserve_tail_msgs=40, provider=_StubProvider()
    )
    assert out == msgs


def test_apply_budget_above_threshold_compresses_head() -> None:
    huge_text = "x" * 8_000  # ~2K tokens estimate (len // 4)
    msgs: list[ChatMessage] = [ChatMessage(role="system", content="be Nell")]
    for i in range(60):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(ChatMessage(role=role, content=huge_text))
    for i in range(40):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(ChatMessage(role=role, content="tail"))

    provider = _StubProvider(response="lots of x")
    out = apply_budget(
        msgs, max_tokens=10_000, preserve_tail_msgs=40, provider=provider
    )

    # Structure: original system + compressed-head system note + 40 tail msgs.
    assert out[0] == msgs[0]
    assert out[1].role == "system"
    # The compressed-head system note should mention either "Earlier" or
    # "earlier" — case-insensitive contains is enough.
    note_text = (
        out[1].content if isinstance(out[1].content, str) else out[1].content_text()
    )
    assert "earlier" in note_text.lower()
    assert list(out[-40:]) == list(msgs[-40:])
    assert len(provider.calls) == 1


def test_apply_budget_compression_failure_falls_back_to_truncation() -> None:
    huge = "y" * 8_000
    msgs: list[ChatMessage] = [ChatMessage(role="system", content="be Nell")]
    for i in range(60):
        msgs.append(
            ChatMessage(role="user" if i % 2 == 0 else "assistant", content=huge)
        )
    for i in range(40):
        msgs.append(
            ChatMessage(role="user" if i % 2 == 0 else "assistant", content="tail")
        )

    out = apply_budget(
        msgs, max_tokens=10_000, preserve_tail_msgs=40, provider=_ExplodingProvider()
    )

    assert out[0] == msgs[0]
    assert out[1].role == "system"
    note_text = (
        out[1].content if isinstance(out[1].content, str) else out[1].content_text()
    )
    assert "truncated" in note_text.lower()
    assert list(out[-40:]) == list(msgs[-40:])


def test_apply_budget_short_session_below_threshold_passes_through() -> None:
    msgs = [
        ChatMessage(role="system", content="be Nell"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hi"),
        ChatMessage(role="user", content="how"),
        ChatMessage(role="assistant", content="good"),
    ]
    out = apply_budget(
        msgs, max_tokens=190_000, preserve_tail_msgs=40, provider=_StubProvider()
    )
    assert out == msgs


def test_apply_budget_with_no_head_to_compress_returns_unchanged() -> None:
    """When the message list is too short to have any head (system +
    preserve_tail_msgs >= total), pass through unchanged even if over budget."""
    # 41 messages (1 system + 40 tail), all huge — would otherwise trigger
    # compression, but there's no head to compress because everything is in
    # the preserved tail.
    huge = "z" * 8_000
    msgs: list[ChatMessage] = [ChatMessage(role="system", content=huge)]
    for i in range(40):
        msgs.append(
            ChatMessage(role="user" if i % 2 == 0 else "assistant", content=huge)
        )

    out = apply_budget(
        msgs, max_tokens=1_000, preserve_tail_msgs=40, provider=_StubProvider()
    )
    assert out == msgs
