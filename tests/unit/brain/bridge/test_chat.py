"""Tests for brain.bridge.chat — ChatMessage, ToolCall, ChatResponse."""

from __future__ import annotations

import json

import pytest

from brain.bridge.chat import (
    ChatMessage,
    ChatResponse,
    ImageBlock,
    TextBlock,
    ToolCall,
)

# ---------------------------------------------------------------------------
# ChatMessage construction + to_dict round-trip
# ---------------------------------------------------------------------------


def test_chat_message_system_to_dict() -> None:
    """System message serialises to role + content only."""
    m = ChatMessage(role="system", content="You are Nell.")
    d = m.to_dict()
    assert d == {"role": "system", "content": "You are Nell."}


def test_chat_message_user_to_dict() -> None:
    """User message serialises cleanly without optional fields."""
    m = ChatMessage(role="user", content="Hello Nell.")
    d = m.to_dict()
    assert d == {"role": "user", "content": "Hello Nell."}
    assert "tool_call_id" not in d
    assert "tool_calls" not in d


def test_chat_message_assistant_no_tool_calls_to_dict() -> None:
    """Assistant message without tool_calls omits that field."""
    m = ChatMessage(role="assistant", content="Hi Hana.")
    d = m.to_dict()
    assert d == {"role": "assistant", "content": "Hi Hana."}
    assert "tool_calls" not in d


def test_chat_message_assistant_with_tool_calls_to_dict() -> None:
    """Assistant message with tool_calls includes serialised tool_calls list."""
    tc = ToolCall(id="call_1", name="search", arguments={"query": "dreams"})
    m = ChatMessage(role="assistant", content="", tool_calls=(tc,))
    d = m.to_dict()
    assert d["role"] == "assistant"
    assert d["tool_calls"] == [tc.to_dict()]
    assert "tool_call_id" not in d


def test_chat_message_tool_response_to_dict() -> None:
    """Tool response message includes tool_call_id."""
    m = ChatMessage(role="tool", content="found it", tool_call_id="call_1")
    d = m.to_dict()
    assert d == {"role": "tool", "content": "found it", "tool_call_id": "call_1"}


def test_chat_message_is_frozen() -> None:
    """ChatMessage is immutable — attribute assignment raises FrozenInstanceError."""
    import dataclasses

    m = ChatMessage(role="user", content="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.content = "changed"  # type: ignore[misc]


def test_chat_message_default_tool_calls_is_empty_tuple() -> None:
    """Default tool_calls is an empty tuple, not a mutable list."""
    m = ChatMessage(role="user", content="test")
    assert m.tool_calls == ()
    assert isinstance(m.tool_calls, tuple)


def test_chat_message_default_tool_call_id_is_none() -> None:
    m = ChatMessage(role="user", content="test")
    assert m.tool_call_id is None


# ---------------------------------------------------------------------------
# ToolCall construction + from_provider_dict
# ---------------------------------------------------------------------------


def test_tool_call_construction() -> None:
    """ToolCall stores id, name, and arguments."""
    tc = ToolCall(id="c1", name="get_weather", arguments={"city": "London"})
    assert tc.id == "c1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "London"}


def test_tool_call_from_provider_dict_with_dict_arguments() -> None:
    """from_provider_dict handles pre-parsed dict arguments (Claude shape)."""
    d = {
        "id": "call_abc",
        "function": {
            "name": "recall",
            "arguments": {"topic": "heartbeat", "limit": 5},
        },
    }
    tc = ToolCall.from_provider_dict(d)
    assert tc.id == "call_abc"
    assert tc.name == "recall"
    assert tc.arguments == {"topic": "heartbeat", "limit": 5}


def test_tool_call_from_provider_dict_with_string_arguments() -> None:
    """from_provider_dict handles JSON-encoded string arguments (Ollama shape)."""
    d = {
        "id": "call_xyz",
        "function": {
            "name": "write_memory",
            "arguments": json.dumps({"content": "Hana loves dreams"}),
        },
    }
    tc = ToolCall.from_provider_dict(d)
    assert tc.name == "write_memory"
    assert tc.arguments == {"content": "Hana loves dreams"}


def test_tool_call_from_provider_dict_malformed_raises_value_error() -> None:
    """from_provider_dict raises ValueError on structurally invalid input."""
    with pytest.raises(ValueError, match="missing required fields"):
        ToolCall.from_provider_dict({"no_id": "here"})


def test_tool_call_from_provider_dict_bad_json_string_raises_value_error() -> None:
    """from_provider_dict raises ValueError when arguments string is not valid JSON."""
    d = {
        "id": "c1",
        "function": {"name": "f", "arguments": "{not valid json}"},
    }
    with pytest.raises(ValueError, match="cannot parse arguments JSON"):
        ToolCall.from_provider_dict(d)


def test_tool_call_from_provider_dict_non_dict_arguments_raises_value_error() -> None:
    """from_provider_dict raises ValueError when arguments decodes to non-dict."""
    d = {
        "id": "c1",
        "function": {"name": "f", "arguments": json.dumps([1, 2, 3])},
    }
    with pytest.raises(ValueError, match="must decode to a dict"):
        ToolCall.from_provider_dict(d)


# ---------------------------------------------------------------------------
# ChatResponse construction
# ---------------------------------------------------------------------------


def test_chat_response_construction() -> None:
    """ChatResponse stores content, tool_calls, and raw."""
    tc = ToolCall(id="c1", name="foo", arguments={})
    resp = ChatResponse(content="hello", tool_calls=(tc,), raw={"model": "llama3"})
    assert resp.content == "hello"
    assert resp.tool_calls == (tc,)
    assert resp.raw == {"model": "llama3"}


def test_chat_response_empty_tool_calls_default() -> None:
    """Default tool_calls is an empty tuple."""
    resp = ChatResponse(content="hi")
    assert resp.tool_calls == ()
    assert isinstance(resp.tool_calls, tuple)


def test_chat_response_raw_defaults_to_none() -> None:
    resp = ChatResponse(content="hi")
    assert resp.raw is None


# ---------------------------------------------------------------------------
# ContentBlock / multimodal support
# ---------------------------------------------------------------------------

_VALID_SHA = "a" * 64


def test_chat_message_str_content_backward_compat() -> None:
    """A ChatMessage built with a str continues to serialise as text."""
    msg = ChatMessage(role="user", content="hi")
    assert msg.content_text() == "hi"


def test_chat_message_typed_blocks_text_and_image() -> None:
    """Typed-block messages flatten via content_text()."""
    blocks = (
        TextBlock(text="look at this"),
        ImageBlock(image_sha=_VALID_SHA, media_type="image/png"),
        TextBlock(text="what do you think"),
    )
    msg = ChatMessage(role="user", content=blocks)
    assert msg.content_text() == "look at this\n[image: aaaaaaaa]\nwhat do you think"


def test_chat_message_to_dict_serialises_blocks() -> None:
    """to_dict() emits Anthropic-shaped block list when content is blocks."""
    blocks = (
        TextBlock(text="hi"),
        ImageBlock(image_sha=_VALID_SHA, media_type="image/png"),
    )
    msg = ChatMessage(role="user", content=blocks)
    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == [
        {"type": "text", "text": "hi"},
        {"type": "image", "image_sha": _VALID_SHA, "media_type": "image/png"},
    ]


def test_chat_message_to_dict_keeps_str_path() -> None:
    """Pure-string content stays serialised as a string for legacy callers."""
    msg = ChatMessage(role="user", content="hi")
    d = msg.to_dict()
    assert d["content"] == "hi"


def test_image_block_rejects_non_hex_sha() -> None:
    with pytest.raises(ValueError, match="image_sha"):
        ImageBlock(image_sha="not_hex_at_all_" + "z" * 49, media_type="image/png")


def test_image_block_rejects_short_sha() -> None:
    with pytest.raises(ValueError, match="image_sha"):
        ImageBlock(image_sha="a" * 63, media_type="image/png")


def test_image_block_rejects_uppercase_sha() -> None:
    with pytest.raises(ValueError, match="image_sha"):
        ImageBlock(image_sha="A" * 64, media_type="image/png")


def test_image_block_rejects_unknown_media_type() -> None:
    with pytest.raises(ValueError, match="media_type"):
        ImageBlock(image_sha=_VALID_SHA, media_type="application/pdf")


def test_image_block_accepts_all_allowed_media_types() -> None:
    for mt in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        ImageBlock(image_sha=_VALID_SHA, media_type=mt)


def test_image_block_with_description() -> None:
    """Optional description for cached single-turn descriptions (D5)."""
    block = ImageBlock(
        image_sha=_VALID_SHA,
        media_type="image/png",
        description="A cropped Korn-hoodie selfie.",
    )
    assert block.description == "A cropped Korn-hoodie selfie."


def test_text_block_basic() -> None:
    block = TextBlock(text="hi")
    assert block.text == "hi"
    assert block.type == "text"
