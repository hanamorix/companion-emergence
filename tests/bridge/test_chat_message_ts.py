"""ChatMessage.ts round-trips through _claude_context_jsonl_lines."""

import json

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import _claude_context_jsonl_lines


def test_message_without_ts_omits_field():
    msg = ChatMessage(role="user", content="hi")
    [line] = list(_claude_context_jsonl_lines([msg]))
    record = json.loads(line)
    assert "ts" not in record


def test_message_with_ts_emits_field():
    msg = ChatMessage(role="user", content="hi", ts="2026-05-20T10:00:00Z")
    [line] = list(_claude_context_jsonl_lines([msg]))
    record = json.loads(line)
    assert record["ts"] == "2026-05-20T10:00:00Z"


def test_mixed_ts_renders_correctly():
    msgs = [
        ChatMessage(role="user", content="a", ts="2026-05-20T10:00:00Z"),
        ChatMessage(role="assistant", content="b"),  # no ts
        ChatMessage(role="user", content="c", ts="2026-05-20T10:05:00Z"),
    ]
    lines = list(_claude_context_jsonl_lines(msgs))
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["ts"] == "2026-05-20T10:00:00Z"
    assert "ts" not in parsed[1]
    assert parsed[2]["ts"] == "2026-05-20T10:05:00Z"


def test_chat_message_default_ts_is_none():
    msg = ChatMessage(role="user", content="hi")
    assert msg.ts is None
