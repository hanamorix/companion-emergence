"""ChatMessage.ts round-trips through _claude_context_jsonl_lines, rendered
in local wall-clock time (issue #217) - msg.ts itself stays raw stored UTC;
only the JSONL rendering converts."""

import json
from datetime import UTC, datetime

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
    # Rendered in local wall-clock time, not the raw UTC string relabeled.
    # Exact offset depends on the CI runner's OS timezone, so assert what's
    # deterministic regardless of it: not the raw 'Z'-suffixed string, and
    # the same underlying instant.
    assert record["ts"] != "2026-05-20T10:00:00Z"
    assert not record["ts"].endswith("Z")
    assert datetime.fromisoformat(record["ts"]).astimezone(UTC) == datetime(
        2026, 5, 20, 10, 0, 0, tzinfo=UTC
    )


def test_mixed_ts_renders_correctly():
    msgs = [
        ChatMessage(role="user", content="a", ts="2026-05-20T10:00:00Z"),
        ChatMessage(role="assistant", content="b"),  # no ts
        ChatMessage(role="user", content="c", ts="2026-05-20T10:05:00Z"),
    ]
    lines = list(_claude_context_jsonl_lines(msgs))
    parsed = [json.loads(line) for line in lines]
    assert datetime.fromisoformat(parsed[0]["ts"]).astimezone(UTC) == datetime(
        2026, 5, 20, 10, 0, 0, tzinfo=UTC
    )
    assert "ts" not in parsed[1]
    assert datetime.fromisoformat(parsed[2]["ts"]).astimezone(UTC) == datetime(
        2026, 5, 20, 10, 5, 0, tzinfo=UTC
    )


def test_chat_message_default_ts_is_none():
    msg = ChatMessage(role="user", content="hi")
    assert msg.ts is None
