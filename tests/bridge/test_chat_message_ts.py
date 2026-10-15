"""ChatMessage.ts round-trips into the narrative context lines, rendered in
local wall-clock time (issue #217) - msg.ts itself stays raw stored UTC;
only the rendered line converts."""

import re
from datetime import UTC, datetime

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import _claude_context_lines

_TIMED = re.compile(r"^(\d{4}-\d{2}-\d{2}T\S+) (?:User|you): ")


def test_message_without_ts_omits_the_time_clause():
    msg = ChatMessage(role="user", content="hi")
    [line] = list(_claude_context_lines([msg]))
    assert line == "User: hi"


def test_message_with_ts_emits_the_time_clause():
    msg = ChatMessage(role="user", content="hi", ts="2026-05-20T10:00:00Z")
    [line] = list(_claude_context_lines([msg]))
    match = _TIMED.match(line)
    assert match is not None
    rendered = match.group(1)
    # Rendered in local wall-clock time, not the raw UTC string relabeled.
    # Exact offset depends on the CI runner's OS timezone, so assert what's
    # deterministic regardless of it: not the raw 'Z'-suffixed string, and
    # the same underlying instant.
    assert rendered != "2026-05-20T10:00:00Z"
    assert not rendered.endswith("Z")
    assert datetime.fromisoformat(rendered).astimezone(UTC) == datetime(2026, 5, 20, 10, 0, 0, tzinfo=UTC)


def test_mixed_ts_renders_correctly():
    msgs = [
        ChatMessage(role="user", content="a", ts="2026-05-20T10:00:00Z"),
        ChatMessage(role="assistant", content="b"),  # no ts
        ChatMessage(role="user", content="c", ts="2026-05-20T10:05:00Z"),
    ]
    lines = list(_claude_context_lines(msgs))
    first, second, third = _TIMED.match(lines[0]), lines[1], _TIMED.match(lines[2])
    assert first is not None and third is not None
    assert datetime.fromisoformat(first.group(1)).astimezone(UTC) == datetime(2026, 5, 20, 10, 0, 0, tzinfo=UTC)
    assert second == "you: b"
    assert datetime.fromisoformat(third.group(1)).astimezone(UTC) == datetime(2026, 5, 20, 10, 5, 0, tzinfo=UTC)


def test_chat_message_default_ts_is_none():
    msg = ChatMessage(role="user", content="hi")
    assert msg.ts is None
