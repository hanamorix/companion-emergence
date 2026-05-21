"""_format_claude_context_block adds a 'Current time' preamble line."""

import re

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import _format_claude_context_block


def test_preamble_includes_current_time():
    msgs = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="b"),
    ]
    block = _format_claude_context_block(msgs, includes_latest_user=True)
    # ISO-8601 UTC pattern e.g. 2026-05-20T14:30:00Z
    assert re.search(r"Current time: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", block) is not None


def test_preamble_explains_ts_field():
    msgs = [
        ChatMessage(role="user", content="a", ts="2026-05-20T10:00:00Z"),
        ChatMessage(role="assistant", content="b", ts="2026-05-20T10:05:00Z"),
    ]
    block = _format_claude_context_block(msgs, includes_latest_user=True)
    assert "ts" in block  # the explanatory line about the field
    assert "wall-clock" in block.lower()
