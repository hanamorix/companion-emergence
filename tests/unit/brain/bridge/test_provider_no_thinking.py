"""Provider no longer emits --thinking/--budget-tokens flags, no thinking_blocks field (v0.0.26)."""
from __future__ import annotations

import inspect

from brain.bridge import provider


def test_provider_source_has_no_thinking_flags():
    src = inspect.getsource(provider)
    assert "--thinking" not in src
    assert "--budget-tokens" not in src
    assert "_write_thinking_log" not in src


def test_chatresponse_has_no_thinking_blocks_field():
    from brain.bridge.chat import ChatResponse
    resp = ChatResponse(content="x", tool_calls=(), raw=None)
    assert not hasattr(resp, "thinking_blocks")
