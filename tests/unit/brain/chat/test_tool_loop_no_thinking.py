"""tool_loop no longer reads thinking_budget_tokens from PersonaConfig (v0.0.26)."""
from __future__ import annotations

import inspect

from brain.chat import tool_loop


def test_tool_loop_source_has_no_thinking_budget_tokens():
    src = inspect.getsource(tool_loop)
    assert "thinking_budget_tokens" not in src
