"""Initiate compose no longer reads thinking_budget_tokens (v0.0.26)."""
from __future__ import annotations

import inspect

from brain.initiate import compose


def test_compose_source_has_no_thinking_reference():
    src = inspect.getsource(compose)
    assert "thinking_budget_tokens" not in src
