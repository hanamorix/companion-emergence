"""Walker no longer references thinking_log.jsonl (v0.0.26)."""
from __future__ import annotations

import inspect

from brain.health import walker


def test_walker_source_has_no_thinking_log_reference():
    src = inspect.getsource(walker)
    assert "thinking_log" not in src
