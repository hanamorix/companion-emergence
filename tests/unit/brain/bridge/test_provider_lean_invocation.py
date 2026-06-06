"""Task A1 — lean CLI invocation.

Guards that every ClaudeCliProvider argv builder:
1. Adds ``--disallowedTools`` covering the costly built-in tools (removes ~14K
   cache-creation tokens / call that were dead weight — Nell can't call them).
2. Adds ``--strict-mcp-config`` to pin the session to the configured MCP server.

Three builder paths in provider.py carry ``--allowedTools``; all three must also
carry the lean flags via ``_apply_lean_flags``.
"""

from __future__ import annotations

from brain.bridge.provider import _BUILTIN_TOOLS_DISALLOWED


def test_builtin_disallow_list_covers_the_costly_tools():
    for t in ("Bash", "Read", "Edit", "Write", "Glob", "Grep", "WebFetch", "WebSearch", "Task"):
        assert t in _BUILTIN_TOOLS_DISALLOWED, f"{t!r} missing from _BUILTIN_TOOLS_DISALLOWED"


def test_apply_lean_flags_adds_disallowed_and_strict():
    from brain.bridge.provider import _BUILTIN_TOOLS_DISALLOWED, _apply_lean_flags
    cmd: list[str] = []
    _apply_lean_flags(cmd)
    assert "--disallowedTools" in cmd
    assert "--strict-mcp-config" in cmd
    for t in _BUILTIN_TOOLS_DISALLOWED:
        assert t in cmd, f"{t!r} not forwarded to cmd by _apply_lean_flags"
