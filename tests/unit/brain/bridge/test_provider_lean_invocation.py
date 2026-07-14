"""Task A1 — lean CLI invocation.

Guards that every ClaudeCliProvider argv builder:
1. Adds ``--disallowedTools`` covering the built-in tools she has no business
   calling (trims their definition tokens from cache-creation cost). This is
   POLICY, not dead weight — anything off the list is genuinely callable, so
   the list decides what she can reach (#71).
2. Adds ``--strict-mcp-config`` to pin the session to the configured MCP server.

Three builder paths in provider.py carry ``--allowedTools``; all three must also
carry the lean flags via ``_apply_lean_flags``.
"""

from __future__ import annotations

from brain.bridge.provider import _BUILTIN_TOOLS_DISALLOWED


def test_builtin_disallow_list_covers_the_costly_tools():
    for t in ("Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task"):
        assert t in _BUILTIN_TOOLS_DISALLOWED, f"{t!r} missing from _BUILTIN_TOOLS_DISALLOWED"


def test_web_tools_stay_callable_at_chat_time():
    """WebFetch/WebSearch must NOT be disallowed — they are a real capability,
    not dead weight (issue #71).

    History: 87bfc692 swept them in with the dev tools on the rationale that
    the persona "can't call them anyway — she's restricted to
    mcp__brain-tools__*". That was false: --allowedTools is a PERMISSION list,
    not an exclusive one (the CLI's exclusive flag, --tools, is unused here),
    and --dangerously-skip-permissions is on every call — so the built-ins were
    genuinely reachable. Blocking Bash/Edit/Write/Task is deliberate policy;
    blocking web access was collateral. This canary stops the next lean pass
    from sweeping them back in.
    """
    for t in ("WebFetch", "WebSearch"):
        assert t not in _BUILTIN_TOOLS_DISALLOWED, (
            f"{t!r} is disallowed — chat-time web access is gone again (#71)"
        )


def test_apply_lean_flags_adds_disallowed_and_strict():
    from brain.bridge.provider import _BUILTIN_TOOLS_DISALLOWED, _apply_lean_flags
    cmd: list[str] = []
    _apply_lean_flags(cmd)
    assert "--disallowedTools" in cmd
    assert "--strict-mcp-config" in cmd
    for t in _BUILTIN_TOOLS_DISALLOWED:
        assert t in cmd, f"{t!r} not forwarded to cmd by _apply_lean_flags"
