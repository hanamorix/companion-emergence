"""P0 image-tool-route — posture preservation (C4).

C4: the MCP-only disallowed-builtins posture must be untouched — built-in
``Read`` stays disallowed (the image must arrive via the MCP ImageContent
block, never a re-enabled built-in file tool).

(C6, the removed image/text builder fork, is listed in tests/unit/test_removed_names_stay_removed.py.)
"""

from __future__ import annotations

# ── C4 — MCP-only posture preserved ────────────────────────────────────────


def test_apply_lean_flags_emits_read_in_disallowed_set() -> None:
    from brain.bridge.provider import _BUILTIN_TOOLS_DISALLOWED, _apply_lean_flags

    assert "Read" in _BUILTIN_TOOLS_DISALLOWED
    assert isinstance(_BUILTIN_TOOLS_DISALLOWED, tuple)
    cmd: list[str] = []
    _apply_lean_flags(cmd)
    assert "--disallowedTools" in cmd
    idx = cmd.index("--disallowedTools")
    disallowed = cmd[idx + 1 :]
    assert "Read" in disallowed
    assert "--strict-mcp-config" in cmd
