"""Verify --dangerously-skip-permissions lands on every claude argv.

Three argv builders in provider.py spawn the claude CLI; each must include
the flag exactly once so a future refactor can't silently drop it from
one of them. The three sites are (the old ``_chat_with_images`` stream-json
path was removed when the image transport was dissolved — P0 image-tool-route):

1. ``generate``                 — single-shot ``-p`` text prompt
2. ``chat`` (legacy text path)  — multi-turn ``-p`` text prompt
3. ``_chat_with_mcp_tools``     — tool-calling ``--mcp-config`` path
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import ClaudeCliProvider


def _run_returns_ok(stdout='{"result": "hi"}', exit_code=0):
    res = MagicMock()
    res.stdout = stdout
    res.stderr = ""
    res.returncode = exit_code
    return res


def test_generate_passes_dangerously_skip_permissions():
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=5)
    with patch("brain.bridge.provider.subprocess.run", return_value=_run_returns_ok()) as run:
        provider.generate("hello", system="be brief")
    argv = run.call_args.args[0]
    assert "--dangerously-skip-permissions" in argv, f"missing in argv: {argv}"
    assert argv.count("--dangerously-skip-permissions") == 1, (
        f"flag should appear exactly once, got {argv.count('--dangerously-skip-permissions')}: {argv}"
    )


def test_chat_text_path_passes_dangerously_skip_permissions():
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=5)
    with patch("brain.bridge.provider.subprocess.run", return_value=_run_returns_ok()) as run:
        provider.chat([ChatMessage(role="user", content="hello")])
    argv = run.call_args.args[0]
    assert "--dangerously-skip-permissions" in argv, f"missing in argv: {argv}"
    assert argv.count("--dangerously-skip-permissions") == 1, (
        f"flag should appear exactly once, got {argv.count('--dangerously-skip-permissions')}: {argv}"
    )


def test_chat_with_mcp_tools_passes_dangerously_skip_permissions(tmp_path: Path):
    """MCP tool-calling path must carry the flag too.

    Reproduces the mcp invocation by passing ``tools=[...]`` + persona_dir
    — that routes through ``_chat_with_mcp_tools`` which builds its own
    argv at provider.py line ~729.
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    with patch("brain.bridge.provider.subprocess.run", return_value=_run_returns_ok()) as run:
        provider = ClaudeCliProvider(model="sonnet", timeout_seconds=5)
        provider.chat(
            [
                ChatMessage(role="system", content="you are nell"),
                ChatMessage(role="user", content="hi"),
            ],
            tools=[{"name": "search_memories", "description": "search"}],
            options={"persona_dir": str(persona_dir)},
        )

    argv = run.call_args.args[0]
    # Sanity: confirm we actually hit the MCP-tools builder
    assert "--mcp-config" in argv, f"expected mcp-config path, got: {argv}"
    assert "--allowedTools" in argv, f"expected allowedTools on mcp path, got: {argv}"
    assert "--dangerously-skip-permissions" in argv, f"missing in argv: {argv}"
    assert argv.count("--dangerously-skip-permissions") == 1, (
        f"flag should appear exactly once, got {argv.count('--dangerously-skip-permissions')}: {argv}"
    )
