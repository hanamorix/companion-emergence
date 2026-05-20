"""Verify --dangerously-skip-permissions lands on every claude argv."""

from unittest.mock import MagicMock, patch

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import ClaudeCliProvider


def _run_returns_ok(stdout='{"result": "hi"}', exit=0):
    res = MagicMock()
    res.stdout = stdout
    res.stderr = ""
    res.returncode = exit
    return res


def test_generate_passes_dangerously_skip_permissions():
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=5)
    with patch("brain.bridge.provider.subprocess.run", return_value=_run_returns_ok()) as run:
        provider.generate("hello", system="be brief")
    argv = run.call_args.args[0]
    assert "--dangerously-skip-permissions" in argv, f"missing in argv: {argv}"


def test_chat_text_path_passes_dangerously_skip_permissions():
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=5)
    with patch("brain.bridge.provider.subprocess.run", return_value=_run_returns_ok()) as run:
        provider.chat([ChatMessage(role="user", content="hello")])
    argv = run.call_args.args[0]
    assert "--dangerously-skip-permissions" in argv, f"missing in argv: {argv}"
