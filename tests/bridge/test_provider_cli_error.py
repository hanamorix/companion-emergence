"""CLI error frames must not be delivered as the companion's reply (#92).

The Claude CLI can exit 0 and report a failure inside the result frame —
``{"is_error": true, "result": "Not logged in · Please run /login"}``.
The non-zero-exit guard never sees those, so before this fix the error string
became ``ChatResponse.content`` and the user read a system error as something
their companion said.

The over-budget frame is the one deliberate exception: it carries partial work
worth keeping and is handled separately (see test_provider_budget.py).

Tests added one at a time per the tdd-guard rule.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import ClaudeCliProvider, ProviderError

_AUTH_FRAME = {
    "type": "result",
    "is_error": True,
    "result": "Not logged in · Please run /login",
}


def test_mcp_tools_path_raises_on_cli_error_frame(tmp_path, monkeypatch):
    """_chat_with_mcp_tools must raise, not hand the error text back as a reply."""

    def fake_run(cmd, *a, **k):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_AUTH_FRAME), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(__import__("sys").modules, "mcp", __import__("types").ModuleType("mcp"))

    persona = tmp_path / "persona"
    persona.mkdir()
    p = ClaudeCliProvider(model="claude-sonnet-4-6")

    with pytest.raises(ProviderError) as exc:
        p.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"name": "noop"}],
            options={"persona_dir": str(persona)},
        )
    # The raw CLI text belongs in the log, not in her mouth — but it must be
    # in the error detail so the failure is diagnosable.
    assert "Not logged in" in str(exc.value)


def test_chat_stream_yields_error_not_done_on_cli_error_frame(monkeypatch):
    """chat_stream: an exit-0 error frame must yield StreamError, not StreamDone.

    Same defect as the non-streaming paths, one layer out — the WS path would
    otherwise stream the CLI's error text to the user as her reply.
    """
    from unittest.mock import MagicMock, patch

    from brain.bridge.chat import StreamDone, StreamError

    proc = MagicMock()
    proc.stdout = iter([json.dumps({"type": "result", **_AUTH_FRAME}) + "\n"])
    proc.stdin = MagicMock()
    proc.poll.return_value = 0
    proc.wait.return_value = 0
    proc.returncode = 0
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""

    provider = ClaudeCliProvider(model="claude-sonnet-4-6")
    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))

    errors = [e for e in events if isinstance(e, StreamError)]
    dones = [e for e in events if isinstance(e, StreamDone)]
    assert errors, f"expected a StreamError, got {events!r}"
    assert errors[0].stage == "claude_cli_error"
    assert "Not logged in" in errors[0].detail
    assert not [d for d in dones if "Not logged in" in (d.content or "")], (
        "the CLI error text must never be delivered as her reply"
    )
