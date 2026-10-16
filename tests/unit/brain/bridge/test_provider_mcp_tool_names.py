"""#416: on the MCP path the CLI registers brain tools as ``mcp__brain-tools__<name>``, but the
prompt text names them bare (``record_monologue``). Models then call the bare name and the CLI
answers "No such tool available" (Haiku 5.5: 373 bare record_monologue calls vs 47 MCP-named
calls in the real login's October CLI logs). The provider rewrites backticked brain-tool names to
the names the model can actually call, in both the system prompt file and stdin.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import ClaudeCliProvider


def _recording_run():
    captured: dict = {}

    def _side_effect(cmd, **kwargs):
        captured["stdin"] = kwargs.get("input")
        if "--system-prompt-file" in cmd:
            captured["system"] = Path(cmd[cmd.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
        m = MagicMock()
        m.returncode, m.stdout, m.stderr = 0, json.dumps({"result": "ok"}), ""
        return m

    _side_effect.captured = captured  # type: ignore[attr-defined]
    return _side_effect


SYSTEM = (
    "- `record_monologue` — Record what's running.\n"
    "**First**, call `record_monologue(monologue, feed_digest)`.\n"
    "Plain prose about record_monologue stays as written; `WebSearch` is a built-in."
)


def test_mcp_path_rewrites_backticked_brain_tool_names(tmp_path: Path) -> None:
    rec = _recording_run()
    with patch("brain.bridge.provider.subprocess.run", side_effect=rec):
        ClaudeCliProvider(model="sonnet").chat(
            [ChatMessage(role="system", content=SYSTEM), ChatMessage(role="user", content="hi")],
            tools=[{"name": "record_monologue"}],
            options={"persona_dir": str(tmp_path), "volatile_suffix": "If you called `record_monologue` this turn"},
        )
    system, stdin = rec.captured["system"], rec.captured["stdin"]
    assert "- `mcp__brain-tools__record_monologue` —" in system
    assert "call `mcp__brain-tools__record_monologue(monologue, feed_digest)`" in system
    assert "`record_monologue" not in system
    assert "Plain prose about record_monologue stays" in system
    assert "`WebSearch`" in system
    assert "If you called `mcp__brain-tools__record_monologue` this turn" in stdin


class _StdinSpy:
    def __init__(self) -> None:
        self.text = ""

    def write(self, s: str) -> None:
        self.text += s

    def close(self) -> None:
        pass


def test_chat_stream_mcp_path_rewrites_names(tmp_path: Path) -> None:
    """The live WS path (_StreamingProxy prefers chat_stream) gets the same rewrite."""
    import io

    seen: dict = {}

    def _popen(cmd, **kwargs):
        seen["system"] = Path(cmd[cmd.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
        proc = MagicMock()
        proc.stdin = seen["stdin"] = _StdinSpy()
        proc.stdout = iter([json.dumps({"type": "result", "is_error": False, "result": "hello"}) + "\n"])
        proc.stderr = io.StringIO("")
        proc.wait.return_value = proc.poll.return_value = 0
        return proc

    with patch("subprocess.Popen", side_effect=_popen):
        list(
            ClaudeCliProvider(model="sonnet").chat_stream(
                [ChatMessage(role="system", content=SYSTEM), ChatMessage(role="user", content="hi")],
                tools=[{"name": "record_monologue"}],
                options={"persona_dir": str(tmp_path), "volatile_suffix": "If you called `record_monologue` this turn"},
            )
        )
    assert "call `mcp__brain-tools__record_monologue(monologue, feed_digest)`" in seen["system"]
    assert "`record_monologue" not in seen["system"]
    assert "If you called `mcp__brain-tools__record_monologue` this turn" in seen["stdin"].text
