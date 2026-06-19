"""Per-reply --max-budget-usd catastrophe backstop (Task 6 / A).

Spike-verified frame: an over-budget run returns (exit 0) a result JSON with
``"subtype": "error_max_budget_usd"``, ``"is_error": true``,
``"errors": ["Reached maximum budget ($X)"]``.

Tests added one at a time per the tdd-guard rule.
"""

from __future__ import annotations

import json
import subprocess

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import _MAX_TURN_BUDGET_USD, ClaudeCliProvider


def test_mcp_tools_path_budget_flag_and_graceful(tmp_path, monkeypatch):
    """_chat_with_mcp_tools: flag present + over-budget frame handled gracefully."""

    captured: dict = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        frame = {
            "type": "result",
            "subtype": "error_max_budget_usd",
            "is_error": True,
            "result": "partial work",
            "errors": ["Reached maximum budget ($0.75)"],
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(frame), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Stub out the mcp import check
    monkeypatch.setitem(__import__("sys").modules, "mcp", __import__("types").ModuleType("mcp"))

    persona = tmp_path / "persona"
    persona.mkdir()
    p = ClaudeCliProvider(model="claude-sonnet-4-6")
    resp = p.chat(
        [ChatMessage(role="user", content="hi")],
        tools=[{"name": "noop"}],
        options={"persona_dir": str(persona)},
    )
    assert "--max-budget-usd" in captured["cmd"]
    # Partial text is preserved
    assert "partial work" in resp.content
    assert "limit" in resp.content.lower() or "continue" in resp.content.lower()


def test_chat_with_images_path_budget_flag_and_graceful(tmp_path, monkeypatch):
    """_chat_with_images: flag present + over-budget frame handled gracefully."""
    import hashlib

    from brain.bridge.chat import ImageBlock

    captured: dict = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        # stream-json format: result frame
        frame = {
            "type": "result",
            "subtype": "error_max_budget_usd",
            "is_error": True,
            "result": "",
            "errors": ["Reached maximum budget ($0.75)"],
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(frame) + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(__import__("sys").modules, "mcp", __import__("types").ModuleType("mcp"))

    # Build a fake image so _chat_with_images can be reached
    persona = tmp_path / "persona"
    images_dir = persona / "images"
    images_dir.mkdir(parents=True)
    img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8  # minimal fake PNG
    sha = hashlib.sha256(img_bytes).hexdigest()
    (images_dir / f"{sha}.png").write_bytes(img_bytes)

    from brain.bridge.chat import TextBlock

    msg = ChatMessage(
        role="user",
        content=(
            TextBlock(text="describe this"),
            ImageBlock(image_sha=sha, media_type="image/png"),
        ),
    )
    p = ClaudeCliProvider(model="claude-sonnet-4-6")
    resp = p.chat(
        [msg],
        tools=None,
        options={"persona_dir": str(persona)},
    )
    assert "--max-budget-usd" in captured["cmd"]
    assert resp.content and (
        "limit" in resp.content.lower() or "continue" in resp.content.lower()
    )


def test_streaming_path_budget_flag_and_graceful_over_budget(monkeypatch):
    """chat_stream: --max-budget-usd present; over-budget result frame yields graceful StreamDone."""
    import json
    from unittest.mock import MagicMock, patch

    from brain.bridge.chat import StreamDone
    from brain.bridge.provider import ClaudeCliProvider

    provider = ClaudeCliProvider(model="claude-sonnet-4-6", timeout_seconds=60)

    # Two partial text deltas then an over-budget result frame.
    partial_lines = [
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "I started writing"},
            },
        }) + "\n",
        json.dumps({
            "type": "result",
            "subtype": "error_max_budget_usd",
            "is_error": True,
            "result": "",
            "errors": ["Reached maximum budget ($0.75)"],
        }) + "\n",
    ]

    captured_cmd: list = []

    def fake_popen(cmd, *a, **k):
        captured_cmd.extend(cmd)
        proc = MagicMock()
        proc.stdout = iter(partial_lines)
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.close = MagicMock()
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        proc.returncode = 0
        proc.terminate = MagicMock()
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = ""
        return proc

    with patch("brain.bridge.provider.subprocess.Popen", side_effect=fake_popen):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))

    assert "--max-budget-usd" in captured_cmd

    dones = [e for e in events if isinstance(e, StreamDone)]
    assert len(dones) == 1
    done = dones[0]
    # Must contain the cut-off note
    assert "limit" in done.content.lower() or "continue" in done.content.lower()
    # Must not be empty
    assert done.content.strip()


def test_generate_path_does_not_get_budget_flag(monkeypatch):
    """generate() (background) must NOT carry --max-budget-usd."""
    captured: dict = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        payload = {"result": "ok"}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    p = ClaudeCliProvider(model="claude-sonnet-4-6")
    p.generate("hello world")
    assert "--max-budget-usd" not in captured["cmd"]


def test_budget_value_is_model_aware():
    """Sonnet ceiling < Opus ceiling; both are well-defined."""
    assert _MAX_TURN_BUDGET_USD("claude-sonnet-4-6") < _MAX_TURN_BUDGET_USD("claude-opus-4-8")
    assert _MAX_TURN_BUDGET_USD("claude-haiku-4-5") < _MAX_TURN_BUDGET_USD("claude-sonnet-4-6")
    # Short aliases also resolve
    assert _MAX_TURN_BUDGET_USD("sonnet") == _MAX_TURN_BUDGET_USD("claude-sonnet-4-6")
    assert _MAX_TURN_BUDGET_USD("unknown-model-xyz") == 1.50  # fallback


def test_budget_flag_present_and_error_frame_is_graceful(monkeypatch):
    """chat() includes --max-budget-usd and handles the over-budget frame gracefully."""
    captured: dict = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        frame = {
            "type": "result",
            "subtype": "error_max_budget_usd",
            "is_error": True,
            "result": "",
            "errors": ["Reached maximum budget ($1.0)"],
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(frame), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    p = ClaudeCliProvider(model="claude-sonnet-4-6")
    resp = p.chat([ChatMessage(role="user", content="hi")])
    assert "--max-budget-usd" in captured["cmd"]
    # graceful, non-empty, signals the cut-off — not a crash, not empty
    assert resp.content and (
        "limit" in resp.content.lower() or "continue" in resp.content.lower()
    )
