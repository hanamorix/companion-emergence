"""Verify --dangerously-skip-permissions lands on every claude argv.

Four argv builders in provider.py spawn the claude CLI; each must include
the flag exactly once so a future refactor can't silently drop it from
one of them. The four sites are:

1. ``generate``                 — single-shot ``-p`` text prompt
2. ``chat`` (legacy text path)  — multi-turn ``-p`` text prompt
3. ``_chat_with_images``        — multimodal ``stream-json`` path
4. ``_chat_with_mcp_tools``     — tool-calling ``--mcp-config`` path
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain.bridge.chat import ChatMessage, ImageBlock, TextBlock
from brain.bridge.provider import ClaudeCliProvider


def _run_returns_ok(stdout='{"result": "hi"}', exit_code=0):
    res = MagicMock()
    res.stdout = stdout
    res.stderr = ""
    res.returncode = exit_code
    return res


def _make_stream_json_result(text: str = "ok"):
    """Build a stream-json subprocess result with one assistant + one result frame."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            ),
            json.dumps({"type": "result", "subtype": "success", "result": text}),
        ]
    )
    m.stderr = ""
    return m


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


def test_chat_image_passthrough_passes_dangerously_skip_permissions(tmp_path: Path):
    """Stream-json multimodal path must carry the flag too.

    Reproduces the image-passthrough invocation by saving a tiny png to
    persona_dir and sending a user message with an ImageBlock — that
    routes through ``_chat_with_images`` which builds its own argv at
    provider.py line ~527.
    """
    from brain.images import save_image_bytes

    tiny_png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63606060600000000400015e36b8c80000000049454e44ae426082"
    )
    save_image_bytes(tmp_path, tiny_png, "image/png")
    sha = hashlib.sha256(tiny_png).hexdigest()
    blocks = (
        TextBlock(text="describe this"),
        ImageBlock(image_sha=sha, media_type="image/png"),
    )

    with patch(
        "brain.bridge.provider.subprocess.run",
        return_value=_make_stream_json_result("a tiny png"),
    ) as run:
        provider = ClaudeCliProvider(model="sonnet", timeout_seconds=5)
        provider.chat(
            [ChatMessage(role="user", content=blocks)],
            options={"persona_dir": str(tmp_path)},
        )

    argv = run.call_args.args[0]
    # Sanity: confirm we actually hit the stream-json builder, not legacy -p
    assert "stream-json" in argv, f"expected stream-json path, got: {argv}"
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

    with patch(
        "brain.bridge.provider.subprocess.run", return_value=_run_returns_ok()
    ) as run:
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
