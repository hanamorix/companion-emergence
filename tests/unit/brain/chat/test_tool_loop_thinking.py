"""run_tool_loop passes thinking_budget_tokens to provider when config has it set."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from unittest.mock import MagicMock

from brain.bridge.provider import ChatResponse


def _make_messages():
    sys = MagicMock()
    sys.role = "system"
    sys.content_text = lambda: "you are nell"
    user = MagicMock()
    user.role = "user"
    user.content_text = lambda: "hello"
    return [sys, user]


def test_run_tool_loop_passes_thinking_budget(tmp_path: Path):
    from brain.persona_config import PersonaConfig
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.store import MemoryStore
    from brain.memory.hebbian import HebbianMatrix

    # Write config with thinking budget
    cfg = PersonaConfig.load(tmp_path / "persona_config.json")
    replace(cfg, thinking_budget_tokens=5000).save(tmp_path / "persona_config.json")

    captured_options: list = []

    provider = MagicMock()
    provider.chat.side_effect = lambda msgs, tools=None, options=None: (
        captured_options.append(options) or ChatResponse(content="ok", tool_calls=(), raw=None)
    )

    store = MagicMock(spec=MemoryStore)
    hebbian = MagicMock(spec=HebbianMatrix)

    run_tool_loop(
        _make_messages(),
        provider=provider,
        tools=None,
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )

    assert captured_options, "provider.chat was not called"
    opts = captured_options[0]
    assert opts.get("thinking_budget_tokens") == 5000
    assert opts.get("thinking_call_site") == "chat"


def test_run_tool_loop_no_thinking_when_budget_none(tmp_path: Path):
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.store import MemoryStore
    from brain.memory.hebbian import HebbianMatrix

    captured_options: list = []

    provider = MagicMock()
    provider.chat.side_effect = lambda msgs, tools=None, options=None: (
        captured_options.append(options) or ChatResponse(content="ok", tool_calls=(), raw=None)
    )

    store = MagicMock(spec=MemoryStore)
    hebbian = MagicMock(spec=HebbianMatrix)

    run_tool_loop(
        _make_messages(),
        provider=provider,
        tools=None,
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )

    opts = captured_options[0] if captured_options else {}
    assert "thinking_budget_tokens" not in opts
