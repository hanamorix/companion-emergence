"""Tests for brain.bridge.provider — LLM provider abstraction."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from brain.bridge.provider import (
    ClaudeCliProvider,
    FakeProvider,
    LLMProvider,
    OllamaProvider,
    get_provider,
)


def test_llm_provider_is_abstract() -> None:
    """LLMProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_fake_provider_is_deterministic() -> None:
    """Same (prompt, system) produces the same output every call."""
    p = FakeProvider()
    a = p.generate("hello", system="be helpful")
    b = p.generate("hello", system="be helpful")
    assert a == b


def test_fake_provider_different_prompts_differ() -> None:
    """Different prompts → different outputs."""
    p = FakeProvider()
    assert p.generate("a") != p.generate("b")


def test_fake_provider_name() -> None:
    """FakeProvider.name() returns 'fake'."""
    assert FakeProvider().name() == "fake"


def test_fake_provider_output_has_dream_prefix() -> None:
    """Fake output starts with 'DREAM:' so downstream dream engine logic works."""
    assert FakeProvider().generate("anything").startswith("DREAM:")


def test_ollama_provider_raises_not_implemented() -> None:
    """OllamaProvider.generate raises NotImplementedError with a clear message."""
    p = OllamaProvider()
    with pytest.raises(NotImplementedError, match="stub"):
        p.generate("anything")


def test_ollama_provider_name_includes_model() -> None:
    """OllamaProvider.name() includes the model identifier."""
    assert OllamaProvider(model="nell-dpo").name() == "ollama:nell-dpo"


def test_claude_cli_provider_builds_expected_command() -> None:
    """ClaudeCliProvider spawns `claude -p <prompt> --output-format json`."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"result": "DREAM: test output"})
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        p = ClaudeCliProvider(model="sonnet")
        out = p.generate("test prompt", system="you are helpful")

    assert out == "DREAM: test output"
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "test prompt" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--model" in cmd
    assert "sonnet" in cmd


def test_claude_cli_provider_forwards_system_prompt() -> None:
    """ClaudeCliProvider passes the system prompt via --system-prompt."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"result": "ok"})

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        ClaudeCliProvider().generate("p", system="you are nell")

    cmd = mock_run.call_args[0][0]
    assert "--system-prompt" in cmd
    assert "you are nell" in cmd


def test_claude_cli_provider_raises_on_nonzero_exit() -> None:
    """Non-zero exit code surfaces a RuntimeError that includes stderr."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "auth failed"

    with patch("subprocess.run", return_value=mock_result):
        p = ClaudeCliProvider()
        with pytest.raises(RuntimeError, match="auth failed"):
            p.generate("p")


def test_claude_cli_provider_name() -> None:
    """Name includes the model identifier."""
    assert ClaudeCliProvider(model="sonnet").name() == "claude-cli:sonnet"


def test_claude_cli_provider_subprocess_timeout_surfaced() -> None:
    """subprocess.TimeoutExpired surfaces as TimeoutError with context."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300)):
        p = ClaudeCliProvider()
        with pytest.raises(TimeoutError, match="timed out"):
            p.generate("p")


def test_get_provider_resolves_known_names() -> None:
    """get_provider returns the right class for each known name."""
    assert isinstance(get_provider("fake"), FakeProvider)
    assert isinstance(get_provider("claude-cli"), ClaudeCliProvider)
    assert isinstance(get_provider("ollama"), OllamaProvider)


def test_get_provider_unknown_name_raises() -> None:
    """Unknown provider name raises ValueError with a clear message."""
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonsense")


def test_claude_cli_provider_malformed_json_surfaces_runtime_error() -> None:
    """Non-JSON stdout from the CLI surfaces as RuntimeError with context.

    Guards the CLI contract drift case — if claude's output format changes
    or a progress prefix appears, dream engine gets a meaningful error
    instead of json.JSONDecodeError leaking up.
    """
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "progress prefix not json"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        p = ClaudeCliProvider()
        with pytest.raises(RuntimeError, match="unexpected output format"):
            p.generate("p")


def test_claude_cli_provider_missing_result_key_surfaces_runtime_error() -> None:
    """Valid JSON missing the 'result' key → RuntimeError, not raw KeyError."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"wrong": "shape"})
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        p = ClaudeCliProvider()
        with pytest.raises(RuntimeError, match="unexpected output format"):
            p.generate("p")
