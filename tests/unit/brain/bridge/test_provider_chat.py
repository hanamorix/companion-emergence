"""Tests for provider.chat() implementations — FakeProvider, ClaudeCliProvider, OllamaProvider."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import (
    ClaudeCliProvider,
    FakeProvider,
    OllamaProvider,
    ProviderError,
    get_provider,
)

# ---------------------------------------------------------------------------
# ProviderError
# ---------------------------------------------------------------------------


def test_provider_error_carries_stage_and_detail() -> None:
    """ProviderError stores stage + detail as attributes."""
    err = ProviderError("ollama_http", "503: service unavailable")
    assert err.stage == "ollama_http"
    assert err.detail == "503: service unavailable"
    assert "[ollama_http]" in str(err)
    assert "503: service unavailable" in str(err)


def test_provider_error_is_runtime_error() -> None:
    """ProviderError is a RuntimeError subclass."""
    assert issubclass(ProviderError, RuntimeError)


# ---------------------------------------------------------------------------
# FakeProvider.chat
# ---------------------------------------------------------------------------


def test_fake_provider_chat_returns_chat_response() -> None:
    """FakeProvider.chat returns a ChatResponse instance."""
    p = FakeProvider()
    resp = p.chat([ChatMessage(role="user", content="hi")])
    assert isinstance(resp, ChatResponse)


def test_fake_provider_chat_is_deterministic() -> None:
    """Same messages → same content every call."""
    p = FakeProvider()
    msgs = [ChatMessage(role="user", content="tell me a dream")]
    a = p.chat(msgs)
    b = p.chat(msgs)
    assert a.content == b.content


def test_fake_provider_chat_content_has_fake_prefix() -> None:
    p = FakeProvider()
    resp = p.chat([ChatMessage(role="user", content="hello")])
    assert resp.content.startswith("FAKE_CHAT:")


def test_fake_provider_chat_tool_calls_always_empty() -> None:
    """FakeProvider never synthesises tool calls."""
    p = FakeProvider()
    resp = p.chat([ChatMessage(role="user", content="use tools")])
    assert resp.tool_calls == ()


def test_fake_provider_chat_with_tools_param_still_empty_tool_calls() -> None:
    """Passing a tools list doesn't make FakeProvider return tool_calls."""
    p = FakeProvider()
    tools = [{"name": "search", "description": "search the web"}]
    resp = p.chat([ChatMessage(role="user", content="search for dreams")], tools=tools)
    assert resp.tool_calls == ()


def test_fake_provider_chat_different_messages_differ() -> None:
    """Different message content → different response content."""
    p = FakeProvider()
    r1 = p.chat([ChatMessage(role="user", content="foo")])
    r2 = p.chat([ChatMessage(role="user", content="bar")])
    assert r1.content != r2.content


# ---------------------------------------------------------------------------
# ClaudeCliProvider.chat
# ---------------------------------------------------------------------------


def _make_claude_result(content: str = "hello", rc: int = 0) -> MagicMock:
    """Build a subprocess.CompletedProcess mock."""
    m = MagicMock()
    m.returncode = rc
    m.stdout = json.dumps({"result": content})
    m.stderr = ""
    return m


def test_claude_cli_chat_calls_subprocess_with_p_flag() -> None:
    """ClaudeCliProvider.chat uses -p for the flattened conversation."""
    with patch("subprocess.run", return_value=_make_claude_result()) as mock_run:
        p = ClaudeCliProvider(model="sonnet")
        p.chat([ChatMessage(role="user", content="hello")])

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "hello" in cmd


def test_claude_cli_chat_passes_system_prompt_flag() -> None:
    """System message in the list → --system-prompt flag."""
    with patch("subprocess.run", return_value=_make_claude_result()) as mock_run:
        p = ClaudeCliProvider()
        p.chat(
            [
                ChatMessage(role="system", content="you are nell"),
                ChatMessage(role="user", content="hi"),
            ]
        )

    cmd = mock_run.call_args[0][0]
    assert "--system-prompt" in cmd
    assert "you are nell" in cmd


def test_claude_cli_chat_parses_success() -> None:
    """Successful call returns ChatResponse with correct content."""
    with patch("subprocess.run", return_value=_make_claude_result("dream response")):
        p = ClaudeCliProvider()
        resp = p.chat([ChatMessage(role="user", content="dream")])

    assert isinstance(resp, ChatResponse)
    assert resp.content == "dream response"
    assert resp.tool_calls == ()


def test_claude_cli_chat_nonzero_exit_raises_provider_error() -> None:
    """Non-zero exit code → ProviderError("claude_cli_exit", ...)."""
    bad = MagicMock()
    bad.returncode = 1
    bad.stdout = ""
    bad.stderr = "auth failure"

    with patch("subprocess.run", return_value=bad):
        p = ClaudeCliProvider()
        with pytest.raises(ProviderError) as exc_info:
            p.chat([ChatMessage(role="user", content="x")])

    assert exc_info.value.stage == "claude_cli_exit"
    assert "auth failure" in exc_info.value.detail


def test_claude_cli_chat_timeout_raises_provider_error() -> None:
    """Subprocess timeout → ProviderError("claude_cli_timeout", ...)."""
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300),
    ):
        p = ClaudeCliProvider()
        with pytest.raises(ProviderError) as exc_info:
            p.chat([ChatMessage(role="user", content="x")])

    assert exc_info.value.stage == "claude_cli_timeout"


def test_claude_cli_chat_bad_json_raises_provider_error() -> None:
    """Non-JSON stdout → ProviderError("claude_cli_parse", ...)."""
    bad = MagicMock()
    bad.returncode = 0
    bad.stdout = "not json at all"
    bad.stderr = ""

    with patch("subprocess.run", return_value=bad):
        p = ClaudeCliProvider()
        with pytest.raises(ProviderError) as exc_info:
            p.chat([ChatMessage(role="user", content="x")])

    assert exc_info.value.stage == "claude_cli_parse"


# ---------------------------------------------------------------------------
# OllamaProvider.chat
# ---------------------------------------------------------------------------


def _make_ollama_response(content: str = "hello", tool_calls: list | None = None) -> MagicMock:
    """Build a mock httpx.Response for Ollama /api/chat."""
    payload = {
        "model": "test-model",
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
    }
    if tool_calls is not None:
        payload["message"]["tool_calls"] = tool_calls

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status = MagicMock()  # no-op
    return mock_resp


def test_ollama_chat_builds_correct_payload() -> None:
    """OllamaProvider.chat POSTs with model, messages, stream=False."""
    with patch("httpx.post", return_value=_make_ollama_response()) as mock_post:
        p = OllamaProvider(model="llama3", host="http://localhost:11434")
        p.chat([ChatMessage(role="user", content="hello")])

    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "llama3"
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_ollama_chat_includes_tools_when_provided() -> None:
    """tools= list is forwarded in the payload."""
    tools = [{"name": "search", "type": "function"}]
    with patch("httpx.post", return_value=_make_ollama_response()) as mock_post:
        p = OllamaProvider()
        p.chat([ChatMessage(role="user", content="x")], tools=tools)

    payload = mock_post.call_args[1]["json"]
    assert payload["tools"] == tools


def test_ollama_chat_parses_content() -> None:
    """Content from message.content lands in ChatResponse.content."""
    with patch("httpx.post", return_value=_make_ollama_response("Nell's reply")):
        p = OllamaProvider()
        resp = p.chat([ChatMessage(role="user", content="x")])

    assert resp.content == "Nell's reply"


def test_ollama_chat_parses_tool_calls() -> None:
    """tool_calls in the response are parsed into ToolCall instances."""
    raw_tc = [{"id": "c1", "function": {"name": "recall", "arguments": {"topic": "hana"}}}]
    with patch("httpx.post", return_value=_make_ollama_response(tool_calls=raw_tc)):
        p = OllamaProvider()
        resp = p.chat([ChatMessage(role="user", content="x")])

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "recall"
    assert resp.tool_calls[0].arguments == {"topic": "hana"}


def test_ollama_chat_http_error_raises_provider_error() -> None:
    """HTTP error response → ProviderError("ollama_http", ...)."""
    import httpx

    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "service unavailable"

    with patch(
        "httpx.post",
        side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=mock_resp),
    ):
        p = OllamaProvider()
        with pytest.raises(ProviderError) as exc_info:
            p.chat([ChatMessage(role="user", content="x")])

    assert exc_info.value.stage == "ollama_http"


def test_ollama_chat_request_error_raises_provider_error() -> None:
    """Network error → ProviderError("ollama_request", ...)."""
    import httpx

    with patch("httpx.post", side_effect=httpx.RequestError("connection refused")):
        p = OllamaProvider()
        with pytest.raises(ProviderError) as exc_info:
            p.chat([ChatMessage(role="user", content="x")])

    assert exc_info.value.stage == "ollama_request"


def test_ollama_chat_invalid_json_raises_provider_error() -> None:
    """Non-JSON response body → ProviderError("ollama_parse", ...)."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.side_effect = ValueError("no json")

    with patch("httpx.post", return_value=mock_resp):
        p = OllamaProvider()
        with pytest.raises(ProviderError) as exc_info:
            p.chat([ChatMessage(role="user", content="x")])

    assert exc_info.value.stage == "ollama_parse"


# ---------------------------------------------------------------------------
# OllamaProvider.healthy
# ---------------------------------------------------------------------------


def test_ollama_healthy_true_on_200() -> None:
    """healthy() returns True when /api/tags responds 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("httpx.get", return_value=mock_resp):
        assert OllamaProvider().healthy() is True


def test_ollama_healthy_false_on_request_error() -> None:
    """healthy() returns False when Ollama is unreachable."""
    import httpx

    with patch("httpx.get", side_effect=httpx.RequestError("refused")):
        assert OllamaProvider().healthy() is False


# ---------------------------------------------------------------------------
# OllamaProvider.generate — calls chat() under the hood
# ---------------------------------------------------------------------------


def test_ollama_generate_delegates_to_chat() -> None:
    """generate() returns the text content from chat()."""
    with patch("httpx.post", return_value=_make_ollama_response("dream text")):
        p = OllamaProvider()
        result = p.generate("dream prompt", system="you are nell")

    assert result == "dream text"


def test_ollama_generate_includes_system_message_when_provided() -> None:
    """generate() with system= sends a system ChatMessage first."""
    with patch("httpx.post", return_value=_make_ollama_response()) as mock_post:
        p = OllamaProvider()
        p.generate("hello", system="be nell")

    payload = mock_post.call_args[1]["json"]
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "be nell"
    assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# get_provider factory — ollama no longer raises
# ---------------------------------------------------------------------------


def test_get_provider_ollama_returns_instance() -> None:
    """get_provider("ollama") returns an OllamaProvider instance (no NotImplementedError)."""
    provider = get_provider("ollama")
    assert isinstance(provider, OllamaProvider)


def test_get_provider_ollama_name_includes_model() -> None:
    """The returned OllamaProvider has the expected default model name."""
    provider = get_provider("ollama")
    assert provider.name().startswith("ollama:")
