"""Tests for provider.chat() implementations — FakeProvider, ClaudeCliProvider, OllamaProvider."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
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
# ClaudeCliProvider.chat — legacy text path (no tools)
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
    with patch("brain.bridge.provider.subprocess.run", return_value=_make_claude_result()) as mock_run:
        p = ClaudeCliProvider(model="sonnet")
        p.chat([ChatMessage(role="user", content="hello")])

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "hello" in cmd


def test_claude_cli_chat_passes_system_prompt_flag() -> None:
    """System message in the list → --system-prompt flag."""
    with patch("brain.bridge.provider.subprocess.run", return_value=_make_claude_result()) as mock_run:
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
    with patch("brain.bridge.provider.subprocess.run", return_value=_make_claude_result("dream response")):
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

    with patch("brain.bridge.provider.subprocess.run", return_value=bad):
        p = ClaudeCliProvider()
        with pytest.raises(ProviderError) as exc_info:
            p.chat([ChatMessage(role="user", content="x")])

    assert exc_info.value.stage == "claude_cli_exit"
    assert "auth failure" in exc_info.value.detail


def test_claude_cli_chat_timeout_raises_provider_error() -> None:
    """Subprocess timeout → ProviderError("claude_cli_timeout", ...)."""
    with patch(
        "brain.bridge.provider.subprocess.run",
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

    with patch("brain.bridge.provider.subprocess.run", return_value=bad):
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


def test_ollama_chat_filters_provider_context_options() -> None:
    """persona_dir is bridge context, not an Ollama generation option."""
    with patch("httpx.post", return_value=_make_ollama_response()) as mock_post:
        p = OllamaProvider()
        p.chat(
            [ChatMessage(role="user", content="x")],
            options={"persona_dir": "/tmp/persona", "temperature": 0.4},
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["options"] == {"temperature": 0.4}


def test_ollama_chat_omits_options_when_only_context_options() -> None:
    with patch("httpx.post", return_value=_make_ollama_response()) as mock_post:
        p = OllamaProvider()
        p.chat(
            [ChatMessage(role="user", content="x")],
            options={"persona_dir": "/tmp/persona"},
        )

    payload = mock_post.call_args[1]["json"]
    assert "options" not in payload


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


# ---------------------------------------------------------------------------
# ClaudeCliProvider.chat — tool-calling via --mcp-config (SP-3 rewrite)
#
# Replaces the old --json-schema tests; the underlying provider was
# rewritten to use --mcp-config (production path per master ref §6 SP-3
# and 2026-04-27 stress-test finding).
# ---------------------------------------------------------------------------


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "persona"
    d.mkdir()
    return d


def _fake_proc(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_chat_with_tools_passes_mcp_config_flag(persona_dir: Path) -> None:
    """The new path must call claude with --mcp-config <tmp_path>."""
    provider = ClaudeCliProvider()
    captured: dict = {}

    def _capture(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["mcp_config_path"] = cmd[cmd.index("--mcp-config") + 1]
        return _fake_proc(json.dumps({"result": "hello back"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_capture):
        response = provider.chat(
            [
                ChatMessage(role="system", content="you are nell"),
                ChatMessage(role="user", content="hi"),
            ],
            tools=[{"name": "search_memories", "description": "search"}],
            options={"persona_dir": str(persona_dir)},
        )

    assert response.content == "hello back"
    assert response.tool_calls == ()
    # --mcp-config flag is present and points at a real json file (now unlinked,
    # but we captured the path during the call)
    assert "--mcp-config" in captured["cmd"]
    assert captured["mcp_config_path"].endswith(".json")


def test_chat_with_tools_writes_correct_mcp_config(persona_dir: Path) -> None:
    """The temp mcp.json must contain the right command + args."""
    provider = ClaudeCliProvider()
    captured: dict = {}

    def _capture(cmd, **kwargs):
        path = cmd[cmd.index("--mcp-config") + 1]
        # Read the temp file BEFORE the provider's finally block deletes it
        captured["config"] = json.loads(Path(path).read_text())
        return _fake_proc(json.dumps({"result": "ok"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_capture):
        provider.chat(
            [
                ChatMessage(role="system", content="sys"),
                ChatMessage(role="user", content="hi"),
            ],
            tools=[{"name": "x", "description": "x"}],
            options={"persona_dir": str(persona_dir)},
        )

    cfg = captured["config"]
    assert "brain-tools" in cfg["mcpServers"]
    server_cfg = cfg["mcpServers"]["brain-tools"]
    assert server_cfg["command"] == sys.executable
    assert server_cfg["args"][0] == "-m"
    assert server_cfg["args"][1] == "brain.mcp_server"
    assert "--persona-dir" in server_cfg["args"]
    assert str(persona_dir) in server_cfg["args"]
    assert server_cfg["env"]["NELL_MCP_AUDIT_REQUEST_ID"]


def test_read_audit_lines_since_filters_other_request_ids(tmp_path: Path) -> None:
    from brain.bridge.provider import _read_audit_lines_since

    audit_path = tmp_path / "tool_invocations.log.jsonl"
    audit_path.write_text(
        "\n".join(
            [
                json.dumps({"name": "mine", "arguments": {}, "result_summary": "ok", "request_id": "req-1"}),
                json.dumps({"name": "other", "arguments": {}, "result_summary": "ok", "request_id": "req-2"}),
                json.dumps({"name": "legacy", "arguments": {}, "result_summary": "ok"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = _read_audit_lines_since(audit_path, 0, request_id="req-1")

    assert [record["name"] for record in records] == ["mine", "legacy"]


def test_chat_with_tools_keeps_existing_flags(persona_dir: Path) -> None:
    """The other flags (-p, --output-format, --model, --system-prompt)
    must remain — only --json-schema is replaced."""
    provider = ClaudeCliProvider()
    captured: dict = {}

    def _capture(cmd, **kwargs):
        captured["cmd"] = cmd
        return _fake_proc(json.dumps({"result": "ok"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_capture):
        provider.chat(
            [
                ChatMessage(role="system", content="sys-prompt"),
                ChatMessage(role="user", content="hi"),
            ],
            tools=[{"name": "x"}],
            options={"persona_dir": str(persona_dir)},
        )

    cmd = captured["cmd"]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--model" in cmd
    assert "--system-prompt" in cmd
    assert "sys-prompt" in cmd
    # The replaced flag must NOT appear
    assert "--json-schema" not in cmd


def test_chat_with_tools_passes_allowed_tools_for_each_brain_tool(persona_dir: Path) -> None:
    """The cmd must include --allowedTools with every brain-tool enumerated.

    Claude CLI's `-p` (non-interactive) mode blocks MCP tool calls unless
    each is explicitly pre-approved. Without this flag, the MCP server starts
    and tools are advertised but Claude refuses to call them — the same
    mechanism gap the live-exercise stress test surfaced.
    """
    from brain.tools import NELL_TOOL_NAMES

    provider = ClaudeCliProvider()
    captured: dict = {}

    def _capture(cmd, **kwargs):
        captured["cmd"] = cmd
        return _fake_proc(json.dumps({"result": "ok"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_capture):
        provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"name": "x"}],
            options={"persona_dir": str(persona_dir)},
        )

    cmd = captured["cmd"]
    assert "--allowedTools" in cmd
    # Every NELL_TOOL_NAME must appear under the mcp__brain-tools__ namespace.
    for name in NELL_TOOL_NAMES:
        assert f"mcp__brain-tools__{name}" in cmd, (
            f"missing --allowedTools entry for {name}"
        )


def test_chat_with_tools_parses_payload_result(persona_dir: Path) -> None:
    provider = ClaudeCliProvider()

    with patch(
        "brain.bridge.provider.subprocess.run",
        return_value=_fake_proc(json.dumps({"result": "the actual reply"})),
    ):
        response = provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"name": "x"}],
            options={"persona_dir": str(persona_dir)},
        )

    assert response.content == "the actual reply"
    assert response.tool_calls == ()


def test_chat_with_tools_missing_result_key_raises_parse(persona_dir: Path) -> None:
    provider = ClaudeCliProvider()

    with patch(
        "brain.bridge.provider.subprocess.run",
        return_value=_fake_proc(json.dumps({"different_key": "x"})),
    ):
        with pytest.raises(ProviderError) as ei:
            provider.chat(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "x"}],
                options={"persona_dir": str(persona_dir)},
            )
    assert ei.value.stage == "claude_cli_parse"


def test_chat_with_tools_nonzero_exit_raises_exit(persona_dir: Path) -> None:
    provider = ClaudeCliProvider()

    with patch(
        "brain.bridge.provider.subprocess.run",
        return_value=_fake_proc("", returncode=2, stderr="boom"),
    ):
        with pytest.raises(ProviderError) as ei:
            provider.chat(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "x"}],
                options={"persona_dir": str(persona_dir)},
            )
    assert ei.value.stage == "claude_cli_exit"


def test_chat_with_tools_missing_persona_dir_option_raises(tmp_path: Path) -> None:
    """tools= without options['persona_dir'] is a programmer bug — fail fast."""
    provider = ClaudeCliProvider()
    with pytest.raises(ProviderError) as ei:
        provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"name": "x"}],
            options={},  # missing persona_dir
        )
    assert ei.value.stage == "mcp_unavailable"


def test_chat_with_tools_cleans_up_temp_file(persona_dir: Path) -> None:
    """The temp mcp.json file must be unlinked after the call returns."""
    provider = ClaudeCliProvider()
    captured_path: list[str] = []

    def _capture(cmd, **kwargs):
        path = cmd[cmd.index("--mcp-config") + 1]
        captured_path.append(path)
        return _fake_proc(json.dumps({"result": "ok"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_capture):
        provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"name": "x"}],
            options={"persona_dir": str(persona_dir)},
        )

    assert len(captured_path) == 1
    assert not Path(captured_path[0]).exists()


def test_chat_with_tools_cleans_up_temp_file_on_subprocess_error(persona_dir: Path) -> None:
    """Temp file must be unlinked even when subprocess.run raises TimeoutExpired."""
    provider = ClaudeCliProvider()
    captured_path: list[str] = []

    def _capture(cmd, **kwargs):
        path = cmd[cmd.index("--mcp-config") + 1]
        captured_path.append(path)
        raise subprocess.TimeoutExpired(cmd="claude", timeout=300)

    with patch("brain.bridge.provider.subprocess.run", side_effect=_capture):
        with pytest.raises(ProviderError) as ei:
            provider.chat(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "x"}],
                options={"persona_dir": str(persona_dir)},
            )

    assert ei.value.stage == "claude_cli_timeout"
    assert len(captured_path) == 1
    assert not Path(captured_path[0]).exists()


def test_chat_with_tools_missing_mcp_sdk_raises_mcp_unavailable(persona_dir: Path) -> None:
    """If the mcp SDK is not installed, raise ProviderError('mcp_unavailable')."""
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "mcp":
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    provider = ClaudeCliProvider()
    with patch("builtins.__import__", side_effect=_fake_import):
        with pytest.raises(ProviderError) as ei:
            provider.chat(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "x"}],
                options={"persona_dir": str(persona_dir)},
            )
    assert ei.value.stage == "mcp_unavailable"
    assert "pip install" in ei.value.detail


def test_chat_with_tools_temp_file_write_failure_raises_setup(persona_dir: Path) -> None:
    """OSError on the temp file write must surface as ProviderError('claude_cli_setup')."""
    provider = ClaudeCliProvider()
    with patch("brain.bridge.provider.tempfile.NamedTemporaryFile", side_effect=OSError("disk full")):
        with pytest.raises(ProviderError) as ei:
            provider.chat(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "x"}],
                options={"persona_dir": str(persona_dir)},
            )
    assert ei.value.stage == "claude_cli_setup"
    assert "disk full" in ei.value.detail


def test_chat_without_tools_unchanged(persona_dir: Path) -> None:
    """When tools is None, the provider falls through the legacy text path —
    no --mcp-config, no persona_dir requirement."""
    provider = ClaudeCliProvider()

    with patch(
        "brain.bridge.provider.subprocess.run",
        return_value=_fake_proc(json.dumps({"result": "plain reply"})),
    ) as mock_run:
        response = provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=None,
        )

    assert response.content == "plain reply"
    cmd = mock_run.call_args.args[0]
    assert "--mcp-config" not in cmd
