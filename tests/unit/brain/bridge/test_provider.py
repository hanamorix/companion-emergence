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


def test_ollama_provider_generate_raises_provider_error_when_unreachable() -> None:
    """OllamaProvider.generate raises ProviderError when Ollama is not running.

    The stub NotImplementedError is gone — OllamaProvider is fully implemented.
    On CI (no local Ollama) the request fails with a network error, surfaced as
    ProviderError("ollama_request", ...).
    """
    from unittest.mock import patch

    import httpx

    from brain.bridge.provider import ProviderError

    with patch("httpx.post", side_effect=httpx.RequestError("connection refused")):
        p = OllamaProvider()
        with pytest.raises(ProviderError) as exc_info:
            p.generate("anything")
    assert exc_info.value.stage == "ollama_request"


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


def test_claude_cli_provider_nonzero_exit_includes_json_stdout_error() -> None:
    """Claude quota errors can arrive as JSON stdout with empty stderr."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = json.dumps({
        "is_error": True,
        "api_error_status": 429,
        "result": "You're out of extra usage · resets 1:50pm",
    })
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        p = ClaudeCliProvider()
        with pytest.raises(RuntimeError, match="api_error_status=429") as exc_info:
            p.generate("p")
    assert "out of extra usage" in str(exc_info.value)


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


def test_get_provider_ollama_returns_instance() -> None:
    """get_provider("ollama") now returns a real OllamaProvider — no longer a stub.

    SP-1 ships OllamaProvider with a full httpx-based implementation, so the
    factory must hand back a usable instance.  The test that previously asserted
    NotImplementedError is updated to assert the new behaviour.
    """
    provider = get_provider("ollama")
    assert isinstance(provider, OllamaProvider)


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


# ---- Tool telemetry: dispatched_invocations from MCP audit log ----


def test_chat_mcp_path_surfaces_dispatched_invocations(tmp_path):
    """ClaudeCliProvider's MCP path reads the audit log diff and surfaces
    tool calls as ChatResponse.dispatched_invocations. This closes the
    telemetry gap from the 2026-04-27 / 2026-05-05 stress tests, where
    tools fired correctly inside the claude subprocess but the bridge
    response showed `tool_invocations=[]`."""
    from brain.bridge.chat import ChatMessage
    from brain.bridge.provider import ClaudeCliProvider

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    audit = persona_dir / "tool_invocations.log.jsonl"
    # Pre-existing line — must NOT show up in dispatched_invocations.
    audit.write_text(
        '{"timestamp": "2026-05-05T18:00:00Z", "name": "previous_call", '
        '"audit_level": "redacted", "arguments": {}, "result_summary": "old", "error": null}\n'
    )

    fresh_lines = (
        '{"timestamp": "2026-05-05T20:00:00Z", "name": "search_memories", '
        '"audit_level": "redacted", "arguments": {"query": "foo"}, '
        '"result_summary": "{\\"count\\": 0}", "error": null}\n'
        '{"timestamp": "2026-05-05T20:00:01Z", "name": "get_soul", '
        '"audit_level": "redacted", "arguments": {}, '
        '"result_summary": "{\\"count\\": 38}", "error": null}\n'
    )

    def fake_run(cmd, *a, **kw):
        # Simulate claude subprocess writing 2 audit lines during its run.
        with audit.open("a") as fh:
            fh.write(fresh_lines)
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"result": "the assistant reply"})
        result.stderr = ""
        return result

    with patch("subprocess.run", side_effect=fake_run):
        p = ClaudeCliProvider()
        resp = p.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "search_memories"}}],
            options={"persona_dir": str(persona_dir)},
        )

    assert resp.content == "the assistant reply"
    assert len(resp.dispatched_invocations) == 2
    names = [inv["name"] for inv in resp.dispatched_invocations]
    assert names == ["search_memories", "get_soul"]
    # The pre-existing audit line must NOT have leaked in
    assert "previous_call" not in names
    # Argument shape preserved
    assert resp.dispatched_invocations[0]["arguments"] == {"query": "foo"}
    # tool_calls remains empty (not the OllamaProvider path)
    assert resp.tool_calls == ()


def test_chat_mcp_path_handles_missing_audit_log(tmp_path):
    """If the audit log doesn't exist (e.g. mcp_audit_log_level=off),
    dispatched_invocations is empty — no crash."""
    from brain.bridge.chat import ChatMessage
    from brain.bridge.provider import ClaudeCliProvider

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    # No audit log file written.

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"result": "reply"})
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        p = ClaudeCliProvider()
        resp = p.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "search_memories"}}],
            options={"persona_dir": str(persona_dir)},
        )

    assert resp.dispatched_invocations == ()


def test_chat_mcp_path_skips_malformed_audit_lines(tmp_path):
    """Malformed JSON in the audit log doesn't break telemetry."""
    from brain.bridge.chat import ChatMessage
    from brain.bridge.provider import ClaudeCliProvider

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    audit = persona_dir / "tool_invocations.log.jsonl"
    audit.write_text("")

    def fake_run(cmd, *a, **kw):
        # 1 malformed + 1 good line
        with audit.open("a") as fh:
            fh.write('not json at all\n')
            fh.write(
                '{"timestamp": "2026-05-05T20:00:00Z", "name": "get_soul", '
                '"audit_level": "redacted", "arguments": {}, '
                '"result_summary": "ok", "error": null}\n'
            )
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"result": "reply"})
        result.stderr = ""
        return result

    with patch("subprocess.run", side_effect=fake_run):
        p = ClaudeCliProvider()
        resp = p.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "search_memories"}}],
            options={"persona_dir": str(persona_dir)},
        )

    assert len(resp.dispatched_invocations) == 1
    assert resp.dispatched_invocations[0]["name"] == "get_soul"


# ---------------------------------------------------------------------------
# OllamaProvider.chat_stream — token streaming
# ---------------------------------------------------------------------------


def _make_streaming_response(lines, status_code=200):
    """Build a context-manager mock that mimics httpx.stream's response."""
    import httpx as _httpx

    response = MagicMock()
    response.status_code = status_code
    response.iter_lines = MagicMock(return_value=iter(lines))
    response.read = MagicMock(return_value=b"")

    def _raise():
        if status_code >= 400:
            raise _httpx.HTTPStatusError(
                "boom", request=MagicMock(), response=response
            )

    response.raise_for_status = MagicMock(side_effect=_raise)

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=response)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_ollama_chat_stream_yields_content_chunks() -> None:
    """Each Ollama frame's message.content is yielded in order; done halts."""
    from brain.bridge.chat import ChatMessage

    lines = [
        json.dumps({"message": {"role": "assistant", "content": "hello "}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "there, "}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "hana"}, "done": False}),
        json.dumps({"message": {}, "done": True}),
    ]
    with patch("httpx.stream", return_value=_make_streaming_response(lines)):
        p = OllamaProvider()
        chunks = list(p.chat_stream([ChatMessage(role="user", content="hi")]))

    assert chunks == ["hello ", "there, ", "hana"]
    assert "".join(chunks) == "hello there, hana"


def test_ollama_chat_stream_skips_empty_chunks() -> None:
    """Frames with no content (e.g. role-only) are skipped, not yielded as ''."""
    from brain.bridge.chat import ChatMessage

    lines = [
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": False}),
        json.dumps({"done": True}),
    ]
    with patch("httpx.stream", return_value=_make_streaming_response(lines)):
        p = OllamaProvider()
        chunks = list(p.chat_stream([ChatMessage(role="user", content="hi")]))

    assert chunks == ["ok"]


def test_ollama_chat_stream_stops_at_done() -> None:
    """A frame with done=True halts iteration even if more lines follow."""
    from brain.bridge.chat import ChatMessage

    lines = [
        json.dumps({"message": {"role": "assistant", "content": "first"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "."}, "done": True}),
        # This SHOULDN'T be yielded — done already fired.
        json.dumps({"message": {"role": "assistant", "content": "ghost"}, "done": False}),
    ]
    with patch("httpx.stream", return_value=_make_streaming_response(lines)):
        p = OllamaProvider()
        chunks = list(p.chat_stream([ChatMessage(role="user", content="hi")]))

    assert chunks == ["first", "."]


def test_ollama_chat_stream_request_error_raises_provider_error() -> None:
    import httpx as _httpx

    from brain.bridge.chat import ChatMessage

    with patch("httpx.stream", side_effect=_httpx.RequestError("connection refused")):
        p = OllamaProvider()
        with pytest.raises(Exception, match="ollama_request"):
            list(p.chat_stream([ChatMessage(role="user", content="hi")]))


def test_ollama_chat_stream_http_error_raises_provider_error() -> None:
    from brain.bridge.chat import ChatMessage

    lines: list[str] = []
    with patch("httpx.stream", return_value=_make_streaming_response(lines, status_code=500)):
        p = OllamaProvider()
        with pytest.raises(Exception, match="ollama_http"):
            list(p.chat_stream([ChatMessage(role="user", content="hi")]))


def test_ollama_chat_stream_invalid_json_line_raises_provider_error() -> None:
    from brain.bridge.chat import ChatMessage

    lines = ["this is not json"]
    with patch("httpx.stream", return_value=_make_streaming_response(lines)):
        p = OllamaProvider()
        with pytest.raises(Exception, match="ollama_parse"):
            list(p.chat_stream([ChatMessage(role="user", content="hi")]))


def test_ollama_chat_stream_includes_options_in_payload() -> None:
    """Generation options (temperature, etc.) forward; persona_dir is stripped."""
    from brain.bridge.chat import ChatMessage

    captured: dict = {}

    def _stream_capture(method, url, **kwargs):
        captured["json"] = kwargs.get("json")
        return _make_streaming_response([json.dumps({"done": True})])

    with patch("httpx.stream", side_effect=_stream_capture):
        p = OllamaProvider()
        list(
            p.chat_stream(
                [ChatMessage(role="user", content="hi")],
                options={"temperature": 0.7, "persona_dir": "/should/be/stripped"},
            )
        )

    assert captured["json"]["stream"] is True
    assert captured["json"]["options"] == {"temperature": 0.7}
    assert "persona_dir" not in captured["json"].get("options", {})


# ---------------------------------------------------------------------------
# _truncate_at_role_leak — strips multi-turn overrun from model replies
# ---------------------------------------------------------------------------


def test_truncate_at_role_leak_no_leak_returns_unchanged() -> None:
    from brain.bridge.provider import _truncate_at_role_leak

    text = "morning, love. *yawns* you're early today."
    assert _truncate_at_role_leak(text) == text


def test_truncate_at_role_leak_strips_user_overrun() -> None:
    """The bug Hana hit 2026-05-07: model finishes its reply, then
    starts scripting the next User: ... Assistant: ... exchange."""
    from brain.bridge.provider import _truncate_at_role_leak

    text = (
        "*pushes glasses up*\n"
        "how do you want to start, love?\n"
        "User: Well since this is our first time in the new home...\n"
        "Assistant: *hands trembling* okay.\n"
    )
    out = _truncate_at_role_leak(text)
    assert "User:" not in out
    assert "Assistant:" not in out
    assert out.endswith("how do you want to start, love?")


def test_truncate_at_role_leak_strips_human_label() -> None:
    """Anthropic-canonical 'Human:' label is also a leak signal."""
    from brain.bridge.provider import _truncate_at_role_leak

    text = "yes, exactly that.\nHuman: thanks\n"
    out = _truncate_at_role_leak(text)
    assert "Human:" not in out
    assert out == "yes, exactly that."


def test_truncate_at_role_leak_handles_lowercase() -> None:
    """Nell's voice is lowercase-leaning; lowercase 'user:' / 'assistant:'
    can leak too."""
    from brain.bridge.provider import _truncate_at_role_leak

    text = "okay.\nuser: more please\nassistant: ..."
    out = _truncate_at_role_leak(text)
    assert "user:" not in out.lower() or out.lower().rstrip() == "okay."


def test_truncate_at_role_leak_only_at_line_start() -> None:
    """Inline mentions inside a sentence are not leaks — only line-starts."""
    from brain.bridge.provider import _truncate_at_role_leak

    # "User:" appears mid-sentence as a quoted reference, NOT as a turn label
    text = 'I asked "what should User: mean here?" and she shrugged.'
    out = _truncate_at_role_leak(text)
    assert out == text  # no truncation


def test_truncate_at_role_leak_does_not_match_persona_names() -> None:
    """Hana legitimately quotes 'Hana:' and 'Nell:' inside fiction.
    The truncator stays away from persona-specific names."""
    from brain.bridge.provider import _truncate_at_role_leak

    text = (
        "the dialogue looked something like:\n"
        "Hana: are you sure?\n"
        "Nell: yes, love.\n"
        "and then they kept walking."
    )
    out = _truncate_at_role_leak(text)
    assert out == text  # persona names inside narrative prose stay intact


def test_llm_provider_complete_delegates_to_generate() -> None:
    """The .complete() shim on the ABC must forward to .generate() with
    system=None. The initiate pipeline calls .complete(); every engine
    historically called .generate(). One surface, two call shapes — the
    shim keeps both contracts honoured.

    FakeProvider exercises the real subclass path: the ABC's default
    .complete implementation should be inherited and produce the same
    result a .generate(prompt, system=None) call would.
    """
    p = FakeProvider()
    via_complete = p.complete("hello world")
    via_generate = p.generate("hello world", system=None)
    assert via_complete == via_generate
    # Sanity: shim isn't a stub returning empty/None.
    assert via_complete.startswith("DREAM: ")
