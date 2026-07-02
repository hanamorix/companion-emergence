"""ClaudeCliProvider.chat_stream emits structured events from stream-json stdout.

Event shape probed against claude CLI 2.1.x — see Phase 5A notes. The actual
NDJSON the CLI emits looks like:

    {"type":"stream_event","event":{"type":"message_start","message":{...}}}
    {"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}}
    {"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}}
    {"type":"stream_event","event":{"type":"content_block_stop","index":0}}
    {"type":"stream_event","event":{"type":"message_stop"}}
    {"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}
    {"type":"result","subtype":"success","is_error":false,"result":"hi","duration_ms":...,"num_turns":...}

Thinking deltas (delta.type == "thinking_delta") are NOT surfaced as TextDelta;
they're internal reasoning, not user-visible output. The `assistant` snapshot
frames are ignored — content_block_delta is the source of incremental text.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from brain.bridge.chat import ChatMessage, StreamDone, StreamError, TextDelta
from brain.bridge.provider import ClaudeCliProvider


def _fake_popen(lines: list[str], exit_code: int = 0):
    """Return a MagicMock mimicking subprocess.Popen with stdout iterating over lines."""
    proc = MagicMock()
    proc.stdout = iter(lines)
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.close = MagicMock()
    proc.poll.return_value = exit_code
    proc.wait.return_value = exit_code
    proc.returncode = exit_code
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""
    return proc


def _delta_line(text: str) -> str:
    return (
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": text},
                },
            }
        )
        + "\n"
    )


def _result_line(text: str, **extras) -> str:
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "duration_ms": 1234,
        "num_turns": 1,
        "stop_reason": "end_turn",
        **extras,
    }
    return json.dumps(payload) + "\n"


def test_chat_stream_yields_text_deltas():
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    lines = [
        json.dumps(
            {
                "type": "stream_event",
                "event": {"type": "message_start", "message": {"role": "assistant"}},
            }
        )
        + "\n",
        _delta_line("Hello"),
        _delta_line(" there"),
        json.dumps({"type": "stream_event", "event": {"type": "message_stop"}}) + "\n",
        _result_line("Hello there"),
    ]
    with patch("brain.bridge.provider.subprocess.Popen", return_value=_fake_popen(lines)):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    dones = [e for e in events if isinstance(e, StreamDone)]
    assert [d.text for d in text_deltas] == ["Hello", " there"]
    assert len(dones) == 1
    assert dones[0].content == "Hello there"
    assert dones[0].metadata.get("duration_ms") == 1234
    assert dones[0].metadata.get("num_turns") == 1


def test_chat_stream_ignores_thinking_deltas():
    """thinking_delta is internal reasoning, not user-visible output."""
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    lines = [
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "reasoning..."},
                },
            }
        )
        + "\n",
        _delta_line("visible"),
        _result_line("visible"),
    ]
    with patch("brain.bridge.provider.subprocess.Popen", return_value=_fake_popen(lines)):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert [d.text for d in text_deltas] == ["visible"]


def test_chat_stream_argv_carries_stream_json_flags():
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    lines = [_result_line("")]
    with patch("brain.bridge.provider.subprocess.Popen") as popen_mock:
        popen_mock.return_value = _fake_popen(lines)
        list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    argv = popen_mock.call_args.args[0]
    assert "--dangerously-skip-permissions" in argv
    assert "--output-format" in argv
    assert "stream-json" in argv
    assert "--verbose" in argv  # required for stream_event frames
    assert "--model" in argv
    assert "sonnet" in argv


def test_chat_stream_result_carries_correct_content():
    """chat_stream yields a StreamDone whose content matches the result frame."""
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    lines = [
        _delta_line("ok"),
        _result_line("ok"),
    ]
    with patch("brain.bridge.provider.subprocess.Popen", return_value=_fake_popen(lines)):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    dones = [e for e in events if isinstance(e, StreamDone)]
    assert len(dones) == 1
    assert dones[0].content == "ok"


def test_chat_stream_error_event_on_nonzero_exit():
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    proc = _fake_popen([], exit_code=1)
    proc.stderr.read.return_value = "boom"
    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    errs = [e for e in events if isinstance(e, StreamError)]
    assert len(errs) == 1
    assert errs[0].stage == "claude_cli_exit"


def test_chat_stream_error_propagates_on_nonzero_exit():
    """chat_stream yields StreamError(stage='claude_cli_exit') on non-zero exit."""
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    proc = _fake_popen([], exit_code=1)
    proc.stderr.read.return_value = "auth failure"
    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    errs = [e for e in events if isinstance(e, StreamError)]
    assert len(errs) == 1
    assert errs[0].stage == "claude_cli_exit"


def test_chat_stream_idle_timeout_yields_error(monkeypatch):
    """If no stdout line arrives within the per-event budget, surface idle_timeout."""
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)

    # An iterator that hangs forever — the reader thread will block in
    # stdout.readline, so the consumer-side queue.get will time out first.
    class _SlowStdout:
        def __iter__(self):
            return self

        def __next__(self):
            # Block long enough that the idle-timeout fires.
            import time

            time.sleep(5)
            raise StopIteration

    proc = MagicMock()
    proc.stdout = _SlowStdout()
    proc.stdin = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    proc.returncode = 0
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""

    # Override the idle-timeout knob to keep the test fast.
    monkeypatch.setattr("brain.bridge.provider._STREAM_FIRST_EVENT_SECONDS", 0.1, raising=True)
    monkeypatch.setattr("brain.bridge.provider._STREAM_PER_EVENT_IDLE_SECONDS", 0.1, raising=True)

    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    errs = [e for e in events if isinstance(e, StreamError)]
    assert len(errs) == 1
    assert errs[0].stage == "claude_cli_idle_timeout"
    proc.terminate.assert_called()


def test_chat_stream_early_close_terminates_subprocess():
    """If the consumer abandons the generator mid-stream, Popen.terminate runs."""
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    lines = [_delta_line(str(i)) for i in range(100)] + [_result_line("done")]
    proc = _fake_popen(lines)
    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        gen = provider.chat_stream([ChatMessage(role="user", content="hi")])
        # Pull two events then bail.
        next(gen)
        next(gen)
        gen.close()
    proc.terminate.assert_called()


def test_chat_stream_skips_unparseable_lines():
    """Malformed NDJSON lines do not break the stream."""
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    lines = [
        "garbage not json\n",
        _delta_line("ok"),
        _result_line("ok"),
    ]
    with patch("brain.bridge.provider.subprocess.Popen", return_value=_fake_popen(lines)):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert [d.text for d in text_deltas] == ["ok"]


def test_chat_stream_falls_back_to_assistant_snapshot_when_no_result():
    """If somehow the result frame is missing, fall back to assistant content."""
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    lines = [
        _delta_line("hello"),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                },
            }
        )
        + "\n",
        # NO result frame
    ]
    with patch("brain.bridge.provider.subprocess.Popen", return_value=_fake_popen(lines)):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    dones = [e for e in events if isinstance(e, StreamDone)]
    # We should still get a terminal event — either Done or Error.
    # The acceptable behaviours: Done with joined deltas, OR a StreamError.
    # The implementation should not just hang or yield nothing.
    assert len(events) >= 2  # at least one delta + something terminal
    errs = [e for e in events if isinstance(e, StreamError)]
    assert dones or errs


def test_chat_stream_mcp_tools_argv():
    """Tools-enabled path includes --mcp-config and --allowedTools."""
    import tempfile
    from pathlib import Path

    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    lines = [_result_line("ok")]
    with tempfile.TemporaryDirectory() as td:
        persona = Path(td) / "persona"
        persona.mkdir()
        with patch("brain.bridge.provider.subprocess.Popen") as popen_mock:
            popen_mock.return_value = _fake_popen(lines)
            list(
                provider.chat_stream(
                    [ChatMessage(role="user", content="hi")],
                    tools=[{"name": "noop"}],
                    options={"persona_dir": str(persona)},
                )
            )
        argv = popen_mock.call_args.args[0]
        assert "--mcp-config" in argv
        assert "--allowedTools" in argv
        assert "--output-format" in argv
        assert "stream-json" in argv


# ---- Stream idle-timeout instrumentation (spec 2026-06-01) ----


def _assistant_tool_use_line(tool_name: str) -> str:
    return (
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": tool_name, "input": {}}],
                },
            }
        )
        + "\n"
    )


class _StallAfter:
    """Yield the given lines, then block forever so the idle watchdog fires."""

    def __init__(self, lines):
        self._lines = list(lines)
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < len(self._lines):
            line = self._lines[self._i]
            self._i += 1
            return line
        import time as _t

        _t.sleep(5)
        raise StopIteration


def _stall_proc(lines):
    proc = MagicMock()
    proc.stdout = _StallAfter(lines)
    proc.stdin = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    proc.returncode = 0
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""
    return proc


def test_idle_timeout_records_tool_in_flight(tmp_path, monkeypatch):
    """A tool_use frame then silence → stream_timeouts.jsonl names the tool and
    the per-event budget that fired."""
    persona = tmp_path / "persona"
    persona.mkdir()
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    monkeypatch.setattr("brain.bridge.provider._STREAM_FIRST_EVENT_SECONDS", 0.1, raising=True)
    monkeypatch.setattr("brain.bridge.provider._STREAM_PER_EVENT_IDLE_SECONDS", 0.2, raising=True)

    proc = _stall_proc([_assistant_tool_use_line("mcp__brain-tools__recall_arc")])
    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        events = list(
            provider.chat_stream(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "noop"}],
                options={"persona_dir": str(persona)},
            )
        )

    errs = [e for e in events if isinstance(e, StreamError)]
    assert errs and errs[0].stage == "claude_cli_idle_timeout"
    assert "recall_arc" in errs[0].detail

    log = persona / "stream_timeouts.jsonl"
    assert log.is_file()
    row = json.loads(log.read_text().splitlines()[-1])
    assert row["tool_in_flight"] is True
    assert row["tool_name"] == "mcp__brain-tools__recall_arc"
    assert row["last_frame_type"] == "assistant"
    assert row["budget_s"] == 0.2  # per-event budget fired (a frame arrived first)
    assert row["frames_seen"] == 1


def test_idle_timeout_cold_start_records_no_frame(tmp_path, monkeypatch):
    """No frame ever arrives → first-event budget fired, last_frame_type null,
    frames_seen 0, tool_in_flight false."""
    persona = tmp_path / "persona"
    persona.mkdir()
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    monkeypatch.setattr("brain.bridge.provider._STREAM_FIRST_EVENT_SECONDS", 0.1, raising=True)
    monkeypatch.setattr("brain.bridge.provider._STREAM_PER_EVENT_IDLE_SECONDS", 0.2, raising=True)

    proc = _stall_proc([])  # emits nothing, then blocks
    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        list(
            provider.chat_stream(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "noop"}],
                options={"persona_dir": str(persona)},
            )
        )

    row = json.loads((persona / "stream_timeouts.jsonl").read_text().splitlines()[-1])
    assert row["last_frame_type"] is None
    assert row["frames_seen"] == 0
    assert row["tool_in_flight"] is False
    assert row["budget_s"] == 0.1  # first-event budget fired


def test_idle_timeout_mid_generation_records_stream_event(tmp_path, monkeypatch):
    """A text delta then silence → last_frame_type 'stream_event', no tool."""
    persona = tmp_path / "persona"
    persona.mkdir()
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    monkeypatch.setattr("brain.bridge.provider._STREAM_FIRST_EVENT_SECONDS", 0.1, raising=True)
    monkeypatch.setattr("brain.bridge.provider._STREAM_PER_EVENT_IDLE_SECONDS", 0.2, raising=True)

    proc = _stall_proc([_delta_line("partial")])
    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        list(
            provider.chat_stream(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "noop"}],
                options={"persona_dir": str(persona)},
            )
        )

    row = json.loads((persona / "stream_timeouts.jsonl").read_text().splitlines()[-1])
    assert row["last_frame_type"] == "stream_event"
    assert row["tool_in_flight"] is False
    assert row["tool_name"] is None
    assert row["budget_s"] == 0.2


def test_idle_timeout_without_persona_dir_only_logs(monkeypatch, caplog):
    """No persona_dir (non-tool stream) → no file write, no exception, still
    logger.warning."""
    import logging

    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    monkeypatch.setattr("brain.bridge.provider._STREAM_FIRST_EVENT_SECONDS", 0.1, raising=True)
    monkeypatch.setattr("brain.bridge.provider._STREAM_PER_EVENT_IDLE_SECONDS", 0.1, raising=True)

    proc = _stall_proc([])
    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        with caplog.at_level(logging.WARNING, logger="brain.bridge.provider"):
            events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))

    assert any(isinstance(e, StreamError) for e in events)
    assert any("stream idle timeout" in r.message for r in caplog.records)


def test_normal_stream_writes_no_timeout_log(tmp_path):
    """A normal stream (frames then result) must not create stream_timeouts.jsonl."""
    persona = tmp_path / "persona"
    persona.mkdir()
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)
    lines = [_delta_line("ok"), _result_line("ok")]
    with patch("brain.bridge.provider.subprocess.Popen", return_value=_fake_popen(lines)):
        events = list(
            provider.chat_stream(
                [ChatMessage(role="user", content="hi")],
                tools=[{"name": "noop"}],
                options={"persona_dir": str(persona)},
            )
        )
    assert any(isinstance(e, StreamDone) for e in events)
    assert not (persona / "stream_timeouts.jsonl").exists()


def test_result_frame_terminates_stream_without_spurious_idle_timeout(monkeypatch):
    """A fully-emitted result frame must END the stream — the loop must not keep
    polling for EOF and fire a spurious idle-timeout StreamError on top of an
    already-successful reply (bug-hunt finding #2). Models a subprocess that
    emits the result line then holds stdout open briefly before EOF."""
    provider = ClaudeCliProvider(model="sonnet", timeout_seconds=60)

    class _ResultThenHang:
        def __init__(self):
            self._sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if not self._sent:
                self._sent = True
                return _result_line("done")
            # Hold the pipe open past the (shrunk) idle budget before EOF.
            import time

            time.sleep(1.0)
            raise StopIteration

    proc = MagicMock()
    proc.stdout = _ResultThenHang()
    proc.stdin = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    proc.returncode = 0
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""

    monkeypatch.setattr("brain.bridge.provider._STREAM_FIRST_EVENT_SECONDS", 0.1, raising=True)
    monkeypatch.setattr("brain.bridge.provider._STREAM_PER_EVENT_IDLE_SECONDS", 0.1, raising=True)

    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))

    dones = [e for e in events if isinstance(e, StreamDone)]
    errs = [e for e in events if isinstance(e, StreamError)]
    assert len(dones) == 1 and dones[0].content == "done"
    assert errs == [], f"result frame should terminate cleanly, got {errs}"
