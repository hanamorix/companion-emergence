"""Tests for brain.bridge.usage_log and its integration into provider/detector."""
from __future__ import annotations

import io
import json
import subprocess
from unittest.mock import patch

from brain.bridge.chat import ChatMessage, StreamDone
from brain.bridge.provider import ClaudeCliProvider
from brain.bridge.usage_log import log_usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_result(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(payload), stderr=""
    )


# ---------------------------------------------------------------------------
# Unit tests for log_usage helper
# ---------------------------------------------------------------------------


def test_log_usage_writes_one_line_from_a_result_frame(tmp_path):
    frame = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 43,
            "cache_creation_input_tokens": 37264,
            "cache_read_input_tokens": 0,
        },
        "total_cost_usd": 0.047,
        "num_turns": 1,
        "duration_ms": 1200,
        "session_id": "abc",
    }
    log_usage(tmp_path, call_type="chat", model="sonnet", frame=frame)
    lines = (tmp_path / "chat_usage.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["input_tokens"] == 10
    assert row["cache_creation_input_tokens"] == 37264
    assert row["call_type"] == "chat"
    assert row["total_cost_usd"] == 0.047


def test_log_usage_best_effort_never_raises(tmp_path):
    log_usage(tmp_path, call_type="chat", model="sonnet", frame={})  # missing usage → must not raise


def test_log_usage_none_persona_dir_is_noop(tmp_path):
    log_usage(None, call_type="generate", model="haiku", frame={"usage": {"input_tokens": 1}})  # no crash, no file


def test_log_usage_creates_persona_dir_if_missing(tmp_path):
    missing = tmp_path / "new_persona" / "subdir"
    # must not exist yet
    assert not missing.exists()
    log_usage(missing, call_type="chat", model="haiku",
              frame={"usage": {"input_tokens": 1}, "total_cost_usd": 0.001})
    assert (missing / "chat_usage.jsonl").exists()


# ---------------------------------------------------------------------------
# ClaudeCliProvider.generate() integration
# ---------------------------------------------------------------------------


def test_generate_logs_usage_when_persona_dir_given(tmp_path):
    """ClaudeCliProvider.generate() writes a chat_usage.jsonl row when persona_dir provided."""
    fake_payload = {
        "result": "hello",
        "usage": {
            "input_tokens": 5,
            "output_tokens": 12,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 0,
        },
        "total_cost_usd": 0.001,
        "num_turns": 1,
        "duration_ms": 500,
        "session_id": "xyz",
    }
    provider = ClaudeCliProvider(model="haiku")
    with patch("subprocess.run", return_value=_fake_result(fake_payload)):
        result = provider.generate("hi", persona_dir=tmp_path)
    assert result == "hello"
    lines = (tmp_path / "chat_usage.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["call_type"] == "generate"
    assert row["input_tokens"] == 5
    assert row["total_cost_usd"] == 0.001


# ---------------------------------------------------------------------------
# ClaudeCliProvider.chat_stream() integration
# ---------------------------------------------------------------------------


def test_chat_stream_logs_usage_on_result_frame(tmp_path):
    """ClaudeCliProvider.chat_stream() writes a chat_usage.jsonl row on the result frame."""
    result_frame = json.dumps({
        "type": "result",
        "result": "nice reply",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 30,
            "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 10,
        },
        "total_cost_usd": 0.002,
        "num_turns": 1,
        "duration_ms": 800,
        "session_id": "s1",
    })

    fake_stdout_lines = [result_frame + "\n"]

    class _FakeProc:
        returncode = 0
        stdin = io.StringIO()
        stderr = io.StringIO()
        stdout = iter(fake_stdout_lines)

        def terminate(self) -> None:
            pass

        def wait(self) -> int:
            return 0

    provider = ClaudeCliProvider(model="sonnet")
    messages = [ChatMessage(role="user", content="hello")]

    with patch("subprocess.Popen", return_value=_FakeProc()):
        events = list(provider.chat_stream(
            messages,
            options={"persona_dir": str(tmp_path)},
        ))

    done_events = [e for e in events if isinstance(e, StreamDone)]
    assert done_events, "expected at least one StreamDone"

    lines = (tmp_path / "chat_usage.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["call_type"] == "chat"
    assert row["input_tokens"] == 20
    assert row["total_cost_usd"] == 0.002


# ---------------------------------------------------------------------------
# ClaudeCliProvider.chat() non-streaming text path integration
# ---------------------------------------------------------------------------


def test_nonstreaming_chat_logs_usage(tmp_path):
    """ClaudeCliProvider.chat() (no tools, no images) writes a chat_usage.jsonl row."""
    payload = {
        "result": "hi",
        "usage": {
            "input_tokens": 5,
            "output_tokens": 9,
            "cache_creation_input_tokens": 1234,
            "cache_read_input_tokens": 0,
        },
        "total_cost_usd": 0.01,
        "num_turns": 1,
        "duration_ms": 100,
        "session_id": "s",
    }
    provider = ClaudeCliProvider(model="sonnet")
    with patch(
        "brain.bridge.provider.subprocess.run",
        return_value=_fake_result(payload),
    ):
        provider.chat(
            [ChatMessage(role="user", content="hi")],
            options={"persona_dir": str(tmp_path)},
        )

    rows = (tmp_path / "chat_usage.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["call_type"] == "chat"
    assert row["input_tokens"] == 5
    assert row["output_tokens"] == 9
    assert row["cache_creation_input_tokens"] == 1234
    assert row["total_cost_usd"] == 0.01


# ---------------------------------------------------------------------------
# Attunement detector threading
# ---------------------------------------------------------------------------


def test_run_detector_passes_persona_dir_to_generate(tmp_path):
    """run_detector accepts persona_dir and passes it through to generate() so usage is logged."""
    from brain.attunement.detector import run_detector
    from brain.attunement.store import BufferTurn

    fake_payload = {
        "result": json.dumps({
            "current_read": {
                "tone_label": "warm",
                "tone_justification": "x",
                "cadence_label": "fast",
                "cadence_justification": "y",
                "mood_valence": 0.5,
                "mood_intensity": 0.5,
                "predicted_arc_shape": "rising",
            },
            "pattern_candidates": [],
        }),
        "usage": {
            "input_tokens": 7,
            "output_tokens": 9,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "total_cost_usd": 0.0005,
        "num_turns": 1,
        "duration_ms": 300,
        "session_id": "att1",
    }
    buffer = [BufferTurn(id="t1", content="hello there how are you doing today")]
    with patch("subprocess.run", return_value=_fake_result(fake_payload)):
        run_detector(buffer, reply_text="fine thanks", persona_dir=tmp_path)

    lines = (tmp_path / "chat_usage.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["call_type"] == "generate"
    assert row["input_tokens"] == 7
