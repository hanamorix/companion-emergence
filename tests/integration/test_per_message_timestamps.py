"""End-to-end: per-message timestamps flow from buffer file to the
Claude context block."""

import json
import re
from pathlib import Path

from brain.bridge.provider import _format_claude_context_block
from brain.chat.engine import _buffer_turns_to_messages
from brain.ingest.buffer import read_session


def test_buffer_timestamps_appear_in_context_block(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    persona_dir = tmp_path / "personas" / "test_persona"
    (persona_dir / "active_conversations").mkdir(parents=True)

    # Buffer with two timestamped turns using the on-disk schema (speaker/text/ts)
    buffer_path = persona_dir / "active_conversations" / "test-session.jsonl"
    turns = [
        {
            "session_id": "test-session",
            "speaker": "user",
            "text": "hi",
            "ts": "2026-05-20T10:00:00Z",
        },
        {
            "session_id": "test-session",
            "speaker": "assistant",
            "text": "hello",
            "ts": "2026-05-20T10:00:30Z",
        },
    ]
    with open(buffer_path, "w") as fh:
        for t in turns:
            fh.write(json.dumps(t) + "\n")

    # Load via read_session (the canonical reader) then build ChatMessage list
    loaded_turns = read_session(persona_dir, "test-session")
    messages = _buffer_turns_to_messages(persona_dir, loaded_turns)

    # Both ts values should survive into the ChatMessage objects
    assert messages[0].ts == "2026-05-20T10:00:00Z"
    assert messages[1].ts == "2026-05-20T10:00:30Z"

    # And end up in the context block JSONL
    block = _format_claude_context_block(messages, includes_latest_user=True)
    records = [json.loads(line) for line in block.splitlines() if line.startswith("{")]
    assert records[0]["ts"] == "2026-05-20T10:00:00Z"
    assert records[1]["ts"] == "2026-05-20T10:00:30Z"

    # The preamble should also carry a real timestamp anchor
    assert re.search(r"Current time: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", block)


def test_old_buffer_without_ts_loads_cleanly(tmp_path: Path, monkeypatch):
    """Pre-v0.0.16 buffers that have no ts field load without error."""
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    persona_dir = tmp_path / "personas" / "test_persona"
    (persona_dir / "active_conversations").mkdir(parents=True)

    buffer_path = persona_dir / "active_conversations" / "test-session.jsonl"
    turns = [
        {"session_id": "test-session", "speaker": "user", "text": "hi"},
        {"session_id": "test-session", "speaker": "assistant", "text": "hello"},
    ]
    with open(buffer_path, "w") as fh:
        for t in turns:
            fh.write(json.dumps(t) + "\n")

    loaded_turns = read_session(persona_dir, "test-session")
    messages = _buffer_turns_to_messages(persona_dir, loaded_turns)

    # ts should be None for both (not an error)
    assert messages[0].ts is None
    assert messages[1].ts is None

    # Context block should still emit correctly — no ts fields in JSONL records
    block = _format_claude_context_block(messages, includes_latest_user=True)
    records = [json.loads(line) for line in block.splitlines() if line.startswith("{")]
    assert "ts" not in records[0]
    assert "ts" not in records[1]
