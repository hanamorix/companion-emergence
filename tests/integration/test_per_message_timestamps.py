"""End-to-end: per-message timestamps flow from buffer file to the
Claude context block, rendered in the companion's local time (issue #217)
at the display seam - storage stays UTC throughout."""

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from brain.bridge.provider import _format_claude_context_block
from brain.chat.engine import _buffer_turns_to_messages
from brain.ingest.buffer import read_session

# A fixed non-UTC zone (not the host's OS timezone): brain.utils.time.to_local
# treats a non-zero offset as already local and returns it unconverted,
# whereas a UTC-aware `now` gets re-converted via astimezone() to whatever
# zone the test host runs in. Used for the `now` anchor below so its
# assertion is deterministic regardless of the CI runner's TZ - same
# pattern as tests/unit/brain/initiate/test_gates.py.
_LOCAL = ZoneInfo("America/Los_Angeles")


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

    # Storage stays UTC: both ts values survive into the ChatMessage objects
    # unconverted - the buffer round-trip is unaffected by #217.
    assert messages[0].ts == "2026-05-20T10:00:00Z"
    assert messages[1].ts == "2026-05-20T10:00:30Z"

    # But the context block JSONL the companion actually reads renders ts in
    # local wall-clock time (#217), not the raw UTC string relabeled. The
    # exact offset depends on the CI runner's OS timezone, so assert what's
    # deterministic regardless of it: the raw 'Z'-suffixed UTC string does
    # NOT survive verbatim, and the rendered value resolves back to the
    # exact same instant (a true zone conversion, not a stray reformat).
    now = datetime(2026, 5, 20, 7, 30, 0, tzinfo=_LOCAL)
    block = _format_claude_context_block(messages, includes_latest_user=True, now=now)
    records = [json.loads(line) for line in block.splitlines() if line.startswith("{")]
    assert records[0]["ts"] != "2026-05-20T10:00:00Z"
    assert records[1]["ts"] != "2026-05-20T10:00:30Z"
    assert not records[0]["ts"].endswith("Z")
    assert not records[1]["ts"].endswith("Z")
    assert datetime.fromisoformat(records[0]["ts"]).astimezone(UTC) == datetime(
        2026, 5, 20, 10, 0, 0, tzinfo=UTC
    )
    assert datetime.fromisoformat(records[1]["ts"]).astimezone(UTC) == datetime(
        2026, 5, 20, 10, 0, 30, tzinfo=UTC
    )

    # The preamble carries the "Current time" anchor in local wall-clock
    # time too (injected `now` above is trusted as already-local, per
    # brain.utils.time.to_local's guard - deterministic regardless of TZ),
    # rendered with no offset suffix (issue #218 - the offset token was
    # itself what the substrate was echoing back).
    assert "Current time: 2026-05-20T07:30:00." in block
    assert "Current time: 2026-05-20T07:30:00Z." not in block
    assert "Current time: 2026-05-20T07:30:00-07:00." not in block


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
