"""Tests for the attunement pass-2 spawn from tool_loop.

Verifies:
- Spawns a daemon thread with a unique, identifiable name
- Skip-list shortcuts (empty buffer, short message) honoured
- Budget cap-reached defers without spawning
- Errors isolated to attunement_errors.jsonl (don't crash main reply path)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from brain.attunement.store import BufferTurn
from brain.chat.tool_loop import _spawn_pass2_attunement


def _wait_for_attunement_threads(timeout: float = 5.0) -> None:
    # Pass-2 now flows through the in-process pass2_queue (single worker, #27),
    # not a per-turn daemon thread. Drain it synchronously (cli_throttle idle in
    # tests via conftest). Kept the name so callers are unchanged.
    from brain.bridge import cli_throttle
    from brain.chat import pass2_queue

    cli_throttle.reset()
    pass2_queue.drain_pending()


def test_spawn_skips_when_buffer_empty(tmp_path: Path):
    with patch("brain.chat.tool_loop.run_detector") as mock_detector:
        _spawn_pass2_attunement(
            tmp_path,
            turn_id="t1",
            user_message="hi there friend, how are you today",
            reply_text="hi",
            buffer_slice=[],
        )
        _wait_for_attunement_threads()
        mock_detector.assert_not_called()


def test_spawn_defers_when_budget_exhausted(tmp_path: Path):
    with patch("brain.chat.tool_loop._attunement_consume_call", return_value=False), \
         patch("brain.chat.tool_loop.run_detector") as mock_detector:
        _spawn_pass2_attunement(
            tmp_path,
            turn_id="t1",
            user_message="I had a long day today, love.",
            reply_text="Tell me about it.",
            buffer_slice=[BufferTurn(id="t1", content="I had a long day today, love.")],
        )
        _wait_for_attunement_threads()
        mock_detector.assert_not_called()


def test_detector_exception_logged_to_errors_jsonl(tmp_path: Path):
    with patch("brain.chat.tool_loop.run_detector", side_effect=RuntimeError("boom")):
        _spawn_pass2_attunement(
            tmp_path,
            turn_id="t1",
            user_message="I had a long day today, love.",
            reply_text="Tell me about it.",
            buffer_slice=[BufferTurn(id="t1", content="I had a long day today, love.")],
        )
        _wait_for_attunement_threads()
    errors_path = tmp_path / "attunement_errors.jsonl"
    assert errors_path.exists()
    assert "boom" in errors_path.read_text()


def test_attunement_spawn_called_from_tool_loop(tmp_path: Path):
    """_spawn_pass2_attunement fires from run_tool_loop on a substantive turn."""
    from brain.bridge.chat import ChatMessage, ChatResponse
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    class _TrivialProvider:
        def chat(self, messages, *, tools=None, options=None):
            return ChatResponse(content="right here with you", tool_calls=(), raw=None)

        def name(self):
            return "trivial"

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    with patch("brain.chat.tool_loop._spawn_pass2_attunement") as mock_spawn:
        try:
            run_tool_loop(
                messages=[ChatMessage(role="user", content="I had a really long day today, love.")],
                provider=_TrivialProvider(),
                tools=None,
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
        finally:
            store.close()
            hebbian.close()

    mock_spawn.assert_called_once()


def test_spawn_skips_when_message_too_short(tmp_path: Path):
    with patch("brain.chat.tool_loop.run_detector") as mock_detector:
        _spawn_pass2_attunement(
            tmp_path,
            turn_id="t1",
            user_message="ok",
            reply_text="ok",
            buffer_slice=[BufferTurn(id="t1", content="ok")],
        )
        _wait_for_attunement_threads()
        mock_detector.assert_not_called()


def test_run_attunement_pass2_calls_mark_addressed(tmp_path: Path):
    """_run_attunement_pass2 calls mark_addressed with addressed_pattern_ids from DetectorOutput."""
    from brain.attunement.schemas import SCHEMA_VERSION, CurrentRead, DetectorOutput
    from brain.chat.tool_loop import _run_attunement_pass2

    fake_read = CurrentRead(
        ts="2026-06-02T10:00:00Z",
        source_turn_id="t1",
        tone_label="warm",
        tone_justification="open",
        cadence_label="flowing",
        cadence_justification="long",
        mood_valence=0.5,
        mood_intensity=0.5,
        predicted_arc_shape="steady",
        schema_version=SCHEMA_VERSION,
    )
    fake_output = DetectorOutput(
        current_read=fake_read,
        pattern_candidates=[],
        addressed_pattern_ids=["abc"],
    )

    with (
        patch("brain.chat.tool_loop.run_detector", return_value=fake_output),
        patch("brain.chat.tool_loop.merge_into_learned"),
        patch("brain.chat.tool_loop.write_current_read"),
        patch("brain.chat.tool_loop.check_crystallisations"),
        patch("brain.chat.tool_loop._attunement_consume_call", return_value=True),
        patch("brain.chat.tool_loop.mark_addressed") as mock_mark,
    ):
        _run_attunement_pass2(
            tmp_path,
            turn_id="t1",
            user_message="I had a long day today, love.",
            reply_text="Tell me about it.",
            buffer_slice=[BufferTurn(id="t1", content="I had a long day today, love.")],
        )

    mock_mark.assert_called_once_with(tmp_path, ["abc"], now_iso=mock_mark.call_args[1]["now_iso"])
