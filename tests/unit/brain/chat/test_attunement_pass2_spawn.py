"""Tests for the attunement pass-2 spawn from tool_loop.

Verifies:
- Pass-2 work is queued (one item per spawn) for the central cadence to drain
- Skip-list shortcuts (empty buffer, short message) honoured
- Budget cap-reached defers without spawning
- Errors isolated to attunement_errors.jsonl (don't crash main reply path)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from brain.attunement.store import BufferTurn
from brain.chat.tool_loop import _spawn_pass2_attunement
from tests.thread_support import finish_threads


def _drain_pass2_queue(persona_dir) -> None:
    # Pass-2 work is saved in the persisted pass2_queue (no worker thread, #27);
    # drain it synchronously (cli_throttle is idle in tests via conftest).
    from brain.bridge import cli_throttle
    from brain.chat import pass2_queue

    cli_throttle.reset()
    pass2_queue.drain_pending(persona_dir)


def test_spawn_skips_when_buffer_empty(tmp_path: Path):
    with patch("brain.chat.tool_loop.run_detector") as mock_detector:
        _spawn_pass2_attunement(
            tmp_path,
            turn_id="t1",
            user_message="hi there friend, how are you today",
            reply_text="hi",
            buffer_slice=[],
        )
        _drain_pass2_queue(tmp_path)
        mock_detector.assert_not_called()


@pytest.mark.no_base_failure(reason='#540 rewrite of an existing test (new id from a rename or a new parameter) asserting what its predecessor asserted; the old code passes it')
@pytest.mark.parametrize(("message", "queued"), [("yes that is fine", 0), ("yes that is totally fine", 1)])
def test_the_spawn_gate_is_five_words(tmp_path: Path, message: str, queued: int):
    """The call site honours the five-word default (the boundary rule itself is in test_detector_skip_list.py):
    a 4-word message queues no pass-2 item, a 5-word message queues one. Nothing drains the queue here."""
    from brain.chat import pass2_queue

    _spawn_pass2_attunement(
        tmp_path,
        turn_id="t1",
        user_message=message,
        reply_text="ok",
        buffer_slice=[BufferTurn(id="t1", content=message)],
    )
    assert pass2_queue._queue_size(tmp_path) == queued


def test_spawn_defers_when_budget_exhausted(tmp_path: Path):
    # #473: the budget is checked read-only before the call (and charged after it), so exhaustion is seen there.
    with patch("brain.chat.tool_loop._attunement_budget_remaining", return_value=0), \
         patch("brain.chat.tool_loop.run_detector") as mock_detector:
        _spawn_pass2_attunement(
            tmp_path,
            turn_id="t1",
            user_message="I had a long day today, love.",
            reply_text="Tell me about it.",
            buffer_slice=[BufferTurn(id="t1", content="I had a long day today, love.")],
        )
        _drain_pass2_queue(tmp_path)
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
        _drain_pass2_queue(tmp_path)
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
        _drain_pass2_queue(tmp_path)
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


def test_pass2_threads_user_name_from_persona_config(tmp_path: Path):
    """v0.0.33 identity-grounding: the live pass-2 path reads user_name from
    persona_config.json and threads it to the detector (no hardcoded names)."""
    import json

    (tmp_path / "persona_config.json").write_text(json.dumps({"user_name": "Alex"}))
    with patch("brain.chat.tool_loop.run_detector") as mock_detector:
        _spawn_pass2_attunement(
            tmp_path,
            turn_id="t1",
            user_message="I had a long day today, love.",
            reply_text="Tell me about it.",
            buffer_slice=[BufferTurn(id="t1", content="I had a long day today, love.")],
        )
        _drain_pass2_queue(tmp_path)
    assert mock_detector.call_args.kwargs["user_name"] == "Alex"


def test_two_attunement_calls_at_once_cannot_overshoot_the_daily_cap(tmp_path: Path):
    """Pre-merge review O-7: the budget is checked before the call and charged after it (D24), so the check and the
    charge must be one critical section: two callers with one call left must not both pass the check. The first
    detector call waits (up to 5 s) until a second caller has arrived at the budget lock; with the lock, the second
    waits there instead and then finds the day's budget spent. Without the lock around check and charge nobody ever
    arrives at it, the second caller reaches the detector too, and the test fails."""
    import threading
    from contextlib import contextmanager
    from datetime import UTC, datetime

    from brain.attunement import budget
    from brain.attunement.schemas import (
        DAILY_BUDGET_DEFAULT,
        SCHEMA_VERSION,
        CurrentRead,
        DetectorOutput,
    )
    from brain.chat import tool_loop
    from brain.chat.tool_loop import _run_attunement_pass2

    now = datetime.now(UTC)
    for _ in range(DAILY_BUDGET_DEFAULT - 1):
        assert budget.consume_call(tmp_path, now=now)
    assert budget.get_remaining(tmp_path, now=now) == 1

    fake_read = CurrentRead(
        ts="2026-06-02T10:00:00Z", source_turn_id="t1", tone_label="warm", tone_justification="open",
        cadence_label="flowing", cadence_justification="long", mood_valence=0.5, mood_intensity=0.5,
        predicted_arc_shape="steady", schema_version=SCHEMA_VERSION,
    )
    calls: list[int] = []
    arrivals: list[int] = []
    second_arrived = threading.Event()
    real_lock = tool_loop._attunement_budget_lock

    @contextmanager
    def tracked_lock(persona_dir):
        arrivals.append(1)
        if len(arrivals) >= 2:
            second_arrived.set()
        with real_lock(persona_dir):
            yield

    def detector(**_kw):
        calls.append(1)
        if len(calls) == 1:
            second_arrived.wait(timeout=5.0)  # bounded on purpose: with the lock removed (the regression) nobody ever sets it
        return DetectorOutput(current_read=fake_read, pattern_candidates=[], addressed_pattern_ids=[])

    def run():
        _run_attunement_pass2(
            tmp_path, turn_id="t1", user_message="I had a long day today, love.", reply_text="Tell me about it.",
            buffer_slice=[BufferTurn(id="t1", content="I had a long day today, love.")],
        )

    with (
        patch("brain.chat.tool_loop.run_detector", side_effect=detector),
        patch("brain.chat.tool_loop._attunement_budget_lock", tracked_lock),
        patch("brain.chat.tool_loop.merge_into_learned"),
        patch("brain.chat.tool_loop.write_current_read"),
        patch("brain.chat.tool_loop.check_crystallisations"),
        patch("brain.chat.tool_loop.mark_addressed"),
    ):
        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            finish_threads(t)
    assert len(calls) == 1
    assert budget._load(tmp_path)["count"] == DAILY_BUDGET_DEFAULT
