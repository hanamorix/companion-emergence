"""Pass-2 spawners enqueue into pass2_queue instead of spawning threads (#27)."""
from __future__ import annotations

from brain.attunement.store import BufferTurn
from brain.bridge.provider import FakeProvider
from brain.chat import pass2_queue, tool_loop


def test_spawn_pass2_attunement_enqueues(tmp_path):
    slice_ = [BufferTurn(id="msg-0", content="this is a substantive user turn with plenty of words")]
    tool_loop._spawn_pass2_attunement(
        tmp_path,
        "turn-1",
        "this is a substantive user turn with plenty of words",
        "her reply",
        slice_,
    )
    assert pass2_queue._queue_size() == 1


def test_spawn_pass2_enqueues_instead_of_threading(tmp_path):
    tool_loop._spawn_pass2(
        provider=FakeProvider(),
        monologue_text="a thought worth keeping",
        visible_reply="hi love",
        recent_user_msgs=("how are you?",),
        persona_dir=tmp_path,
    )
    assert pass2_queue._queue_size() == 1


def test_spawn_pass2_attunement_skips_trivial_turn(tmp_path):
    # should_run_detector gate: a too-short user message enqueues nothing
    slice_ = [BufferTurn(id="msg-0", content="ok")]
    tool_loop._spawn_pass2_attunement(tmp_path, "turn-1", "ok", "reply", slice_)
    assert pass2_queue._queue_size() == 0
