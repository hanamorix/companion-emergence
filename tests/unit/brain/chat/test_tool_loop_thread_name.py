"""Pass-2 work items carry distinct labels so logs disambiguate.

Pre-#27 each pass-2 ran in its own daemon thread with a unique name; now they
run in a single queue worker (brain.chat.pass2_queue), so the disambiguation
moved from thread names to per-item labels. This test guards that the labels
stay distinct (the original log-disambiguation intent).
"""
from __future__ import annotations

from pathlib import Path

from brain.bridge.provider import FakeProvider
from brain.chat import pass2_queue, tool_loop


def test_concurrent_pass2_items_have_distinct_labels(tmp_path: Path):
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    for i in range(3):
        tool_loop._spawn_pass2(
            provider=FakeProvider(),
            monologue_text=f"monologue {i}",
            visible_reply="reply",
            recent_user_msgs=(),
            persona_dir=persona_dir,
        )

    # Worker is inhibited in tests (conftest), so the 3 items sit in the queue.
    labels = [label for (_fn, label) in pass2_queue._queue]
    assert len(labels) == 3
    assert len(set(labels)) == 3, f"expected 3 distinct labels, got {labels}"
    assert all(la.startswith("monologue-") for la in labels)
