"""Pass-2 work items carry distinct ids so logs disambiguate.

Pre-#27 each pass-2 ran in its own daemon thread with a unique name; now they
run in a single queue worker (brain.chat.pass2_queue), so the disambiguation
moved from thread names to per-item labels, and (ram-spike-fix INC-8, S64) to
each persisted record's own "id" once the queue moved off in-memory closures
(which carried a label but couldn't be persisted at all). This test guards
that ids stay distinct (the original log-disambiguation intent).
"""
from __future__ import annotations

from pathlib import Path

from brain.bridge.provider import FakeProvider
from brain.chat import pass2_queue, tool_loop


def test_concurrent_pass2_items_have_distinct_ids(tmp_path: Path):
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

    # Worker is inhibited in tests (conftest), so the 3 items sit persisted.
    items = pass2_queue._load_queue_unlocked(persona_dir)
    ids = [it["id"] for it in items]
    assert len(ids) == 3
    assert len(set(ids)) == 3, f"expected 3 distinct ids, got {ids}"
    assert all(it["kind"] == "monologue" for it in items)
