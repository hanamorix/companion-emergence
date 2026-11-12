"""`_mem_to_result` (search and read results) carries the engine's faded emotions for the persona."""

from __future__ import annotations

import pytest

from brain.memory.store import Memory
from brain.tools.impls._common import _mem_to_result


@pytest.mark.no_base_failure(
    reason=(
        "moved unchanged from tests/unit/brain/tools/test_old_branch_mem_to_result.py (deleted with the old reader "
        "branches); the behaviour it checks exists on the base, so it passes there by design"
    )
)
def test_a_search_result_carries_the_faded_emotions(tmp_path, emotion_engine_stub):
    """The tool result's emotions are the engine's faded values for this persona."""
    memory = Memory.create_new(
        content="a stored memory",
        memory_type="episodic",
        domain="us",
        emotions={"love": 7.2, "grief": 3.4},
    )
    emotion_engine_stub.faded = {(memory.id, "love"): 4.0, (memory.id, "grief"): 1.0}
    assert _mem_to_result(memory, tmp_path)["emotions"] == {"love": 4.0, "grief": 1.0}
