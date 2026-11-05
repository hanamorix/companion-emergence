"""Pin for the search and read result's emotions field, before the readers move to the engine."""

from __future__ import annotations

import pytest

from brain.memory.store import Memory
from brain.tools.impls._common import _mem_to_result


@pytest.mark.no_base_failure(
    reason=(
        "pins the old search result branch (stored emotions); "
        "passes on the current code by design and is deleted with the branch at phase C"
    )
)
def test_a_search_result_carries_the_stored_emotions():
    memory = Memory.create_new(
        content="a stored memory",
        memory_type="episodic",
        domain="us",
        emotions={"love": 7.2, "grief": 3.4},
    )

    assert _mem_to_result(memory)["emotions"] == {"love": 7.2, "grief": 3.4}
