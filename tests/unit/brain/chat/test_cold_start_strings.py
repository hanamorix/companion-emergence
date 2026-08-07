"""#98 — cold-start strings must describe their own subsystem, not the
relationship. A persona with a long history can legitimately have zero arcs or
zero lived-age; the string must not then assert that the relationship is new.
"""
from __future__ import annotations

from pathlib import Path

from brain.felt_time.prompt import render_prompt_context
from brain.felt_time.state import FeltTimeState
from brain.narrative_memory.prompt import render_current_arc_block

# Phrasings retired by #98. A future edit must not reintroduce them.
_RETIRED = ("too new", "still forming", "no anchors have seeded")


def test_felt_time_cold_start_does_not_claim_novelty():
    block = render_prompt_context(FeltTimeState())
    lowered = block.lower()
    for phrase in _RETIRED:
        assert phrase not in lowered, f"retired phrasing reintroduced: {phrase!r} in {block!r}"
    assert "felt time" in lowered


def test_arcs_cold_start_does_not_claim_novelty(tmp_path: Path):
    block = render_current_arc_block(tmp_path)
    lowered = block.lower()
    for phrase in _RETIRED:
        assert phrase not in lowered, f"retired phrasing reintroduced: {phrase!r} in {block!r}"
    assert "arcs" in lowered
