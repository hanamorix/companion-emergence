"""Pins for two places the forgetting code reads a memory's stored emotions, before the readers move.

The salience input is the stored maximum over 10, and a graveyard entry carries the stored emotions as they were.
Deleted with the old branch when the reader switch-over lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.forgetting import graveyard
from brain.forgetting.salience import SalienceInputs, _emotion_input
from brain.health.json_io import read_jsonl
from brain.memory.store import Memory


@pytest.mark.no_base_failure(
    reason=(
        "pins the old salience input branch (stored maximum over 10); "
        "passes on the current code by design and is deleted with the branch at phase C"
    )
)
def test_emotion_input_is_the_stored_peak_over_ten():
    memory = Memory.create_new(
        content="a stored memory", memory_type="episodic", domain="us", emotions={"joy": 7.2, "love": 9.0}
    )

    assert _emotion_input(memory, None) == pytest.approx(0.9)


@pytest.mark.no_base_failure(
    reason=(
        "pins the old graveyard branch (stored emotions); "
        "passes on the current code by design and is deleted with the branch at phase C"
    )
)
def test_graveyard_entry_carries_the_stored_emotions(tmp_path: Path):
    memory = Memory.create_new(
        content="apple banana", memory_type="episodic", domain="chat", emotions={"grief": 7.2, "love": 3.0}
    )

    graveyard.append(
        tmp_path,
        memory=memory,
        salience_at_drop=0.07,
        inputs=SalienceInputs(emotion=0.72, hebbian=0.05, recall=0.0, soul=0.0, freshness=0.0),
        lived_age_hours=612.4,
        reason="salience<0.10 for 2 consecutive passes",
    )

    entries = read_jsonl(tmp_path / graveyard.GRAVEYARD_FILENAME)
    assert len(entries) == 1
    assert entries[0]["emotion_at_ingest"] == {"grief": 7.2, "love": 3.0}
