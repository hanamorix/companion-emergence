"""Pins for two prompt blocks that read emotions in their own way, before the readers move to the engine.

Both are deleted with the old branch they pin when the reader switch-over lands.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.chat import prompt
from brain.memory.store import Memory, MemoryStore


def _stored(store: MemoryStore, created_at: datetime, memory_type: str, emotions: dict[str, float]) -> None:
    mem = Memory.create_new(
        content=f"{memory_type} at {created_at.isoformat()}",
        memory_type=memory_type,
        domain="us",
        emotions=emotions,
    )
    store.create(dataclasses.replace(mem, created_at=created_at))


@pytest.mark.no_base_failure(
    reason=(
        "pins the old mood summary branch (newest 200 emotion-bearing rows, top 3 by pooled value); "
        "passes on the current code by design and is deleted with the branch at phase C"
    )
)
def test_emotion_summary_reads_the_newest_200_emotion_bearing_rows(tmp_path: Path):
    base = datetime.now(UTC) - timedelta(hours=8)
    store = MemoryStore(tmp_path / "memories.db")
    try:
        rows: list[tuple[str, dict[str, float]]] = [
            ("episodic", {"nostalgia": 9.0}),  # the 201st emotion-bearing row from the newest: out of the window
            ("episodic", {"relief": 4.0}),
        ]
        rows += [("episodic", {"joy": 2.0, "awe": 3.0, "love": 6.0})] * 199
        rows += [("heartbeat", {})] * 30
        for i, (memory_type, emotions) in enumerate(rows):
            _stored(store, base + timedelta(minutes=i), memory_type, emotions)

        assert prompt._build_emotion_summary(store) == "love:6.0, relief:4.0, awe:3.0"
    finally:
        store.close()


@pytest.mark.no_base_failure(
    reason=(
        "pins the old journal block branch (top 2 by stored value); "
        "passes on the current code by design and is deleted with the branch at phase C"
    )
)
def test_recent_journal_block_names_the_two_strongest_stored_emotions(tmp_path: Path):
    store = MemoryStore(tmp_path / "memories.db")
    try:
        _stored(store, datetime.now(UTC), "journal_entry", {"love": 9.0, "joy": 4.0, "awe": 2.0})

        block = prompt._build_recent_journal_block(store)

        assert "primary: love 9, joy 4" in block
        assert "awe" not in block
    finally:
        store.close()
