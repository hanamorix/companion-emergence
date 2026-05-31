"""Verify the chat prompt builder includes the attunement block.

Task 13 — spec §14: attunement → ambient prompt is the primary consumer
wire-back. build_system_message must include the rendered attunement
block (current read + learned patterns) in the assembled system message.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from brain.attunement.schemas import SCHEMA_VERSION, CurrentRead
from brain.attunement.store import write_current_read
from brain.chat.prompt import build_system_message
from brain.engines.daemon_state import DaemonState
from brain.memory.store import MemoryStore
from brain.soul.store import SoulStore


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memories.db")
    yield s
    s.close()


@pytest.fixture()
def soul_store() -> SoulStore:
    ss = SoulStore(":memory:")
    yield ss
    ss.close()


def test_attunement_block_appears_in_assembled_system_message(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """When a current_read exists, build_system_message includes tone + cadence."""
    write_current_read(
        persona_dir,
        CurrentRead(
            ts="2026-05-31T12:00:00Z",
            source_turn_id="t1",
            tone_label="warm",
            tone_justification="soft phrasing throughout",
            cadence_label="measured",
            cadence_justification="full sentences, no rushing",
            mood_valence=0.3,
            mood_intensity=0.5,
            predicted_arc_shape="settling in",
            schema_version=SCHEMA_VERSION,
        ),
    )

    system = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=DaemonState(),
        soul_store=soul_store,
        store=store,
    )

    assert "warm" in system
    assert "measured" in system
