"""Pins for how the reflex engine treats an emotion that is absent from the store, before the readers move.

An arc that names an emotion no memory carries reads it as 0.0 in the trigger test and in the recorded
``trigger_state``. (The template placeholder for a missing emotion, which also renders "0", is already pinned by
``test_reflex.py::test_run_tick_prompt_resolves_variant_spelled_placeholders``.) Deleted with the old branch when
the reader switch-over lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.engines import reflex as reflex_module
from brain.engines.reflex import ReflexEngine
from brain.health.json_io import write_json
from brain.memory.store import Memory, MemoryStore

_REASON = (
    "pins the old reflex branch (absent counts as 0.0 in the trigger test and in trigger_state); "
    "passes on the current code by design and is deleted with the branch at phase C"
)


def _engine_with_one_arc(tmp_path: Path, store: MemoryStore, trigger: dict) -> ReflexEngine:
    arc = {
        "name": "test_arc",
        "description": "test",
        "trigger": trigger,
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "generate_journal",
        "output_memory_type": "reflex_journal",
        "prompt_template": "You are {persona_name}. Write something.",
    }
    arcs_path = tmp_path / "arcs.json"
    write_json(arcs_path, {"version": 1, "arcs": [arc]})
    return ReflexEngine(
        store=store,
        provider=FakeProvider(),
        persona_name="Test",
        persona_system_prompt="You are Test.",
        arcs_path=arcs_path,
        log_path=tmp_path / "log.json",
        default_arcs_path=Path(reflex_module.__file__).parent / "default_reflex_arcs.json",
    )


def _seed(store: MemoryStore, emotions: dict[str, float]) -> None:
    store.create(
        Memory.create_new(
            content="seed",
            memory_type="observation",
            domain="brain",
            emotions=emotions,
            metadata={"source_summary": "conversation:test_seed"},
        )
    )


@pytest.mark.no_base_failure(reason=_REASON)
def test_a_threshold_zero_arc_is_eligible_when_its_emotion_is_absent(tmp_path: Path):
    store = MemoryStore(tmp_path / "memories.db")
    try:
        _seed(store, {"joy": 5.0})
        engine = _engine_with_one_arc(tmp_path, store, {"love": 0})

        result = engine.run_tick(dry_run=True)

        assert result.would_fire == "test_arc"
        assert not any(s.reason == "trigger_not_met" for s in result.arcs_skipped)
    finally:
        store.close()


@pytest.mark.no_base_failure(reason=_REASON)
def test_a_fire_records_an_absent_trigger_emotion_as_zero(tmp_path: Path):
    store = MemoryStore(tmp_path / "memories.db")
    try:
        _seed(store, {"joy": 5.0})
        engine = _engine_with_one_arc(tmp_path, store, {"love": 0})

        result = engine.run_tick(dry_run=False)

        assert len(result.arcs_fired) == 1
        assert result.arcs_fired[0].trigger_state == {"love": 0.0}
    finally:
        store.close()
