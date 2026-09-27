"""ram-spike-fix INC-11 — `/persona/state`'s `background_jobs` field
(spec §6, S15/S23/S38; criterion C11(a)).
"""

from __future__ import annotations

from pathlib import Path

from brain.bridge import background_jobs
from brain.bridge.persona_state import build_persona_state


def setup_function() -> None:
    background_jobs._reset_for_tests()


def teardown_function() -> None:
    background_jobs._reset_for_tests()


def test_background_jobs_empty_by_default(tmp_path: Path) -> None:
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    state = build_persona_state(persona_dir=persona_dir)
    assert state["background_jobs"] == []


def test_background_jobs_reflects_the_live_registry(tmp_path: Path) -> None:
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    with background_jobs.running("compaction"), background_jobs.running("heartbeat"):
        state = build_persona_state(persona_dir=persona_dir)
    assert state["background_jobs"] == ["compaction", "heartbeat"]
    # cleared once both context managers exit
    state_after = build_persona_state(persona_dir=persona_dir)
    assert state_after["background_jobs"] == []
