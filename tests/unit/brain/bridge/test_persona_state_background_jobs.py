"""ram-spike-fix INC-11 (+ follow-up) — `/persona/state`'s `background_jobs`
field (spec §6, S15/S23/S38; criterion C11(a); follow-up: longest-running-
first ordering with elapsed seconds, owner-set layout).
"""

from __future__ import annotations

import time
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


def test_background_jobs_reflects_the_live_registry_with_elapsed(tmp_path: Path) -> None:
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    with background_jobs.running("compaction"):
        time.sleep(0.01)
        with background_jobs.running("heartbeat"):
            state = build_persona_state(persona_dir=persona_dir)
    entries = state["background_jobs"]
    assert [e["name"] for e in entries] == ["compaction", "heartbeat"]  # compaction started first
    assert all(isinstance(e["running_for_seconds"], float) for e in entries)
    assert entries[0]["running_for_seconds"] > entries[1]["running_for_seconds"]
    # cleared once both context managers exit
    state_after = build_persona_state(persona_dir=persona_dir)
    assert state_after["background_jobs"] == []


def test_background_jobs_field_shape_is_list_of_name_and_running_for_seconds(
    tmp_path: Path,
) -> None:
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    with background_jobs.running("pass2"):
        state = build_persona_state(persona_dir=persona_dir)
    assert state["background_jobs"] == [
        {"name": "pass2", "running_for_seconds": state["background_jobs"][0]["running_for_seconds"]}
    ]
    assert set(state["background_jobs"][0].keys()) == {"name", "running_for_seconds"}
