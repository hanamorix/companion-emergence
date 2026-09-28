"""ram-spike-fix INC-11 (+ follow-up) — `/persona/state`'s `background_jobs`
field (spec §6, S15/S23/S38; criterion C11(a); follow-up: longest-running-
first ordering with elapsed seconds, owner-set layout).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_background_jobs_reflects_the_live_registry_with_elapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    # Deterministic fake clock injected in place of the registry's `time`
    # module: a real sleep can measure 0 elapsed on a coarse clock (Windows'
    # monotonic clock ticks at ~15.6ms), which made `>` flaky there.
    now = [1000.0]
    monkeypatch.setattr(background_jobs, "time", SimpleNamespace(monotonic=lambda: now[0]))
    with background_jobs.running("compaction"):
        now[0] += 0.5
        with background_jobs.running("heartbeat"):
            now[0] += 0.25
            state = build_persona_state(persona_dir=persona_dir)
    entries = state["background_jobs"]
    assert [e["name"] for e in entries] == ["compaction", "heartbeat"]  # compaction started first
    assert all(isinstance(e["running_for_seconds"], float) for e in entries)
    assert [e["running_for_seconds"] for e in entries] == [0.75, 0.25]
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
