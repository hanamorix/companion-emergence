"""scripts/time_old_emotion_paths.py - the start step: the copy is started through `ensure_engine_ready` and must be live.

The script starts the scratch copy once, untimed, after the vocabulary load, so that the timed calls read the live engine
(Engine 6, spec 3.1 S6). It skips the step on a commit with no `ensure_engine_ready` and ends the run when the copy is not
live afterwards.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "time_old_emotion_paths.py"

_spec = importlib.util.spec_from_file_location("time_old_emotion_paths_start_under_test", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


def _persona_with_a_database(root: Path) -> Path:
    from brain.memory.store import MemoryStore

    root.mkdir(parents=True, exist_ok=True)
    MemoryStore(root / "memories.db").close()
    return root


def test_the_start_step_runs_the_first_rebuild_and_leaves_the_engine_live(tmp_path: Path) -> None:
    from brain.emotion import engine

    persona = _persona_with_a_database(tmp_path / "persona")
    assert not engine.engine_for(persona).is_live()
    lines = mod._step_start(persona)
    assert engine.engine_for(persona).is_live()
    assert len(lines) == 1 and "ensure_engine_ready" in lines[0] and "live" in lines[0]


def test_the_start_step_is_skipped_with_a_reason_on_a_commit_without_ensure_engine_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brain.emotion import engine

    monkeypatch.delattr(engine, "ensure_engine_ready")
    lines = mod._step_start(_persona_with_a_database(tmp_path / "persona"))
    assert lines == ["start: skipped, ensure_engine_ready is not on this commit"]


def test_a_copy_that_is_not_live_after_the_start_step_ends_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.emotion import engine

    monkeypatch.setattr(engine, "ensure_engine_ready", lambda persona_dir: None)  # a start that switches nothing over
    with pytest.raises(mod.StepError) as raised:
        mod._step_start(_persona_with_a_database(tmp_path / "persona"))
    assert raised.value.step == "start"
