"""Engine 6, S107 step (c), HS5 (HS-T): the one test that runs the production step with its spawn left real.

`daemon._rebuild_under_lock` spawns `python -P -m brain.cli supervisor rebuild --persona nell` itself, so this test is the only
one that proves the subcommand, the entry point and the venv-launcher spawn work: every other test fakes the spawn, and a wrong
spawn would ship green and then leave every persona unswitched. It costs one interpreter start per case and a tiny database, and
runs on every CI leg (Windows included). The persona is built under the bridge conftest's tmp `KINDLED_HOME`, and the child
resolves it by name from the same environment.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import pytest

from brain.bridge import daemon
from brain.emotion import engine, engine_files
from brain.emotion.engine_utils import day_of
from brain.paths import get_persona_dir
from tests.conftest import RealPopenSpyLog
from tests.unit.brain.emotion._engine_support import (
    local_day,
    local_noon,
    make_persona,
    row,
    ruin_the_state,
    save,
    simple_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
T20 = local_noon(local_day(20))
SWITCHED = local_noon(local_day(10))
SAVED = local_noon(local_day(30))
BOUND_S = 120


def _persona() -> Path:
    path = get_persona_dir("nell")
    make_persona(path, [row(T20, emotions={"joy": 6.0}, mid="seed")])
    (path / "emotion_vocabulary.json").write_text('{"version": 1, "emotions": []}', encoding="utf-8")
    return path


@pytest.mark.real_rebuild_step(reason="the production step under test")
@pytest.mark.real_rebuild_child(reason="the one test that proves the subcommand, the entry point and the spawn work for real")
@pytest.mark.parametrize("case", ["switched, no newer rows", "unswitched, tiny", "ruined state"])
def test_the_production_step_runs_the_real_child_and_it_exits_zero(
    case: str, rebuild_spawns: RealPopenSpyLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona()
    unchanged: bytes | None = None
    if case.startswith("switched"):
        state = simple_state(saved_at=SAVED, day=day_of(SAVED), switch_over_at=SWITCHED)
        state.rows_through = 1
        save(persona, state)
        unchanged = engine_files.state_path(persona).read_bytes()
    elif case == "ruined state":
        save(persona, simple_state(saved_at=T20))
        ruin_the_state(persona)
    # the child imports the code under test: the checkout's root on its path (bridge_python's env inherits, or copies, os.environ)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH", "")])))

    outcome: dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["result"] = daemon._rebuild_under_lock(persona)
        except BaseException as exc:  # noqa: BLE001 - stored, re-raised by the test thread
            outcome["error"] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(BOUND_S)
    if worker.is_alive():
        for child in rebuild_spawns.children:
            child.kill()
        pytest.fail(f"the rebuild step did not finish within {BOUND_S} s")
    if "error" in outcome:
        raise outcome["error"]

    assert outcome["result"] is None
    assert len(rebuild_spawns.children) == 1
    assert rebuild_spawns.children[0].returncode == 0  # the child's own exit code: a child that died at argv parsing cannot pass
    if unchanged is not None:
        assert engine_files.state_path(persona).read_bytes() == unchanged
        assert engine.engine_for(persona).is_live()
    else:
        assert engine_files.read_state(persona).state is not None
        assert engine.engine_for(persona).is_live()
