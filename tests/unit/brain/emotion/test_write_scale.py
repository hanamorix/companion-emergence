"""The scale rule of the writers that only pick a scale (emotion rework B1 PR 7, ThinkerOfThoughts' ruling on #633).

Switched over → the engine's 0-10 scale. Otherwise (not switched, or the state cannot be read) the x10 fix's progress note
decides against the writer's cut. A progress note that cannot be read → today's small scale.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.dev_constants import NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED as QUEUE
from brain.dev_constants import NUDGE_SCALE_FIX_PROGRESS_ROWS_DONE as ROWS
from brain.emotion.write_scale import writes_engine_scale
from brain.memory.store import MemoryStore
from brain.utils import log_throttle
from tests.unit.brain.emotion._engine_support import make_live, ruin_the_state


def _persona(tmp_path: Path, switch: str, progress: int) -> tuple[Path, MemoryStore]:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    store = MemoryStore(persona_dir / "memories.db")
    store.set_x10_progress(progress)
    if switch in ("switched", "unreadable"):
        make_live(persona_dir, datetime.now(UTC))
    if switch == "unreadable":
        ruin_the_state(persona_dir)
    return persona_dir, store


@pytest.mark.parametrize(
    ("switch", "progress", "cut", "expected"),
    [
        # The switch-over answer comes first: a switched persona writes 0-10 whatever the progress note says.
        ("switched", 0, ROWS, True),
        ("switched", 0, QUEUE, True),
        # Not switched over: the progress note decides against the writer's cut.
        ("not_switched", 0, ROWS, False),
        ("not_switched", ROWS, ROWS, True),
        ("not_switched", ROWS, QUEUE, False),  # the one case that tells the per-writer cut from a single cut
        ("not_switched", QUEUE, ROWS, True),
        ("not_switched", QUEUE, QUEUE, True),
        # The state cannot be read: the same as not switched over.
        ("unreadable", ROWS, ROWS, True),
        ("unreadable", ROWS, QUEUE, False),
        ("unreadable", QUEUE, QUEUE, True),
        # Accepted edge (owner and ThinkerOfThoughts, #633): switched, state unreadable, progress 0 stays small.
        ("unreadable", 0, ROWS, False),
    ],
)
def test_scale_rule_matrix(tmp_path: Path, switch: str, progress: int, cut: int, expected: bool) -> None:
    persona_dir, store = _persona(tmp_path, switch, progress)
    try:
        assert writes_engine_scale(persona_dir, store, cut=cut) is expected
    finally:
        store.close()


@pytest.mark.parametrize("switch", ["not_switched", "unreadable"])
def test_unreadable_progress_writes_todays_scale(
    tmp_path: Path, switch: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    persona_dir, store = _persona(tmp_path, switch, QUEUE)
    monkeypatch.setattr(log_throttle, "_last", {})  # each case gets its own loud warning, not the throttled DEBUG line

    def _broken() -> int:
        raise OSError("disk I/O error")

    monkeypatch.setattr(store, "x10_progress", _broken)
    try:
        assert writes_engine_scale(persona_dir, store, cut=ROWS) is False
    finally:
        store.close()
    assert any("could not read the x10 progress" in r.getMessage() for r in caplog.records)
