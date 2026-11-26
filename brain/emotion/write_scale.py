"""Which scale a scale writer stores: the engine's 0-10, or today's small values (emotion rework B1 PR 7).

The writers that only pick the scale of a stored value (the note and reach-out values, the making value, the peer memory
with its reflection prompt) cannot use `EmotionEngine.is_live()`: it is false both for a persona not yet switched over,
whose small values the x10 fix multiplies at its next start, and for a switched persona whose state cannot be read right
now, whose small values nothing would ever rescale. ThinkerOfThoughts' ruling on #633 (2026-11-16):

1. The state says the persona is switched over: 0-10.
2. Otherwise (not switched, or the state cannot be read), the x10 fix's progress note in `memories.db` decides, with the cut
   of the x10 step that rescales where the value lands: `NUDGE_SCALE_FIX_PROGRESS_ROWS_DONE` for a value written straight
   to `memories.db`, `NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED` for a value that goes to the pending queue.
3. The progress note cannot be read: today's small scale.

A selection that reads the engine, or that picks what the model is asked for and whether its answer is parsed, keeps
`is_live()` and is not decided here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brain.memory.store import MemoryStore
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)


def writes_engine_scale(persona_dir: Path, store: MemoryStore, *, cut: int) -> bool:
    """True when a scale writer stores the engine's 0-10 scale; False for today's small scale. See the module docstring.

    `cut` is the progress number at which the x10 fix has rescaled where this writer's value lands. Read once per write.

    Accepted edge (#633, 2026-11-16): a switched persona whose state cannot be read and whose `memories.db` reads progress 0
    (rebuilt or new) writes small values, because the x10 fix never runs for a switched persona.
    """
    from brain.emotion import engine_files

    try:
        switched = engine_files.switch_over_state(engine_files.read_state(persona_dir))
    except Exception:  # noqa: BLE001
        warn_throttled(logger, "write_scale.switch_read", "scale writer: could not read the switch-over; using the x10 progress")
        switched = None
    if switched is True:
        return True
    try:
        progress = store.x10_progress()
    except Exception:  # noqa: BLE001
        warn_throttled(logger, "write_scale.progress_read", "scale writer: could not read the x10 progress; writing today's scale")
        return False
    return progress >= cut
