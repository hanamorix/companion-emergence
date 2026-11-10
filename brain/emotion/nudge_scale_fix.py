"""The one-time x10 fix of small feelings, run at a persona's first engine start (Engine 6, #584).

Some writers store a feeling below 1 (a small constant such as 0.12 or 0.15, or a peer's model delta capped under 1) where the
engine works on a 0-10 scale. Before the switch-over those rows are multiplied once, so the first rebuild folds them at their
real size. The types are `dev_constants.NUDGE_SCALE_FIX_MEMORY_TYPES`, the factor, threshold and marker are
`dev_constants.NUDGE_SCALE_FIX_*`, and the SQL is `MemoryStore.scale_up_small_nudge_rows` (the store is the one writer of
`memories.db`). This module only decides WHEN: never on a persona whose switch exists, never when the switch cannot be
inspected (an unknown counts as not switched, and nothing irreversible runs on an unknown), never with no `memories.db`. It
goes, with its constants and the store method, when #388 removes it.

Accepted edges (the commit that adds this file names them):
- E5 (owner 2026-10-21): a row whose strongest value is below 1 only because it faded there, not because it was written small,
  is multiplied once. One odd day for the kindled, then it sinks back.
- E6: a feeling the old decay pass deleted at 0.01 stays deleted; multiplying the rest does not bring it back.
- E7.3g: `memories.db` restored alone under a surviving switch keeps its unfixed rows: the fix does not run again once the
  switch exists.
- The fix runs outside the engine lock, before the first rebuild takes it. A writer that has not been updated (another process
  on old code) can write a small row after the fix and before the switch exists. The marker is already set, so that row is not
  multiplied. The fix assumes there is no such writer: it has no cutoff date and no second pass.
- A `making` still waiting in the pending-candidate queue (`pending_candidates.jsonl`, `brain/memory/pending.py`) when this runs
  is not in `memories.db` yet, so it is not multiplied; the consolidation gate can promote it later at its old small size. The
  gate can hold a candidate for a long time, so how many sit in a real queue depends on its backlog. `kindled_peer` rows are
  written straight to the database and `initiate_outbound` is a gate-bypass type, so neither has this gap. A second writer of the
  queue file for one one-time pass is not worth it (the queue file has one writer).
- A peer row can hold a NEGATIVE value (a model delta keeps its sign). It passes "below 1" but x10 would take it out of 0-10, so
  the store method leaves such a row alone, with a warning naming its id, and counts nothing for it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brain.emotion import engine_files
from brain.memory.store import MemoryStore
from brain.utils import file_ops

logger = logging.getLogger(__name__)


def scale_up_small_feelings(persona_dir: Path) -> int | None:
    """Run the fix for one persona: the number of rows multiplied, or None when it was not run (see the module docstring).

    The store is opened with its default integrity check (this is a write, it runs once in a persona's life, and a damaged file
    must not be written into): a failing check raises `BrainIntegrityError` and nothing is written. Any other error of the
    store method (the transaction rolled back, the marker not set) also goes out to the caller.
    """
    if engine_files.switch_state(persona_dir) is not False:
        return None
    database = persona_dir / "memories.db"
    if file_ops.stat_path(database) is None:
        return None
    store = MemoryStore(database)
    try:
        return store.scale_up_small_nudge_rows()
    finally:
        store.close()
