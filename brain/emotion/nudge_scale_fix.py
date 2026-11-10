"""The one-time x10 fix of small feelings, run at a persona's first engine start (Engine 6, #584).

Some writers store a feeling below 1 (a small constant such as 0.12 or 0.15, or a peer's model delta capped under 1) where the
engine works on a 0-10 scale. Before the switch-over those rows are multiplied once, so the first rebuild folds them at their
real size. The types are `dev_constants.NUDGE_SCALE_FIX_MEMORY_TYPES`, the factor, threshold and marker key are
`dev_constants.NUDGE_SCALE_FIX_*`, and the work is in two places that own their files: `MemoryStore.scale_up_small_nudge_rows`
for the rows of `memories.db` (the store is the one writer of that file) and `PendingQueue.scale_up_small_making_candidates` for
the `making` candidates still waiting in the pending queue (the queue is the one writer of its files; a waiting candidate is a
memory the gate has not promoted yet, so it is multiplied too, Planning B21 Q10c). This module only decides WHEN: never on a
persona whose switch exists, never when the switch cannot be inspected (an unknown counts as not switched, and nothing
irreversible runs on an unknown), the database half never with no `memories.db`. The database runs first, then the queue. It
goes, with its constants and the two methods, when #388 removes it.

Each multiplied row and each multiplied candidate carries the marker key in its `metadata`, and both halves select only what is
not marked. A start-up that fails after this fix is therefore run again from the top at the next start: nothing is multiplied
twice and what was written in between is caught.

Accepted edges (the commit that adds this file names them):
- E5 (owner 2026-10-21): a row whose strongest value is below 1 only because it faded there, not because it was written small,
  is multiplied once. One odd day for the kindled, then it sinks back.
- E6: a feeling the old decay pass deleted at 0.01 stays deleted; multiplying the rest does not bring it back.
- E7.3g: `memories.db` restored alone under a surviving switch keeps its unfixed rows: the fix does not run again once the
  switch exists.
- The fix runs outside the engine lock, before the first rebuild takes it. A writer that has not been updated (another process
  on old code) can write a small row, or queue a small candidate, after the fix and before the switch exists. If this start goes
  on to write the switch, that row stays small: the fix has no cutoff date and no second pass after the switch. (If the start
  fails before it writes the switch, the next start catches it, because only unmarked rows and candidates are selected.)
- A candidate that an old-code process has already claimed into memory (the gate's batch, read before the fix and promoted after
  it) is promoted at its old size, the same window. An old-code process that finishes its batch while the queue half runs can
  delete the processing file (`ClaimedBatch.close` takes no queue lock) after the fix has read it; the fix's replace then puts
  the file back with entries that process already promoted, and the next claim offers them again. New code cannot be in that
  state: a process that runs a gate has already passed `ensure`, so the switch exists and this fix does not run.
- The queue half rewrites a file only when it multiplied something, and writes every other line back as the shared reader
  yielded it: the bytes of the line without its terminator (a blank line, a carriage return before the newline and a leading
  byte-order mark are not kept, and a torn last line gets its newline; the app writes none of them).
- A peer row can hold a NEGATIVE value (a model delta keeps its sign). It passes "below 1", and the whole row is multiplied,
  the negative member too (Planning B18 Q11b, derived from the owner's R13, one threshold for all three types). The engine
  holds a stored negative at 0.0 before any fold, so that member still reads as 0.0, and the row's positive members reach the
  0-10 scale. A row with no feelings has nothing to multiply and is not counted. A row whose product would not be a finite
  number (a damaged value such as 1e308) is skipped with a warning and not counted.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brain.emotion import engine_files
from brain.memory.pending import PendingQueue
from brain.memory.store import MemoryStore
from brain.utils import file_ops

logger = logging.getLogger(__name__)


def scale_up_small_feelings(persona_dir: Path) -> int | None:
    """Run the fix for one persona, the database half and then the queue half: the number of DATABASE rows multiplied, or None when it was not run.

    None when the switch is not known to be missing (see the module docstring) and, for the database half only, when there is no
    `memories.db`; the queue half then still runs when the switch is known missing (it creates nothing when no queue file is
    there), so the return is None for a persona with no database whatever the queue held. The queue half logs its own count.

    The database store is opened with its default integrity check (this is a write, it runs once in a persona's life, and a
    damaged file must not be written into): a failing check raises `BrainIntegrityError` and nothing is written. Any other error
    of the store method (the transaction rolled back, nothing marked) or of the queue method (a read or write fault, the file as it
    was) also goes out to the caller, which stops the start-up.
    """
    if engine_files.switch_state(persona_dir) is not False:
        return None
    rows = _scale_up_database(persona_dir)
    PendingQueue(persona_dir).scale_up_small_making_candidates()
    return rows


def _scale_up_database(persona_dir: Path) -> int | None:
    database = persona_dir / "memories.db"
    if file_ops.stat_path(database) is None:
        return None
    store = MemoryStore(database)
    try:
        return store.scale_up_small_nudge_rows()
    finally:
        store.close()
