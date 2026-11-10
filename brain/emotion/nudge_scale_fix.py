"""The one-time x10 fix of small feelings, run at a persona's first engine start (Engine 6, #584).

Some writers store a feeling below 1 (a small constant such as 0.12 or 0.15, or a peer's model delta capped under 1) where the
engine works on a 0-10 scale. Before the switch-over those rows are multiplied once, so the first rebuild folds them at their
real size. The types are `dev_constants.NUDGE_SCALE_FIX_MEMORY_TYPES`, the factor, threshold and progress
numbers are `dev_constants.NUDGE_SCALE_FIX_*`, and the work is in two places that own their files: `MemoryStore.scale_up_small_nudge_rows`
for the rows of `memories.db` (the store is the one writer of that file) and `PendingQueue.scale_up_small_making_candidates` for
the `making` candidates still waiting in the pending queue (the queue is the one writer of its files; a waiting candidate is a
memory the gate has not promoted yet, so it is multiplied too, Planning B21 Q10c). This module only decides WHEN: never on a
persona that is switched over (its state holds `switch_over_at`), and never with no `memories.db` (nothing to multiply and
nowhere to record progress; skipped deliberately and logged). A state that exists but cannot be read cannot say whether the
persona was switched, so it does not stop the fix: the two steps below read the database's progress number themselves and skip
what is done (Planning B31 SW4, owner S79). The database runs first, then the queue. It goes, with its
constants and the two methods, when #388 removes it.

Negative feelings (a peer's model delta keeps its sign) are held to 0.0 and then multiplied, in rows and queue entries alike
(owner 2026-10-06: negatives are converted to 0; Planning B29 C2 item 1).

No row and no queue entry carries a marker (owner S77). A finished switch-over is known from `switch_over_at` in the state
(the guard above). A half-finished x10 fix is known from `memories.db`'s `PRAGMA user_version`, the fix's progress note (owner
S78: "yes use the version number for a half finished repair"), read and written only through `MemoryStore.x10_progress` and
`set_x10_progress`: 0 nothing done; 1 (`NUDGE_SCALE_FIX_PROGRESS_ROWS_DONE`) the rows are multiplied, set in the rows' own
transaction; 2 (`NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED`) the queue step has started, set before any queue file is rewritten
(so 2 does not mean the files are done). A start-up that fails after this fix is run again at the next start from where the
number says: the rows are not multiplied twice, and a queue step that failed before setting 2 starts again from the untouched
files (`PendingQueue.scale_up_small_making_candidates`). Nothing here repeats what the database's own transaction and the
shared atomic writer already make all-or-nothing (owner S84).
Dev personas made by the marker build that never left the branch (rows marked, number 0) would be multiplied again: they are
test data to recreate.

Accepted edges (the commit that adds this file names them):
- Owner S77 (P4): a row or queue entry written by something else while the app is down between the steps of a start-up is
  missed (a row written after the database step, an entry appended after the queue step read its files).
- Owner S84 (P4): the number 2 is set before the queue files are rewritten, so a crash or a failed write after it leaves that
  file's selected entries at the old, faint scale, never multiplied twice (the other order would multiply a rewritten file
  again at the retry). A failed write is logged and start-up goes on; it is not retried.
- E5 (owner 2026-10-21): a row whose strongest value is below 1 only because it faded there, not because it was written small,
  is multiplied once. One odd day for the kindled, then it sinks back.
- E6: a feeling the old decay pass deleted at 0.01 stays deleted; multiplying the rest does not bring it back.
- E7.3g: `memories.db` restored alone under a surviving state keeps its unfixed rows: the fix does not run again once the
  state holds `switch_over_at`.
- Re-run after a lost state (Planning B25 SW2, edge (a)): if every copy of the state file is lost or unreadable, or the live
  state is corrupted right after the first rebuild while `.bak1` is still the pre-switch state (edge (b)), the persona reads as
  not switched and the start-up goes on to the first rebuild, which records a later `switch_over_at`. This fix multiplies nothing
  on that re-run: its progress note in `memories.db` is already 2.
- The fix runs outside the engine lock, before the first rebuild takes it. A writer that has not been updated (another process
  on old code) can write a small row, or queue a small candidate, after the fix and before the state records the switch-over.
  If this start goes on to record it, that row stays small: the fix has no cutoff date and no second pass after the switch-over.
  (If the start fails before the number reaches 1 for the rows, or before it reaches 2 for the queue, the next start catches
  it; after that it does not.)
- A candidate that an old-code process has already claimed into memory (the gate's batch, read before the fix and promoted after
  it) is promoted at its old size, the same window. An old-code process that finishes its batch while the queue half runs can
  delete the processing file (`ClaimedBatch.close` takes no queue lock) after the fix has read it; the fix's rewrite then puts
  the file back with entries that process already promoted, and the next claim offers them again. New code cannot be in that
  state: a process that runs a gate has already passed `ensure`, so the state holds `switch_over_at` and this fix does not run.
- The queue half rewrites a file only when something in it was multiplied, and writes every other line back as the shared
  reader yielded it: the bytes of the line without its terminator (a blank line, a carriage return before the newline and a
  leading byte-order mark are not kept, and a torn last line gets its newline; the app writes none of them).
- A peer row can hold a NEGATIVE value (a model delta keeps its sign). It passes "below 1" and is held to 0.0 and then
  multiplied with the rest of its row (above), so that member reads 0.0 as the engine would hold it, and the row's positive
  members reach the 0-10 scale. A row or entry with no feelings is still multiplied in its peak and score (Planning B27 Q15: a
  decayed row keeps its peak). A row or entry whose values cannot be multiplied safely (NaN, a product that is not a finite
  number such as 1e308, text, a metadata or emotions that is not a mapping) is left unchanged, logged by its id, and not counted.
- Stored negatives outside this fix's selector stay as they are: the engine holds them at 0.0 when read, the writers of them
  are being fixed, and a one-time pass over every stored row was the rejected alternative.
- The queue's lock (`file_lock`) waits without a timeout, so a stuck holder delays start-up (existing behaviour of every queue
  writer, unchanged).
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

    None when the persona is switched (its state holds `switch_over_at`; an unreadable state is not a reason to skip, see the
    module docstring) and when there is no `memories.db`: with no database there is nothing to multiply and nowhere to
    record progress, so BOTH halves are skipped deliberately and the skip is logged (P9b; practically unreachable, since a queue
    entry is written through a store that creates the database). Otherwise the rows are done (0 when an earlier start did them),
    then the queue under the same open store, which logs its own count.

    The database store is opened with its default integrity check (this is a write, it runs once in a persona's life, and a
    damaged file must not be written into): a failing check raises `BrainIntegrityError` and nothing is written. Any other error
    of the store method (the transaction rolled back, the progress number as it was) or of the queue method (a read fault or a
    failure to set the number: raised before any file is written) also goes out to the caller, which stops the start-up. A
    failed write of a queue file is not one of them: the queue method logs it and goes on.
    """
    if engine_files.switch_over_state(engine_files.read_state(persona_dir)) is True:
        return None  # switched; a state that cannot be read (None) goes on: the progress number below decides (Planning B31 SW4)
    database = persona_dir / "memories.db"
    if file_ops.stat_path(database) is None:
        logger.warning("x10 fix: there is no memories.db, so nothing to multiply and nowhere to record progress; both halves skipped")
        return None
    store = MemoryStore(database)
    try:
        rows = store.scale_up_small_nudge_rows()
        PendingQueue(persona_dir).scale_up_small_making_candidates(store)
        return rows
    finally:
        store.close()
