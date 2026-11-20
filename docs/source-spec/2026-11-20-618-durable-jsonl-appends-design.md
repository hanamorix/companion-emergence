# JSONL appends reach the disk by default, and the NUL runs already in a persona are repaired once (#618)

Severity high. Filed by ThinkerOfThoughts from a read-only copy of a production persona.

## Why

`append_jsonl` (`brain/health/json_io.py`) writes and flushes, but calls `os.fsync` only when the caller passes
`fsync=True`. When the machine stops uncleanly (power cut, forced VM stop, kernel panic) right after an append, the
file can come back longer but with that record's bytes still zero: a run of NUL bytes one record long. #618 found 12
such runs in six files of one persona, at nine whole-persona stops.

Since #434, `append_jsonl` starts a new line when the file's last byte is not a newline, so new damage no longer
swallows the record written after the restart. But the record being written is still lost, and the damage already on
disk still fuses each NUL run to the next valid record, which the reader rejects and warns about on every read.

## Goals

- Every `append_jsonl` call reaches the disk before it returns, unless its caller opts out with a stated reason.
- The NUL runs already in a persona's JSONL files are removed once, so the valid record each one swallowed is read
  again, with a copy of every changed file kept beside it.
- Both of #618's acceptance tests, written first and failing on main.

## Non-goals

- The reader stays as it is: a damaged line is still warned about, so new damage stays visible (#618's own ask).
  #646 (ToT's) covers the reader's other abort case.
- `write_jsonl`, `save_with_backup_text` and the other whole-file writers already default to `fsync=True`.
- The zeroed records themselves were never written and can't be recovered.

## Anti-goals

- A repair that deletes or rewrites anything beyond the NUL bytes, or that changes a file without keeping a copy.
- A user-facing command or setting for the repair (CLAUDE.local.md: the surface is install, name, talk).

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | `fsync=True` becomes `append_jsonl`'s default; a caller that can't afford it passes `fsync=False` with its reason on the line above, and a guard test lists those callers | 36 of the 37 default call sites become durable (the emotion engine keeps its own no-fsync choice, below), and any new one is durable unless it opts out on purpose |
| B | Pass `fsync=True` at the six damaged streams only | Smallest diff, but the other default callers stay exposed, and every new log has to remember to opt in |

Hana picked A on 2026-11-20.

### Cost (measured 2026-11-20, this Mac, APFS)

300 appends of a 200-byte record through `append_jsonl`: median 0.062 ms without fsync, 0.074 ms with it (p95 0.075
vs 0.086 ms). The per-call cost is small enough that no caller needs to opt out for speed. Linux fsync is slower (a
few ms on a VM disk), which is still small next to any caller's own work.

### Callers that stay without fsync

- `brain/memory/recall_diagnostics.py:191` already passes `fsync=False`, with its reason in a comment (the recall path's
  never-wait contract, diagnostic data). Kept.
- `brain/emotion/engine_files.py::_append_line` (the engine's event queue and daily change logs) opts out. The
  engine's own design chose "no fsync per append" (its section 2.9), and
  `tests/unit/brain/emotion/test_engine_queue.py::test_the_append_asks_append_jsonl_for_a_held_lock_and_logging_with_no_fsync`
  pins it. The change passes `fsync=False` there explicitly, with that reason, instead of changing the engine's
  behaviour, and that test's expected keywords gain `"fsync": False`. It's ToT's lane, and ToT's #642 derives the engine's
  lock wait from measured hold times, so the draft PR asks ToT whether the opt-out should stay. Found while writing the
  plan, after the design's first draft.
- No other caller opts out. One writes in bursts and is named here so a reviewer can object: `_archive_preimage` in
  `brain/engines/dedup_sweep.py` and `brain/engines/consolidation.py`, one append per merged or dropped memory in a
  sweep. These are the undo records for memory changes, so they're the last place to skip it.

## Design

### `brain/health/json_io.py`

- `append_jsonl(..., fsync: bool = True, ...)`. The docstring says why: an unclean stop can leave an appended record as
  zeros, and the opt-out needs a reason.
- No other change to the function.

### The opt-out guard

A test parses every `append_jsonl(...)` call in `brain/` and fails when one passes `fsync=False` without a comment
starting `# fsync off:` on the line above or the same line. `recall_diagnostics.py` gets that comment.

### One-time repair: `brain/health/jsonl_nul_repair.py`

Shaped like `brain/health/soul_candidate_repair.py`, and run from the supervisor's startup block beside it, in its own
`try/except`.

- `should_run_jsonl_nul_repair(persona_dir) -> bool`: false once `<persona_dir>/jsonl_nul_repair_state.json` holds
  `{"done": true}`.
- `run_jsonl_nul_repair(persona_dir) -> RepairReport`:
  - Walks `persona_dir` for files named `*.jsonl` or `*.jsonl.<digits>` (rotated, uncompressed). `.gz` files are skipped,
    since NUL is normal in gzip.
  - JSON text never contains a raw NUL byte (JSON escapes it), so any raw NUL in these files is damage.
  - For a file that holds a NUL byte (found by a chunked scan, not a whole-file load): under `file_lock(path)`, so a
    concurrent append waits, copy it to `<name>.pre-618` (or `.pre-618.<n>` if that exists), then rewrite it
    atomically with each NUL run replaced by nothing when it starts a line, or by one newline when it follows other
    bytes on its line. Written in chunks to a temp file and swapped in with the repo's atomic helpers.
  - Records per file: runs removed, bytes removed, backup path. Logs one info line per repaired file and one summary.
  - Writes the state file with `{"done": true, "repaired": {...}}` when every file was scanned; a file it couldn't
    read or lock leaves the state unwritten, so the next start tries again.
- Idempotent: a second run, with or without the state file, finds no NUL and changes nothing.

### Older brains

The state file is new and only this module reads it. The backups end in `.pre-618`, which no reader picks up.

## Wiring

- **Reads from:** every JSONL file in the persona folder (the repair), once.
- **Feeds into:** every reader of those files (`iter_jsonl`, `read_jsonl`, `read_jsonl_tail`, `parse_jsonl` and their
  callers) sees the recovered records and stops warning about the fused lines. The default change feeds every JSONL
  stream's durability.
- No new organ; the repair is a startup migration like the soul-candidate repair.

## Tests (written first)

In `tests/unit/brain/health/`:

- **Unclean stop (fails on main):** a fake filesystem layer around `append_jsonl` that, on a simulated crash, keeps the
  file's length but turns every byte written since the last `os.fsync` on that file into NUL. After a default
  `append_jsonl` and a crash, the record reads back. On main the record is NUL and the next append's line is the one
  that survives.
- **Default is durable (fails on main):** `append_jsonl` with no `fsync` argument calls `os.fsync` on the file's
  descriptor after the write.
- **Opt-out guard:** fails on an `fsync=False` with no `# fsync off:` comment; passes on the tree.
- **Repair (fails on main: the module doesn't exist):** a synthetic file with a NUL run at a line start fused to a
  valid record, one after a torn record, and one at the end of the file. After the repair every valid record is on its
  own line and parses; the `.pre-618` copy holds the original bytes; a second run changes nothing (same bytes, no new
  backup); and `iter_jsonl` then logs no warning for the file. Rotated `.jsonl.1` files are repaired; `.gz` files are
  untouched.
- **Through the startup path:** the supervisor's startup runs the repair on a persona with a damaged file, and writes
  the state file.
- **Retry:** a file that can't be read leaves the state unwritten, and the next run repairs it.

## Deferred

- **`F_FULLFSYNC` on macOS.** Plain `fsync` on macOS hands the data to the drive but doesn't flush the drive's own
  cache, so a power cut can still lose a just-written record there. The repo uses plain `fsync` everywhere; changing
  that is a separate decision. Recorded in the deferred-items memory and the next brainstorm.
- **What caused the stops (#618 item 3).** They were on the machine the evidence came from, before its logs; nothing
  in this change can find them. Left on #618 for its filer.
