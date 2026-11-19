# A forced migrate install takes the persona's start lock (#754)

`nell migrate --install-as X --force` replaces the live persona folder. All three migrators (`nellbrain`, `emergence-kit`, `companion-emergence`) move the live folder aside and put the new one in place without the persona's start lock (`bridge.json.lock`). A running bridge holds that lock for its whole life, and every rebuild runs under it, so a bridge or rebuild running during the install keeps writing by path, into the newly installed persona. Severity medium.

## Why

A rebuild that ends during the install saves its emotion state into the new persona, marked as already switched to the new feelings scale, so nothing rebuilds it. `companion-emergence` is worse: it moves the live folder aside and then copies straight into the persona path, so a start during the copy runs in a half-copied folder.

## Goals

- A forced install refuses, with a plain message and a non-zero exit, while the persona's start lock is held. The live folder is left as it was.
- The refusal comes before the migration work, not only at the swap.
- No start, rebuild or bridge can take the lock while the folders are swapped.
- A start that was already waiting for the lock during the swap does not end up holding the backup folder's lock file.

## Non-goals

- Making the two renames atomic together (the issue's separate hardening).
- Locking writers that don't take the start lock today.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | Take the start lock (non-blocking) in one shared install helper, early and across the swap | The lock already guards every bridge and rebuild; one helper for three migrators |
| B | Check `bridge.json` for a running bridge | Racy, and misses a start or rebuild that has the lock but no record yet |
| C | Stop the bridge from the migrate | Surprising side effect; a refusal is what the issue asks for |

Hana approved the design on 2026-11-19.

## Design

New module `brain/migrator/install_swap.py`:

- `PersonaInUseError(RuntimeError)`: the message names the persona, says it is running or starting, says to stop it first (quit the app, or `nell supervisor stop --persona X`), and says the installed persona was not changed.
- `refuse_if_in_use(final_dir)`: when the folder exists, take the start lock and let it go at once; held → `PersonaInUseError`. Each migrator calls it before any migration work when `--force` meets an existing folder.
- `install_over(final_dir, work_dir) -> Path | None`: moves an existing `final_dir` to `<name>.backup-<UTC %Y%m%dT%H%M%S>`, renames `work_dir` into place, and returns the backup path (`None` when there was no folder).
  - **macOS/Linux:** it takes the start lock before the first rename and lets it go after the second. The locked file moves with the old folder into the backup, and the lock is never unlinked (`release_lock` closes the fd only).
  - **Windows:** a folder can't be renamed while a file inside it is open, including our own lock handle. So it takes the lock and lets it go just before the renames. A `PermissionError` from the first rename (a bridge opened the folder in between) becomes the same refusal, with nothing moved.
- `companion-emergence` copies into `<name>.new` (writing its manifest there) and swaps it in with `install_over`, like the other two.
- `nell migrate` catches `PersonaInUseError`, prints it to stderr and exits 1, with no traceback. `nell init --migrate-from` already prints any migrate error as `migration failed: …`.

`brain/bridge/daemon.py::acquire_lock` (macOS/Linux): after it takes the lock, it checks the locked file is still the one at the persona's path (`os.fstat(fd)` against `os.stat(path)`, device and inode). If not (the folder was swapped while it waited), it lets go and locks the file now at the path. Without this, a `nell supervisor run` that waited out the swap would hold the backup's lock while running the new persona, and a later start could lock the new file: two bridges.

Rulings (mine): one backup-name format for all three (`%Y%m%dT%H%M%S`; nothing reads the names, and the tests match the `<name>.backup-` prefix); a refusal at the swap leaves `<name>.new` for the next run to remove, as a failure does today.

## Review (fresh reviewer, 2026-11-19)

No Critical findings. Everything below was fixed in this change, each with a test that failed first:

- **Two `companion-emergence` runs for one name** could install each other's half-built `<name>.new`. Both it and `emergence-kit` (the same race, already on main) now take the migrate lock, as `run_migrate` does.
- **`acquire_lock`'s re-lock was unbounded**, and an unexpected `stat` error escaped with the fd still locked. It now re-locks at most twice and keeps the last lock; an error it can't read keeps the lock, as before the check.
- **A failed second rename** (a file held open in the build, e.g. antivirus) left the persona path empty. `install_over` now puts the live folder back and re-raises.
- **The Windows refusal** now says a file in the folder may be open (Explorer, the indexer, antivirus) and gives the OS error.
- **`nellbrain` printed "Migration complete."** before a swap that could refuse. The report now prints after the swap.
- **`emergence-kit` wrote its manifest after the swap**, by path and outside the lock. It now writes it into the build first.
- **A start between the two renames** got `FileNotFoundError` from `acquire_lock`. Every start path checks the folder exists first, so a missing folder there is mid-swap, and `acquire_lock` now answers "held".

Ruling (mine): **no change** for the reviewer's last case, a migrate swapping between `cmd_start` handing its lock over and the runner taking it. `cmd_start` deletes the stop intent and resets the start back-off while it still holds the lock, so both land in the folder that was live then (now the backup). The runner then locks and runs the new persona, which is what the migrate installed. Nothing is lost that the new persona needed.

## Wiring

No new organ. The helper is called by the three migrators' install paths; `acquire_lock` is shared by every start path.

## Tests (written first, one at a time)

- Each migrator refuses a forced install while the start lock is held: `PersonaInUseError`, the live folder's files untouched, no backup folder.
- `install_over` holds the lock during the renames (POSIX), and the lock is free afterwards.
- On the Windows branch (patched), a `PermissionError` on the first rename is a refusal and nothing moves.
- `companion-emergence` builds the new persona in `<name>.new` and swaps it in.
- `nell migrate` prints the refusal and exits 1, no traceback.
- `acquire_lock` locks the file now at the path when the one it opened was swapped out (POSIX).
