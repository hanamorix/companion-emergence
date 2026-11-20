# The integrity walker closes its works.db connection (#779)

## Intent

`brain/health/walker.py` checks the works database with `with sqlite3.connect(str(db_path)) as conn:`. In `sqlite3` that context manager only commits or rolls back. It never closes the connection, so every walk leaves a handle on the file until the cycle collector runs. On Windows a held handle blocks renaming, replacing or deleting the file (`WinError 32`), and a walk that has just reported the file as corrupt keeps that very file open. #772 fixed the same pattern in `WorksStore`.

## Goals

- Every connection `walk_persona` opens is closed when the walk returns, for a valid works database and for a corrupt one.
- The anomalies a walk reports don't change.

## Non-goals

- The other databases' checks. They go through their stores' constructors and `close()`, and the new test covers them too.
- Which file the walker checks, `data/works.db` on main and `works.db` after #768.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | `with closing(sqlite3.connect(...)) as conn:` around the existing check | One line. The same form as #772 in the store |
| B | Stack on #768 | Adds a rebase after #768's squash-merge |
| C | Wait for #768 to merge | Only saves a one-line conflict on the neighbouring line |

Hana approved A on 2026-11-20.

## Design

- **The fix.** In the walker's works branch, wrap the connection in `contextlib.closing`. The `PRAGMA integrity_check` stays inside the `with`, so a corrupt file that raises `sqlite3.DatabaseError` is still closed on the way out.
- **The test,** in `tests/unit/brain/health/test_walker_closes_connections.py`:
  - It records every `sqlite3.connect` made during `walk_persona`.
  - It asserts that each recorded connection is closed afterwards, for a valid and for a corrupt works database.
  - It puts the works database at both `works.db` and `data/works.db`, so it holds on main and after #768 unchanged.
  - It checks every connection the walk opens, not only the works one.

## Wiring

No new organ. The walker's reports reach the same consumers as before. The change only ends the connection's life sooner.

## Sequencing

#768 edits the comment on the line above the changed `with`, so whichever merges second resolves a one-line conflict. This PR stays a draft until #768 merges and is then brought up to date with main.
