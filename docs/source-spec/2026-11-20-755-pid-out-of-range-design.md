# A pid too large for the OS names no live process (#755)

A pid read from a damaged or hand-edited file can be larger than the OS accepts. `pid_is_alive` then crashes: on POSIX `os.kill` raises `OverflowError`, and on Windows ctypes raises `ArgumentError` passing it to `OpenProcess`. Neither is an `OSError`, so most callers let it escape and the command crashes instead of treating the record as stale. Severity low.

A second path has the same root: `token_of`, used when the record carries a holder token. On Windows the ctypes error is caught there and read as `UNKNOWN`, which `holder_alive` counts as alive, so a damaged lock with a token is never reclaimed. On macOS ctypes silently cuts the number to a C int, so the reading is of whatever process the cut number names.

## Goals

- A pid outside the platform's range is "not alive" in `pid_is_alive` and `GONE` in `token_of`, on every platform, with no OS call.
- Every caller the issue lists stops crashing and treats the record as stale, with no change at any caller.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | Guard both chokepoints with `process_identity.is_pid_number` (on main since Engine 6, #633) | One check, already sized per platform; covers both crashes and the never-reclaimed lock |
| B | Catch `OverflowError` in `pid_is_alive` | Misses the Windows `ArgumentError` and `token_of` |
| C | Validate at each caller | Eight places to keep in step; the issue asks for one |

Hana approved the design on 2026-11-20.

## Design

- `brain/bridge/state_file.py::pid_is_alive`: `if not process_identity.is_pid_number(pid): return False` replaces `if pid <= 0`.
- `brain/process_identity.py::token_of`: `if pid is None or not is_pid_number(pid): return GONE` replaces `pid <= 0`.

## Review (fresh reviewer, 2026-11-20)

No Critical or Important findings. The reviewer checked every caller of the five liveness functions and found one the two guards miss: `daemon.lock_holder` returned the bridge lock's pid unchecked, and the kill helper (`kill._other_live_bridge`) passes it to `psutil.pid_exists`, which raises `OverflowError` on it. So `nell stop --kill` crashed on a hand-edited lock. `lock_holder` now returns None for an out-of-range pid (test first). Deferred, style only: `holder_alive`'s own `pid <= 0` check (a huge pid is caught one level down), and `pid_is_alive`'s lazy import (a module-level one would be safe).

## Wiring

No new organ. Both functions are reached by every liveness reader: `holder_alive` (migrate lock, overlay `.in-use/`, ingest buffer lock, the engine's records), `bridge_liveness`, `respawn_guard`, `stop_watch`, `recovery/cli`, `kill`, `atomic_write`'s temp-file check.

## Tests (written first, one at a time)

- `pid_is_alive` and `token_of` on the first pid past each platform's limit and on `10**20`: not alive / `GONE`; `holder_alive` with a token: False.
- Through the path: a stale migrate lock holding a huge pid is taken over (the issue's first case); a `bridge.json` with a huge pid reads as not running.
