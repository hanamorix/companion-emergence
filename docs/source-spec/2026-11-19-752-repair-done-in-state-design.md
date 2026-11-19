# The self-model repair records "done" in the same save that clears the gap (#752)

Follow-up to #737 (PR #750). Severity low.

## Why

`run_self_model_repair` (`brain/health/self_model_repair.py`) clears the old artifact gap with one state save, then writes "done" to its own marker file (`self_model_repair_state.json`, `on_error="log"`). If the state save succeeds and the marker write fails, nothing records the success. Every later start runs the repair again, and once the reflection tick has computed an honest gap, each re-run clears it, for as long as the marker can't be written.

## Goals

- A repair whose state save succeeded is never run again, whether or not the marker write succeeds.
- No change to any other writer of `self_model_state.json`, so the change doesn't overlap #635 and #656, which edit `reconcile.py`.
- #737's behaviour stays: a failed state save is counted and retried, up to 3 starts.

## Non-goals

- The marker file stays. It is still written, and still read as a second record.
- No change to how the gap is computed, loaded or rendered.

## Anti-goals

- A flag that a later save of the state quietly drops (that would bring the re-run back).
- The repair overwriting a state file it could only partly read.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | A top-level `windowed_peak_cleared` key in `self_model_state.json`, kept by `save()` itself | One file changes for the flag (`state.py`), no writer changes, and a writer added later can't drop it |
| B | A `SelfModelState` field that every writer carries | Changes 6 constructors across the supervisor, `reconcile.py` and `state.py`, overlaps #635 and #656, and needs a guard test so a new writer doesn't drop it |
| C | Accept the edge and close #752 | No code, and the re-run stays possible |
| — | Write the marker first, or clear only gaps that look like the artifact | Rejected: the first is #737 again; the second can't tell an artifact from an honest gap, because `reset_seen` is only set on the engine path |

Hana picked A on 2026-11-19.

## Design

### `brain/self_model/state.py`

- `save(persona_dir, state, *, windowed_peak_cleared: bool = False) -> None`. The payload gains `"windowed_peak_cleared": True` when the argument is true **or** the file on disk already holds `true`. Otherwise the key is left out, so a file never holds `false`.
  - The on-disk value is read with the same `read_json_result(path, default=lambda: None, expect=dict)` call `load_or_recover` uses, so the read adds no new side effect. Every writer already holds `locked(persona_dir)` around its load and save, so the read is consistent.
  - The flag is never cleared once set.
- `windowed_peak_cleared(persona_dir) -> bool`: true only when the file reads cleanly as an object holding `windowed_peak_cleared is True`. A missing, unreadable or non-object file is false.
- `SelfModelState` is unchanged.

### `brain/health/self_model_repair.py`

- `should_run_self_model_repair` is false when `sm_state.windowed_peak_cleared(persona_dir)` is true. Otherwise it checks the marker exactly as today.
- `run_self_model_repair`, under the state lock:
  - **A gap to clear:** `save(..., SelfModelState(current_gap=None, gap_history=...), windowed_peak_cleared=True)`, so the clear and its record are one atomic write.
  - **No gap, and the file loaded cleanly or is missing:** save the loaded state unchanged with `windowed_peak_cleared=True`. A missing file becomes `{"current_gap": null, "gap_history": [], "windowed_peak_cleared": true}`.
  - **No gap, and `load_or_recover` reported a recovered file:** no save, as today, so the repair never overwrites a file it only partly read. The marker is the record there.
- Everything after is unchanged: the marker is written as `{"done": true}`, and the lock-timeout and failure paths stay as #737 left them.

### Older brains

- They read the state with `raw.get(...)`, so the extra key is ignored.
- One that rewrites the file drops the key, and then the marker is the record, as today.

### Cases left open

Each needs a failed marker write plus a second fault. In each, the marker is the only record, as on main:
- a rollback to an older brain that rewrites the state file;
- a state file that is partly unreadable and has no gap: the repair doesn't save it, and the supervisor's next save rewrites it without the key (found in the branch review);
- a state file that is corrupt when a writer saves: `save()` can't read the key, so it doesn't carry it.
- The key is additive and defaults to absent, so the file stays readable by an older brain (CLAUDE.local.md's persisted-state rule).

## Wiring

No new organ. The repair runs at supervisor start (`brain/bridge/supervisor.py`), as before. The flag is read only by `should_run_self_model_repair`.

## Tests (written first)

In `tests/unit/brain/health/test_self_model_repair.py` and `tests/unit/brain/self_model/test_state.py`:

**Fail on main:**
- The clear succeeds and the marker write fails (make `_write_marker` raise, or make the marker path a directory). The next `should_run_self_model_repair` is false, and an honest gap saved afterwards survives the next start's check.
- With no gap and no state file, the repair leaves `windowed_peak_cleared` true in the state.
- After the repair, each kind of later save keeps the flag:
  - a plain `save(persona_dir, SelfModelState(...))`, the way the supervisor and reconcile write;
  - `push_gap` followed by `save`.
- `save` never writes `false`, and `windowed_peak_cleared` reads a missing, corrupt or non-object file as false.

**Pin today's behaviour (`no_base_failure`):**
- A recovered state file with no gap is not overwritten by the repair.
- #737's three tests (fail and retry, give up after 3 starts, an unreadable marker reads as done) pass unchanged.
- `test_repair_noop_when_no_state` is updated: the repair now creates the state file holding only the flag.
