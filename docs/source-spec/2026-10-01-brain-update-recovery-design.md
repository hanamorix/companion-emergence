# Recovering an interrupted in-app brain update (#335)

Follow-up to the brain-update overlay (#286, `2026-09-26-brain-update-overlay-design.md`).
Builds on #336 (one module-level bridge restart).

## 1. Problem

The in-app brain update is a chain the frontend drives (`useBrainUpdate.apply`): Rust's
`apply_brain_update` installs the overlay, the bridge restarts onto it, the app checks the
new commit actually loaded, and an unhealthy bridge triggers rollback, then the release brain.
Closing the window mid-chain stops it, and nothing finishes it later.

Verified on `main` @ 7365c0a9:

- **Install itself is safe.** `nell update` stages, smoke-imports and swaps atomically
  (`brain/update/install.py`); a kill before the swap changes nothing.
- **After the swap, the next launch can land in two bad states:**
  - **A, unfinished:** the window closed before the restart, so the old bridge is still
    running while `current.json` names the new overlay. The app works, silently on the old
    brain, until something restarts the bridge onto an unproven build.
  - **C, stranded:** the new build can't start. `ensureBridgeRunning` fails and
    `BridgeErrorScreen` (`App.tsx`) offers Retry / Open anyway / Run setup. "Use the release
    brain" lives in `BrainUpdateRow`, which only shows when the *running* bridge reports an
    overlay through `/health`. With the bridge down it is hidden. The only ways out are
    `KINDLED_NO_OVERLAY=1` or `nell update --revert` from a terminal.

## 2. Decisions (Hana, 2026-10-01)

1. **Start fails at launch:** if the active overlay never proved itself (unconfirmed), roll
   back automatically and retry, with a notice. If it was confirmed (it has run before), ask:
   the error screen offers **Use the release brain**. A confirmed build that fails to start
   may be failing for an unrelated reason, so don't silently drop it.
2. **Unfinished update at launch (A):** finish it. Restart onto the active overlay, as the
   version handshake already does for a stale bridge, then the normal loaded-check.
3. **Placement:** Python owns the flag, Rust adds two thin commands, and the frontend's launch
   path makes the decisions, reusing the existing restart and rollback chain. Rejected: Rust
   driving it inside `ensure_bridge_running` (the graceful restart lives in TypeScript, so the
   chain would be split or duplicated); boot counting in the overlay hook (the hook runs at
   *every* interpreter start, including plain `nell` commands, and must stay tiny and never
   raise).

## 3. Design

### 3.1 State: `confirmed` on the active entry (Python)

- `current.json`'s entry gains `"confirmed": bool`. **Additive** (#286 §7): readers treat a
  missing field as `true`, so overlays installed before this change, and entries an older
  brain rewrites, keep working. `_write_state` already writes the entry dicts it read back
  whole, so older brains preserve the field.
- The entry also gains `"undo": "rollback" | "revert"`: how to back this build out if it
  never proves itself. `"rollback"` when an overlay was active at install time;
  `"revert"` when the release brain was. Then `previous` may be an older overlay the user
  deliberately left, which a rollback would bring back. This is the same distinction
  `useBrainUpdate.apply` makes today from `had_active`, recorded where the launch path can
  read it. A missing `undo` reads as `"revert"`, the floor.
- `install.install_overlay` activates with `{**entry, "confirmed": False, "undo": ...}`.
  Every activation by install is unconfirmed, including the instant reuse of an existing
  folder. The stamp (`stamp.json`) carries neither field: they are state, not build identity.
- `overlay.confirm(root, commit)`: under the lock, set `active["confirmed"] = True` only when
  `active.commit == commit`; otherwise a no-op. The active entry changed underneath, which is
  harmless.
- `rollback` / `revert` are unchanged: a restored `previous` keeps its own flag.
- CLI: `nell update --confirm COMMIT`, in the existing mutually exclusive `update_mode` group;
  it prints the state like `--rollback`. `--status` already prints the whole entry, so the
  flag shows there with no change.

### 3.2 Rust commands (`app/src-tauri/src/brain_update.rs`)

Both run on the release brain through `run_nell`'s `floor()`, so they work while the overlay
brain is broken.

- `brain_overlay_status() -> { active_commit: Option<String>, confirmed: bool, undo: String }`:
  wraps `nell update --status`. Reuses `active_commit_from_status` (bundle check included);
  `confirmed` is the entry's flag, `true` when missing or when there is no active overlay;
  `undo` is the entry's field, `"revert"` when missing.
- `confirm_brain_update(commit: String)`: runs `nell update --confirm <commit>`.
- Both are registered in `generate_handler!`, with camelCase-arg wrappers in `appConfig.ts`.

### 3.3 Frontend

- **`useRestartBridge.ts`:** export the module-level restart as a plain
  `restartBridge(persona): Promise<boolean>` (the #336 store); the hook's `restart` calls it.
  Launch code runs before `Ready` mounts, so it can't use the hook.
- **`brainRecovery.ts` (new):** `recoverUnhealthyBrain(persona, undo, reason)`, the rollback
  chain moved out of `useBrainUpdate.apply`: `undo` (rollback or revert), restart, then
  revert and restart again if still unhealthy (skipping the second revert when `undo` was
  already a revert). It returns `"rolled_back" | "release" | "failed"`. `apply` passes
  `had_active ? "rollback" : "revert"`; the launch path passes `status.undo`. `apply` and the launch path both call it, so the
  chain exists once (#315's lesson).
- **`useBrainUpdate.apply`:** after its existing check (`running.commit === applied.commit`),
  call `confirmBrainUpdate(applied.commit)`. A failed confirm is logged, not shown: the next
  launch re-confirms a healthy loaded build.
- **`App.startPersona`** runs the launch table (§4) around `ensureBridgeRunning` and
  `ensureBridgeCurrent`, and sets a `recoveryNotice` shown in the version-mismatch notice spot.
- **`BridgeErrorScreen`:** when `brain_overlay_status` reports an active overlay, it shows
  **Use the release brain**, which runs revert and then `startPersona` again.

## 4. The launch table (`startPersona`)

`status` = `brain_overlay_status()`; `loaded` = `/health.overlay.commit` of the running bridge.

| Situation | Action |
|---|---|
| No active overlay | Unchanged. |
| Bridge healthy, `loaded == active`, unconfirmed | `confirm_brain_update(active)`, silently. |
| Bridge healthy, `loaded != active` (A) | `restartBridge`; on success with `loaded == active`, confirm. Otherwise, if the active overlay is **unconfirmed**: `recoverUnhealthyBrain(status.undo, …)` and a notice. If it is **confirmed** (decision 1: never dropped automatically): a failed restart goes to `BridgeErrorScreen` with **Use the release brain**; a healthy bridge on another build is left running. |
| Start fails, active **unconfirmed** (C) | `recoverUnhealthyBrain(status.undo, "unconfirmed overlay failed to start at launch")`, retry the start, notice. |
| Start fails, active **confirmed** | `BridgeErrorScreen` with **Use the release brain**. |

**Stopping rule.** `rollback` makes `previous` active and clears `previous`; with no
`previous` it becomes `revert`; `revert` leaves no active overlay. Each automatic step therefore removes a build from the chain:
at most rollback, then revert, then the release brain, where there is no overlay and the
recovery stops. A failure after that is an ordinary bridge problem and gets today's
`BridgeErrorScreen`. Recovery is entered only while an unconfirmed overlay is active, so it
cannot cycle across launches either: after a rollback the bad build is no longer active.

**Notices** (dismissible, same position and style as the version-mismatch notice):
- rolled back: "The brain update didn't start, so the previous brain is back."
- release: "The brain update didn't start, so Companion Emergence is on the release brain."

The reason reaches `launch-failures.log` through `rollback_brain`'s existing `log_event`.

## 5. Errors

The release brain stays the floor.

| Failure | Behaviour |
|---|---|
| `brain_overlay_status` fails or takes longer than 5 s (nell won't run on the release brain) | Treat as "no active overlay": today's behaviour. The release brain itself is broken, which is out of scope. |
| `confirm_brain_update` fails | Log it; the build stays unconfirmed and the next healthy launch confirms it. Worst case, a later start failure auto-rolls back a build that had in fact worked: reversible, since re-apply is instant. |
| Recovery's rollback/revert fails | Same as `apply`'s chain today: "Couldn't switch back to the release brain: …", then `BridgeErrorScreen` with the button. |
| Overlay lock busy at launch (an update still running in another process) | The step fails like any other; the user gets `BridgeErrorScreen`, and Retry works once the lock is free. |

## 6. Wiring

- **Reads from:** `current.json` (through `nell update --status`), `/health.overlay` (the
  loaded commit), `ensureBridgeRunning`'s outcome.
- **Feeds into:** `current.json`'s `confirmed` (read by the next launch and by `--status`);
  the active/previous pointers (through rollback/revert, read by the overlay hook at every
  interpreter start); `launch-failures.log`; the launch notice.
- No emotional or memory loop is involved: this is install plumbing, like #286. The organ
  Definition-of-Done doesn't apply.

## 7. Testing (one per interruption point)

| Interruption point | Test |
|---|---|
| During install | Existing atomic-swap tests; new: the swap leaves `active.confirmed == False` with `undo` = `"rollback"` when an overlay was active and `"revert"` when none was; the stamp has neither field. |
| After swap, before restart (A) | vitest, `startPersona`: healthy + `loaded != active` → restart → confirm; restart fails → `recoverUnhealthyBrain` + notice. |
| After shutdown, before health (C) | vitest: start fails + unconfirmed → rollback → retry → notice; restored entry also unconfirmed and failing → revert → retry → release notice. |
| During the rollback chain | vitest: launch from each state rollback/revert can leave (previous active and unconfirmed; no active), with the same rows; `undo: "revert"` never reaches a reverted-away `previous`. |
| Confirmed overlay fails to start | vitest: `BridgeErrorScreen` shows **Use the release brain** → revert → `startPersona`. Without an active overlay the button is absent. |
| State plumbing | pytest: `confirm` flips only the matching commit; a missing field reads as confirmed; rollback keeps the restored entry's flag; `--confirm` CLI. cargo test: status parse, missing `confirmed` → `true`, missing `undo` → `"revert"`, foreign bundle → no active. |
| Shared chain | vitest: `useBrainUpdate.apply`'s existing rollback tests pass unchanged against `recoverUnhealthyBrain`, plus confirm-after-loaded. |

No live run: the first real in-app run is #309 at the 0.0.43 release review, which should
include closing the window once mid-update.

## 8. Deferred

- **Remembering failed commits.** After an automatic rollback, Check can offer the same bad
  commit again. Re-applying it fails and rolls back again: no stuck state, but no memory of
  the failure either. Same property as today's in-app rollback. Revisit if #309 or users hit
  it.
- **CLI-only users.** `nell update` from a terminal also writes `confirmed: false`; only the
  app confirms or recovers. A CLI-only user keeps today's manual escapes.
