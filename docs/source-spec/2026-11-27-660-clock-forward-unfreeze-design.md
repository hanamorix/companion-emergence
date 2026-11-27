# Unfreeze day closes after a corrected clock (#660)

Hana approved this design on 2026-11-27. The build waits for #803 (see Sequencing).

## Intent

If the system clock is set forward and then corrected, the emotion engine stops closing days until real time catches up with the wrong date. This change makes day closes resume from the corrected date.

What happens today:

1. **The clock jumps forward**, say to next June.
   - The drain's `close_days` reads that as the app having been off for months.
   - It closes the open day, skips the "missing" days without aging the weights across them, and sets `active_day` to next June.
   - The save writes `saved_at` at next June too.
   - If the wrong clock keeps running past midnights, each drain closes those future days as well, and each close appends a snapshot dated on a future day.
2. **The clock is corrected.** Each pass still runs at next June:
   - `EmotionEngine._run_apply` (`brain/emotion/engine.py`) holds its clock to the stored `saved_at` (`now = max(now, state.saved_at)`).
   - `close_days` finds the same day as `active_day` and does nothing.
   - The drain's quick skip (`_can_skip`: `day_of(now) <= state.active_day`) lets most passes skip entirely.
3. **Until real time reaches next June:**
   - no day snapshot is taken;
   - the shift stays at its last value;
   - the change-log files are never pruned, so they pile up.

A real long absence is not affected: it leaves the clock *later* than `active_day`, never earlier.

## Goals

- After a corrected clock, the next pass moves the open day and `saved_at` back to the real clock. Day closes then resume on real days. This holds however long the wrong clock ran.
- The state the pass saves still passes `engine_files` validation, so a rewind never triggers a recovery rebuild.
- A rewind deletes no change-log file.
- A pass that read the clock before waiting for the lock never drags the state back. That is the case `saved_at`'s forward-only hold exists for.

## Non-goals

- **Undoing the forward jump.** Hana ruled on 2026-11-27: "Unfreeze only".
  - The real day that was closed early stays closed.
  - The weights stay unaged across the skipped days.
  - Undoing it would mean recording what the jump changed so it could be reversed, a much bigger change for a rare event.
  - **Real snapshot history can already be lost.** Every day close drops snapshots 30 or more days before the day it closes. If the wrong clock passed a midnight more than 30 days ahead, that close dropped the real history, including the day the jump closed early. The rewind can't bring it back: after it drops the future days, the snapshot list can be short or empty. The loss comes from the jump, not from the rewind.
- **Catching the forward jump as it happens.** At that moment a jump looks the same as a real absence, and telling them apart needs a chosen number.
- **Future-dated change-log files.** Events folded while the clock was forward are written to change-log files named for future days. They are left in place, and pruning deletes them once real time passes their boundary. They are not read for anything the rewind changes.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | A new pure function in `engine_utils.py`, next to `close_days`. `_run_apply` calls it once, under the lock, before it holds `now` to `saved_at` | Decided from a clock read under the lock, so neither late events nor lock waits can trigger it |
| B | Rewind inside `close_days` | Ruled out. `_fold_events` calls `close_days` once per event with that event's own time (`close_days(state, min(event.at, now))`), so a late event dated yesterday would look like a corrected clock |
| C | Repair the state when it is loaded | Ruled out. Reads take no lock and change nothing, so the drain would still have to write the repair |

Hana approved A on 2026-11-27 ("Yes, write it up").

## Design

### The rewind rule

Hana ruled on 2026-11-27 ("Clock read under lock"). The rewind is decided from a clock read while the pass holds the engine lock. That clock is the later of the time the pass was given and a fresh `_now()` read inside `_run_apply`.

- **Production:** the fresh read is the real clock.
- **Tests:** a test that passes a fixed future time keeps that time.
- **Callers:** every caller of `note_active` passes `datetime.now(UTC)`. A future caller must not pass a time derived from stored state, because the later-of would keep it.

The rewind is due when the day of that clock is earlier than `active_day`. It **fires** only when the steps below would change the state: the open day moves, or a snapshot day is dropped. A due rewind that would change nothing does nothing. The pass then holds `now` to `saved_at` as today, saves only for its usual reasons, and the quick skip can still skip. Without this rule, a same-day correction would leave the open day at tomorrow, and every pass until midnight would count as a rewind and write the state.

**Why this protects the lock-wait race.** The only way to see a time earlier than the last save is to read the clock before waiting for the lock. `note_active`'s caller does exactly that. A fresh read under the lock comes after any save another pass made while this one waited, so its day is never earlier than the `active_day` that save set. A rejected first rule, the snapshot floor, protected the race only when the racing close had added a snapshot. It also missed every correction made after the wrong clock passed a midnight.

When the rewind fires, with `R` the clock's day:

1. **Future snapshot days are dropped.** Hana ruled on 2026-11-27: "Drop them". Every snapshot day later than `R` is removed from every emotion, because those days never happened. The real day the jump closed early is on or before `R`, so it stays, unless a later wrong close already dropped it (see Non-goals). Dropping a tail is safe for the readers:
   - `shift_for` reads only the last day;
   - each day's run record seeds its own run;
   - the validation's strictly-increasing order holds for a shorter list;
   - the next close counts its off days from the new last snapshot.
2. **The open day moves back.** `active_day` becomes the later of `R` and the day after the newest remaining snapshot day of any emotion. Snapshot days must strictly increase (`engine_files` validation), so the open day never moves onto a day that already has a snapshot. A correction made on the same real day as the jump therefore leaves the open day at tomorrow, and day closes stay frozen for at most a day.
3. **Last-entry days are pulled back.** Every emotion's `last_entry_day` later than the new `active_day` is pulled back to it. Otherwise `engine_files` validation would reject the state, because it refuses a `last_entry_day` later than `active_day`.
4. **The pass runs at the real clock.** It uses the clock as `now` for the rest of the pass, instead of holding it to `saved_at`. The save then writes `saved_at` at that time.

When it does not fire, nothing changes, and the pass holds `now` to `saved_at` as it does today.

### The pass

In `EmotionEngine._run_apply` (`brain/emotion/engine.py`):

- **Order.** The rewind runs on the loaded state, under the lock, before `now = max(now, state.saved_at)`. When it fires, that hold is skipped.
- **Pruning.** `pre_day`, the open day before the pass's own closes, is reset to the rewound open day right after a rewind that fired, before a backup state's catch-up, so the catch-up's own day closes still count as closes and prune as today. A rewind is then not counted as a day close and prunes nothing. Without this, `prune_changes` would anchor on the future day and delete every change-log file older than 30 days before next June, which is all of them. Pruning resumes at the next real day close, anchored on a real day.
- **Saving.** A pass whose rewind fired saves the state even when nothing else changed. The condition that returns early without saving gains a `rewound` term.
- **The quick skip.** `_can_skip` runs before the lock, on the cached state, with the time the drain was given. It does not skip when a rewind at that time would fire on the cached state. The rewind function's "would it change anything" check is pure, so `_can_skip` can call it. The pass then takes the lock, and the rewind decides there. Otherwise an idle persona with an unchanged queue would never reach the rewind.
- **The direct apply.** It (`drain=False`) goes through the same `_run_apply` and gets the same rewind.

### Accepted edges

- **A time-zone move west.** `day_of` uses local days, so after a move west the clock's day can be earlier than a healthy `active_day`.
  - A move of up to a day changes nothing: the open day is already the day after the newest snapshot.
  - Near the extremes, a move can make the day just closed look like a future day. The rewind then drops that one real snapshot day and reopens it, and the next close writes it again.
- **A wall clock that steps back across midnight on its own,** for example a small automatic correction at 00:00. The rewind is due, but with a snapshot for the day before it changes nothing. Without one, it reopens that day, which held no present emotion.

- **A backup state with catch-up.** When the loaded state is a backup copy, the rewind runs before `engine_rebuild.catch_up`, so the catch-up gets the real clock. If the backup's `saved_at` is still in the future, the catch-up takes `after = min(saved_at, now)` and replays nothing between the backup and the correction. That is the edge `engine_rebuild.py` already accepts for a `saved_at` ahead of `now`. The next rebuild folds those rows.
- **Other stored times left in the future.** Nothing the rewind or the day close reads uses these, so the effect is cosmetic:
  - `reset_at`, the time of the last catch-up, shown in the engine status;
  - `persona_start_at`, when it was first set during the jump.
- **Repeat records.** One with a future `first_at` is replaced by the memory's next showing, or its next commit or assignment within the window. Otherwise the existing sweep removes it once real time passes `first_at` plus the window.

### What needs no change

- **Weights:** `fold_event` holds a future `last_update` to `now` on the next event. Until then, the day close keeps advancing it, as it does today.
- **The rebuild:** both modes start from a fresh `EngineState`, fold their points at `min(point.at, now)`, and end with `close_days(state, now)`, so a rebuild can't produce a future open day. The builder confirms this with a test.

### The docstrings

- `close_days`: the "Accepted edges" entry for a clock set forward and corrected is rewritten to say `_run_apply`'s rewind handles it, and to name this spec.
- `_run_apply`: its sentence that `saved_at` never moves backwards gains the rewind's exception.

## Tests

Each through the real `EmotionEngine` on a temporary persona, except the first:

- **The rewind function on its own:**
  - fires when the clock's day is earlier than `active_day`, and not otherwise;
  - drops only snapshot days later than the clock's day;
  - places the open day, both with and without remaining snapshots;
  - pulls `last_entry_day` back.
- **The canary (I16).** Save at a forward clock less than 30 days ahead and run past at least one wrong midnight, so future snapshots exist and the real days are still kept. Then drain at the real clock. Check that:
  - `active_day` and `saved_at` come back;
  - the future snapshot days are gone;
  - the saved state reloads through `engine_files.read_state`;
  - a drain after the next real midnight appends that day's snapshot.
- A same-day correction with no wrong midnight leaves the open day at tomorrow, not today. Later passes that day do not count as rewinds, do not write the state, and the quick skip can skip them again.
- A jump more than 30 days ahead with a wrong midnight: the rewind still unfreezes, with whatever snapshots are left, possibly none.
- A rewinding pass deletes no change-log file.
- **The lock-wait race:** a drain given a time just before an `active_day` that another pass's close has just set does not rewind, with and without a present emotion. The fresh read under the lock is at or after that close.
- An idle persona with an unchanged queue still reaches the rewind (the quick skip does not hide it).
- A late event dated yesterday does not rewind (`close_days` is unchanged; existing tests cover it).
- A rebuild after a corrected clock produces no future `active_day`.

The tests set the times they need on both the given `now` and `_now()`, and avoid assuming the host's time zone, since `day_of` uses local days.

## Invariants

| Invariant | Holds? | Why |
|---|---|---|
| I5, persisted state readable by an older brain | holds | No field is added, renamed or removed. Only values change, and the saved state still passes the same validation |
| I14, works on macOS, Windows and Linux | holds | No platform-specific code. The tests do not depend on the host's time zone |
| I16, a physiology invariant ships with a canary | holds | The canary test above runs through the real drain |
| I1, I2, I3, I4, I6–I13, I15, I17 | n.a. | No user surface, organ, store, cadence, provider, prompt or overlay code |

## Wiring

No new organ. The rewind only lets the engine's existing day closes run again. They feed the shift, the snapshots and the change-log pruning, as they do today.

## Sequencing

This is ToT's half of the emotion rework, and his open #803 changes the lines around the day close in `engine.py` and `engine_utils.py`. So:

1. This design is posted on #660 for him first.
2. The build starts after #803 merges.

## Left to the builder

- The function's name and signature, for example `rewind_future_day(state, clock) -> bool`.
