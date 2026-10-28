# Emotion rework: the engine, the readers, the safety net and the access fixes

Design for ThinkerOfThoughts's half of the emotion rework (#240), written 2026-10-28. It describes what each part does, how it works, the order it is built in and what waits on what. File and function names refer to `origin/main` at 7496149e. The other half (the writers, the vocabulary, the cleanup and the writer-side tests) is planned by hanamorix on her branch `hana/emotion-rework-plan`, with her design documents under `docs/source-spec/` there; this document mentions her work only where an interface or the build order depends on it.

Contents:
1. Overview
2. The emotion engine
3. The other parts
4. Build order
5. Design invariants check
6. Accepted edge cases
7. Persona-facing wording left to the end
8. Live tests after the build
9. Checks at build time
10. Corrections to the shared plan
11. Not in this rework, and follow-up issues

---

## 1. Overview

### 1.1 What changes

Today "how does she feel" is answered separately at each surface: the prompt, the body block, the panels, the tools, reflex, research and the rest each scan memory rows (most of them the newest 200) and max-pool the emotions they find through `aggregate_state`. The answers disagree. A 15-minute decay pass rewrites the emotion values stored on memory rows, and some writers store emotion-only rows that exist only to carry a signal.

After the rework a per-persona **emotion engine** keeps two moving averages per emotion, a short one (about the last day) and a long one (about the last month). Memories she lives through, makes or opens, and emotion changes from events, are folded into those averages. Every reader asks the engine. Values stored on memories are no longer rewritten by a decay pass; they are faded when read.

The rework runs in two phases:
- **Phase B (shadow).** The engine is built and runs beside today's code: writes queue events, drains fold them into a shadow state, and nothing reads it. Every reader is changed to `if engine.is_live(): <new> else: <today, unchanged>`. No persona is live in phase B.
- **Phase C (switch-over).** At a persona's first start under phase C a start-up step fixes the scale of old nudge rows, rebuilds the engine's state from the last 30 days of memories, and writes a switch file. From then on the persona is live, the old decay pass is gone, and the old reader branches are deleted.

### 1.2 The parts of this half

| Part | What it is | PRs |
|---|---|---|
| The engine | The engine itself, built in slices S0 to S7, including the gate work (S5) and the switch-over (S6) | eight, one per slice |
| The readers | Every reader of "how does she feel" moved onto the engine (phase B), then the old paths deleted (phase C) | two |
| The reader-side safety net | Tests that pin what the readers do today, an exact pin of the gate-bypass list, and timings of three old paths; no `brain/` code | one |
| The access fixes | Every full open of a memory sets last-accessed and reports the showing to the engine; three background reads stop counting as accesses | one |

### 1.3 What the other half provides

hanamorix's half, in her own PRs, all of her behaviour changes gated on the switch:
- writer-side tests that pin today's writers;
- cleanup: the blend detector, the residue queue, the peer table, and `protected` taken out of the migrator, the CLI and recovery;
- the weather-shift anchor removal, after #385 merges, keeping `Baseline` and `update_baseline` in `felt_time/weather_shift.py`;
- vocabulary work, including: `add_memory` sets last-accessed and its auto-link no longer bumps, vocab repair stops bumping, the growth crystallizers' reads, `persona_loader.refresh_if_changed`, the store method `list_recent_emotions`, and the literals of the built-in half-life table leaving her vocabulary files;
- the writers, in five PRs: signed changes through `fold_change`; nudges on the 0 to 10 scale, the peer prompt and the trust accessor `peer_trust(persona_dir, peer_id)`; research markers (`EMOTIONS` and `ENGAGEMENT`, the rating stored as `metadata.engagement_rating`); dreams, including the dream mood read; the monologue note;
- an issue for the background reads in her files that stamp last-accessed without showing the memory (the ingest commit, research and dream reads; the dream read can wait for the dream rework).

Interfaces between the halves are stated in the PR that introduces each change, with the question it needs answered (section 4.7).

---

## 2. The emotion engine

### 2.1 Two moving averages per emotion

For each emotion the engine stores a short average and weight and a long average and weight. Each new entry with value `new` and weight `w` is folded in by:

```
weight  = weight × 0.5^(hours since last_update ÷ H) + w
average = average + (new − average) × w ÷ weight
```

`H` is 24 hours for the short average and 30 days for the long one. The average line is the delta form of the weighted mean `average × (weight − w) ÷ weight + new × w ÷ weight`; it gives the same numbers, but an average fed the value it already holds does not move at all, where the other form can be left with a move of about ±9e-16. That matters because the shift rule (2.6) compares daily moves exactly, with no tolerance.

- `w` is the source modifier (2.2) times the repeat factor: 1 for the first full showing of a memory, then ½, ¼ and so on for further showings within 24 hours of that first one. A commit counts as the first showing. The 24-hour list is a map keyed by memory id in the state file, pruned during drains (never on reads).
- "Hours since last_update" are wall-clock hours, except that whole off calendar days are skipped (2.3).
- An entry dated before `last_update` is pre-aged instead (`w × 0.5^(age ÷ H)`, age measured from `last_update`). Both readings store the same weighted sum, so folding the same entries in a different order gives the same averages.
- An entry dated in the future is treated as dated now; a `last_update` ahead of now is pulled back to now. Imported naive times are read as UTC.
- Averages stay within 0 to 10.
- Null is not 0: an emotion missing from an event leaves its averages untouched; an emotion present at 0 counts as 0.
- Names are canonicalised with `canonical_name` from `emotion/vocabulary.py` as they fold, including names on old rows. Two spellings of one emotion in one memory become one entry with the average of the two.
- An emotion is present once it has had an entry (long weight above 0). A zero entry never removes it: a decayed emotion stays present at a small number.

Both averages are moods. The long one is how she feels in general; the short one is the same over the last day or so. How she feels in the moment is her response to the day's events read against both, and is not a third number the engine keeps.

### 2.2 What moves an average

Only entries folded by the formula above, and the climax reset (2.5). There are three kinds of entry:

- **Experience.** A counted memory she lives through (committed), gets emotions later (assigned), or is shown in full (showing). The value is the memory's faded stored value (2.4), held to 0 to 10. A commit and an assigned event are dated at the row's `created_at`; a showing at the time it was shown.
- **Change.** A signed change from an event, through `fold_change(emotions, at, source)` (the chat extractor's pass 2 and `reconcile_self_read`). Each average takes its own stored value plus the change, held to 0 to 10, as the new value. An emotion with no history takes base 0: joy 0 with +3 becomes 3; −2 becomes 0, and that 0 enters at full weight as a present value. Changes count at full weight.
- **Quiet-day zero.** Value 0, weight 1, described in 2.3.

Which memories count, and their source modifier:

| Counts | Test | Modifier |
|---|---|---|
| Conversation memories | `is_conversation_derived(memory)` | 1.0 |
| Memories written with `add_memory` | the metadata is exactly `{"tags", "importance"}` | 1.0 |
| Her makings | `memory_type == "making"` | 1.0 |
| Her reach-outs and notes | `memory_type == "initiate_outbound"` and `initiate_state` is `"note"` or any `StateName` other than `"pending"` (the set taken with `typing.get_args`, not a hand list) | 1.0 |
| Kindled-peer memories | `memory_type == "kindled_peer"` | 0.1 + 0.9 × the peer's trust |
| Research memories | `memory_type == "research"` | 0.1 + 0.9 × rating ÷ 10, a missing or unparseable rating read as 0, the rating held to 0 to 10 |
| Self-corrections and chat changes | through `fold_change` | 1.0 |

Nothing else counts: the monologue family, emotion-only rows, journal entries, grief breadcrumbs, dreams, heartbeat, reflex output, file writes and `self_model_resolved`.

Peer trust is read through one engine-internal function, `engine_files.peer_trust`. Until the switch-over slice it returns 0.0, so every peer memory the shadow engine folds weighs the floor of 0.1; that reaches phase B's shadow state only, because nothing reads the engine before the switch. S6 replaces its body with a call to hanamorix's accessor `peer_trust(persona_dir, peer_id)`.

### 2.3 Active days, quiet days and off days

An **active day** is a calendar day, in the machine's local time, on which the app ran at all. A session across midnight counts as two. All day arithmetic goes through one function, `day_of(t)`. Each persona counts only the days its own supervisor ran.

**Quiet-day zeros.** When an active day D closes, every present emotion that got no entry on D gets one entry of value 0 and weight 1, folded into both its short and its long average. So a day the app ran without touching an emotion lowers that emotion's averages a little. An emotion with no average gets nothing and stays absent. The zero entries are dated at the first instant after D and do not count as "got an entry". The engine tracks, per emotion, `last_entry_day` (the active day of its latest real entry); it is the only record of whether an emotion got an entry that day.

**Off days.** Days with the app off get no call, so no entry. When the app next runs on a later day, the last active day is closed, nothing is made for the days between, and `last_update` is moved forward by the length of those whole off calendar days, so the weights do not age across them. An off stretch inside a day, or across midnight without a whole off day in it, still ages the weights. Ordinary daily use with the app closed overnight therefore has no whole off day.

`close_days(state, t)` is one pure function (no lock, no write) and the only code that opens or closes a day. If `active_day` is unset it sets it to `day_of(t)`; if `day_of(t)` is the same or earlier (a clock set back) it does nothing; otherwise it closes the open day (the zeros, then the snapshot day of 2.6), skips whole off days, and opens `day_of(t)`. Day closes happen inside drains (2.8), and inside a rebuild's or catch-up's scan (2.11), and nowhere else: reads never close a day. Because the drain folds queued events before it closes their day, a day never closes while the queue holds an event dated in it.

Worked numbers (formula as written, entries at noon):
- Joy 8 once, then quiet running days: after 1 quiet day short 2.09, long 3.93; after 3, 0.38 and 1.91; after 7, 0.02 and 0.91; after 30, 0.00 and 0.18.
- Joy 6 once a day for 120 days: short 6.00, long 6.00; after one quiet day short 2.49, long 5.85.
- The same joy 8, then three off days and a quiet running day: short 2.09 and long 3.93 after that day closes, the same as one quiet day. Ageing across the off days would have given 0.34 and 3.79.
- An entry of 6 just after the start of day D, then quiet: the short average holds 6 until the zero for D+1, dated about 48 hours on, lands against an aged weight of 0.25: 6 × 0.25 ÷ 1.25 = 1.2. The short average is therefore coarse; the reader thresholds are measured on it (3.4).

### 2.4 Fading a memory's stored value

A value stored on a memory row is faded when it is read, never by a background pass:

```
faded = value × 0.5^(age ÷ (half_life × (1 + HORIZON_IMPORTANCE_GAIN × importance ÷ 10)))
```

Age is measured from the row's `created_at`, or from the switch-over time when that is later (rows written under the old decay pass were already being faded until then). Love and belonging never fade below 1/3 of the stored value; no other emotion has a floor, and the floor applies only to values on memories, never to the averages. `HORIZON_IMPORTANCE_GAIN` is imported from the salience code, not copied.

The half-life of an emotion is looked up in this order: the persona's vocabulary file, then `EMOTION_HALF_LIFE_DAYS` in `brain/dev_constants.py` (love 730 days, belonging 365), then 14 days (`EMOTION_DEFAULT_HALF_LIFE_DAYS`). A null entry falls through to the next source; a corrupt vocabulary file is logged and the built-in values are used, with the file left alone.

The fade is one call, `faded_value(memory, emotion)`. Before the switch it returns the stored value unchanged, because the old pass is still fading the rows.

### 2.5 Climax reset

When a newly committed memory has a stored climax of 7 or more, the engine first folds the memory's own emotions, then acts directly on the stored short averages: arousal × 0.2 (not below 0.5), desire × 0.6, comfort_seeking and rest_need +2 (held to 10), climax set to 0. Each is changed only if present; an absent emotion stays absent, so the +2 is skipped for a comfort_seeking or rest_need with no history. The reset is not an entry: it adds no weight, does not touch the long averages and does not set `last_entry_day`. It does not fire for assigned, shown, merged or rebuilt memories.

### 2.6 The shift rule: when a feeling has shifted

The self-model block says which feelings have shifted and which way they are moving. A feeling has shifted when its short average has been moving one way for longer than its long average has moved one way in any finished run in the last 30 days. In full:

1. **Sampling.** Once per active day, when day D closes (after D's zeros and any climax reset), each present emotion's **snapshot** gets one day appended: D's label, the short and long averages, whether she felt the emotion on D (its `last_entry_day` is D; a quiet-day zero never counts as felt), and the long average's run record. Days 30 or more calendar days before D are dropped. The snapshot holds numbers and dates only (no memory ids), about 100 KB at 76 feelings, and it is the only history the rule keeps.
2. **Daily move.** A day's move is its value minus the previous snapshot day's, compared exactly. The oldest day has no move. Each calendar day between two snapshot days is an off day and counts as a zero move.
3. **Runs.** A move with the run carries it on. A move against it, the other way or no move at all (an off day, or an average stuck at a value, or at 0 or 10), is a blip: the run carries on and the blip day counts toward its length at once. A second day against it in a row ends the run. The finished run ends before the first of the two days against it; those two days become the first two days of the new run. If the second of them is a zero move, no new run starts. If only the first was a zero move, the new run starts on the second day with length 1. Otherwise the new run starts on the first day with length 2.
4. **The test.** A feeling reads as shifted when its short average has a run in progress whose length (blip days and a pending day against it included) is greater than the longest finished run of its long average in the last 30 days, of either direction. Equal does not count. Only whole-day counts are compared: no average value, difference or tolerance.
5. **The fall bar.** A falling run counts only if she felt the emotion on more of the run's days than not (a tie does not count). Every day of the run counts toward the length, blips and off days included; an off day is never felt. So a fall made only of quiet days never counts. A rising run is never filtered.
6. **The direction.** The flag carries the run's direction: +1 rising, −1 fading. Where the feeling sits is the sign of short minus long. The block states both, so a feeling can be "stronger than usual, but fading" or "quieter than usual, but rising".
7. **Clearing.** A feeling stops reading as shifted as soon as the run it was flagged on ends. The new run is then tested like any other; a new run of length 2 can be flagged at once in the opposite direction if the bar is 1 and, for a fall, both days were felt.
8. **The start gate.** Nothing reads as shifted until 30 days after `persona_start_at`, the `created_at` of the persona's oldest memory of any type (2.11 says how it is kept). The gate is for a new persona only: an existing persona has old memories, so its gate is open at the switch-over, however sparse its last 30 days. A persona created from imported history is also open from the start. The snapshot is kept from the first day, so the history is there when the gate opens.
9. **Quiet rule.** A feeling whose long average has no finished run in the snapshot never reads as shifted.

**The long run's record.** The short run only counts days inside the snapshot (at most 29 moves). A long run is measured from its true start, which may be before the snapshot: each snapshot day stores the long average's current run as a record of three fields: its start date, its direction, and its last day that moved with the run. The next day close carries the record forward from the previous day. Nothing else of the run is stored. Every day after the last with-direction day is a day against the run, so a pending first day against it is exactly the day after that last with-direction day, and whether that day was a zero move or a move the other way is read from the snapshot's own move for that day.

**Where it is computed.** The shift is worked out when `read()` needs it, by one pure function over the snapshots (two passes over at most 30 days per feeling), and cached with the state, so it is recomputed at most once per state write. It therefore changes only when the snapshot does: at a day close, a rebuild or a catch-up. Readers never compute it.

**What it was sized against.** On the synthetic database (109 active days, 76 feelings, four ticks a day, 296 ticks after a 35-day warm-up) the rule flags 2.8% of present feelings, 1.84 on average and at most 5; at least one feeling is flagged on 82.4% of ticks and none on 17.6%; and it allows at most 3.30 articulation calls per active day, before the articulation threshold filters some. Other forms were sized and set aside: the per-feeling "usual gap" test (about a third of feelings open on every tick), a percentile gate (needs a chosen percentile), a strict run end on the first day against (too many false ends), and a majority run end.

### 2.7 Reading the engine

`engine.engine_for(persona_dir)` returns the persona's engine; callers import the module (`from brain.emotion import engine`, or `emotion_engine` in `brain/cli.py`) so one test patch reaches every site. Its calls:

| Call | Returns |
|---|---|
| `read()` | `dict[name, Reading(short, long, difference, shift)]` for each present emotion whose name is in the vocabulary registry; `difference` is short minus long; `shift` is +1, −1 or 0 |
| `status()` | `EngineStatus(caught_up, reset_at)`: whether the loaded state is caught up, and the time of the last rebuild or catch-up |
| `is_live()` | whether the switch file exists |
| `faded_value(memory, emotion)` | the faded stored value (2.4) |
| `half_life(emotion)` | the half-life lookup of 2.4 |
| `oldest_entry_at()` | the oldest entry folded into the state (at most 30 days back after a rebuild); no reader uses it any more, and it stays |
| `note_active(now)` | runs a drain (2.8) |
| hooks and `fold_change` | append events (2.8) |

Rules for `read()`:
- It returns fresh immutable values, never cached objects, and reads never write: no heal, no quarantine, no backup rotation, no day close.
- It returns `{}` while the state is unreadable, while a recovery rebuild is pending, after a catch-up from a backup has failed (until the next successful state write), and on any read error (which is not cached, so the next call retries). Readers then show the persona's emotions as absent.
- The state is cached in-process and re-read when `(st_mtime_ns, st_size, st_ino)` of the file changes, so a read is a dictionary lookup.
- Before it filters by the vocabulary registry, `read()` calls `persona_loader.refresh_if_changed(<path of the persona vocabulary file>)` so a word registered in another process is seen. The call is in its own `try`: if it fails, does not exist yet (before hanamorix's vocabulary work lands) or has a different signature, the failure is logged and the read goes on with the registry as it is, rather than returning `{}`. The refresh is expected to re-read only a vocabulary file this process has already loaded, so the per-turn tool process, which never loads the persona vocabulary, keeps dropping persona words as it does today.
- Feeling names no longer in her vocabulary are left out of every read, as today; the stored numbers keep them.

### 2.8 Hooks, the queue and the drain

Writes never fold. A hook appends an event to a queue file, and a **drain** after the reply folds the queue.

**Hooks.** `MemoryStore.create(memory, *, fold=True)` and `MemoryStore.update(memory_id, *, fold=True, **fields)` call the engine after the commit:
1. A row whose type does not count returns at once, before any lookup, with one exception: if the stored `persona_start_at` is null or later than the row's `created_at`, it queues a start-only event, so the start gate still counts from the oldest row of any type.
2. For a counted row, the hook resolves everything the fold needs before taking the lock (the trust lookup, the modifier, the half-lives, the faded values) and builds the event.
3. It takes the lock, appends the event, and returns. Nothing inside a hook's lock hold touches the memory database, the trust store or a model.

Steps 1 to 3 are inside one `try/except`; a hook never raises into `create` or `update`. `update` fires the "assigned" hook only when emotions go from empty to filled, using the old row it already reads. An in-memory (`:memory:`) store skips the engine. The events are a commit, an assigned event, a showing, a change and the start-only event; each carries its own entry, so a row deleted before the drain still folds.

Which writers fire which hook:

| Writer | Hook |
|---|---|
| `create` by the gate promote, the pass-2 extractor, `add_memory`, the maker, reach-out, note and peer writers | commit, if the type counts |
| `create` by recovery's restore branch and the migrators | none (`fold=False`): they are not new experiences, and the first rebuild covers the past |
| the emotion backfill's `update` (empty to filled) | assigned |
| content-only updates, the gate merge, the temp-gate write-back, `update_emotions_batch`, the x10 fix | none |
| ingest | hanamorix's call; the default folds |
| a full open through the memory door (3.5) | showing (`on_memory_shown(memory, at)`) |
| pass 2's chat changes and `reconcile_self_read` | change (`fold_change`); on a live persona also one change-log line, appended first, in the same lock hold |

**The drain**, `note_active(now)`:
1. If there is no queue file, the day has not advanced and the cached state was not loaded from a backup, return at once with no lock and no write.
2. Take the lock; read the state (read-only walk, 2.9) and the queue. If the state came from a backup copy, catch it up first (2.11).
3. Fold the events in time order, calling `close_days` at each event's time before folding it.
4. Call `close_days(state, now)`.
5. If a day closed, delete change-log files older than 30 days (2.10).
6. Write the state once.
7. Delete the queue file, and release the lock.

**Where drains run.** Each call site calls `note_active` bare; it catches and logs its own failures.
- the end of the supervisor's `run_folded` pass, when no reply is in flight;
- the bridge `lifespan` at start-up (after the start-up step in phase C);
- the WebSocket `stream` handler after `ws.close(code=1000)`, in a worker thread via `tracked_to_thread`, so a waiting lock never stalls the event loop;
- the HTTP `chat` handler through FastAPI `BackgroundTasks`, after the response;
- `nell chat --no-bridge` (`_chat_direct_mode`), after each reply and after each `_drain_pass2_at_exit`;
- the CLI heartbeat (`_heartbeat_handler`), after `run_tick`.

So events from inside a reply reach the averages right after the reply is sent; events outside a reply (a gate commit in a heartbeat tick, a pass-2 change after the lull) are drained by the supervisor pass. At most one turn's worth of events is ever waiting while drains succeed.

**The lock.** One `file_lock` on `state/emotion_state.json` (sidecar `emotion_state.json.lock`) guards all four engine files. It blocks with no timeout and is not re-entrant, so no code path takes it twice. Hooks and drains hold it briefly; a rebuild and a catch-up hold it for their whole run, and anything arriving waits (about a second on the measured database).

**Phase B.** Hooks queue, drains fold and close days and append snapshot days into a shadow state under `state/`. No change-log lines are written, `faded_value` returns stored values, nothing rebuilds, and an unreadable state on a persona that is not live simply starts empty. Nothing reads the shadow state. The first rebuild at the switch-over discards it.

### 2.9 The engine's files

All under `<persona_dir>/state/`, a new directory for single-file records (the existing persona state files stay where they are for now; moving them is #570). The writers create the directory; no migration is needed because every file is new.

| File | Holds | Written with |
|---|---|---|
| `emotion_state.json` | per emotion: the two averages and weights, `last_entry_day` and the snapshot; global: `last_update`, `active_day`, `saved_at` (time of the write), `reset_at` (time of the last rebuild or catch-up), `persona_start_at`, `oldest_entry_at`, and the 24-hour repeat map | `write_json`, three backups (`.bak1` to `.bak3`), compact |
| `emotion_pending.json` | the queue: a plain list of events | `write_json`, no backups (a drained queue must not come back from one), no fsync; removed when empty |
| `emotion_changes/YYYY-MM-DD.jsonl` | the change log (2.10) | `append_jsonl` under the held lock, failures logged |
| `emotion_switch.json` | `{"switch_over_at"}` only | `write_json`, three backups, written only if missing |

The state file keeps no entry history and no memory copies. Losing it never un-switches a persona or moves the switch time, because the switch is its own file.

**Reading the state.** A new read-only walk, `brain/health/json_io.py: read_json_walk(path, *, expect, validate) -> JsonWalk(data, source, problem, signature)`, tries the main file and then `.bak1` to `.bak3`, takes the first that passes the validator, and reports which copy it loaded. It never heals or rewrites anything (the existing backup walk lives only inside the heal path). The felt-time state reader in #385 should use it too; its author gets a note. `brain/paths.py` gains `state_dir(persona_dir)`.

**The validator.** It rejects a non-finite number, a negative weight, a missing or bad `saved_at`, a bad `reset_at`, an emotion missing a field, `last_entry_day` later than `active_day`, and `active_day` unset while `last_update` is set. An average outside 0 to 10 is held to the range, not rejected. `persona_start_at` may be missing or null. A bad snapshot day (a non-finite number, a bad felt flag, a malformed run record, or a label out of order) does not make the state unreadable: it is dropped and treated as a missing day, which counts as a day against a run, with a log line naming the emotion.

### 2.10 The change log

A chat change or self-correction is stored today as a memory row with its sign lost, so a rebuild from memories alone cannot recover it. The change log exists only for that: it is a crash-recovery record, never read in normal running, read only by a rebuild or by the catch-up of a backup.

- One file per local day, `emotion_changes/YYYY-MM-DD.jsonl`, one line per chat change and self-correction (time, source, the feelings with signed amounts). A day with no change has no file.
- At each day close the drain lists the folder and deletes whole each file whose day is before the label 30 calendar days before the active day as it stood at the start of that drain. Nothing is scanned or rewritten. Anchoring on the pre-drain active day means a clock set far forward cannot delete the folder in the same drain that jumps the day.
- A missing file anywhere in the series, a missing folder, a corrupt line or a torn tail adds nothing and is not an error; the readable lines are used. A folder that cannot be listed is logged and treated as missing; it never stops start-up.
- A failed append is logged; the change still folds through the queue, so only a later rebuild loses it.
- Phase B writes no change log, so the first rebuild replays none.

### 2.11 The rebuild and the catch-up

**The rebuild** is one function, `engine_rebuild.rebuild(persona_dir, mode, now)`, with two modes. **First** runs once per persona from the start-up step; **recovery** runs at a quiet moment when a live persona's state is lost. Its caller takes the lock once and holds it (`ensure_engine_ready` for the first mode, `recovery_rebuild` for recovery); `rebuild` itself never takes it. In order:
1. Read the counted memory rows created or accessed in the last 30 days of wall time, active or forgotten; the creation times of all rows in that window; the change-log files from the day before the window starts; and the oldest row's `created_at` (one `SELECT MIN(created_at)`). All reads are plain SELECTs that bump nothing, fire no hook and skip the integrity check.
2. Build the state in memory from empty with `replay(state, from, to)`: one time-ordered pass that folds each memory at `max(created_at, last_accessed_at)`, folds each change-log line at its own time, and calls `close_days` at every point, so quiet-day zeros, off-day skips and snapshot days work exactly as live. A past day counts as active if any memory row of any type was created on it, or a counted memory was accessed, or a change-log line was written. Memories created in the last 24 hours enter the repeat map as first showings. No climax reset is replayed. Then `close_days(state, now)`.
3. In the first mode use stored values as they are (the old pass kept them faded until now); in recovery mode fade them (2.4).
4. Set `persona_start_at`: the query result in the first mode; in recovery mode the earlier of the query result and the old state's value when one is readable, because the temp-gate write-back moves recalled rows' `created_at` forward.
5. Check: every number finite and every average within 0 to 10; then write with `write_json(..., verify=True)`, which re-reads the result and restores `.bak1` if what landed is wrong (a False result is a failed check). `saved_at` and `reset_at` are set to now.
6. In the first mode, write the switch file.
7. Delete the queue: every queued commit, assignment and showing is already a row or an access, and every queued change is a log line, so the rebuild has folded them.

A failed check or an error leaves the old state, writes no switch, and logs what failed. A first rebuild's failure stops start-up (2.12); a recovery rebuild logs once per active day and retries at each quiet moment. A crash before the state write leaves the old state and the same rebuild reruns. A rebuilt state is a fresh 30-day history, not the live one: it folds each memory once where live folded every showing, and its snapshot holds only the replayed days, so flags after a rebuild may differ from those before it.

The prototype on the synthetic database (19,413 rows, 684 counted rows in the window, 45 emotions) took about 1.2 s cold and 0.08 to 0.13 s warm on a low-end laptop VM. The real cost is measured in the switch-over PR (3.3).

**`persona_start_at` in normal running.** No query runs: every committed row of any type min-folds it (the hook's start-only event covers uncounted types), and a new persona with no rows has null, which keeps the gate shut. A catch-up keeps the loaded state's value and folds into it.

**The catch-up.** When the read-only walk falls back to a backup copy, the loaded state is as old as that copy's `saved_at`. The drain that loaded it replays everything since `saved_at` with the same `replay` function, from the loaded state, up to the time of the oldest event still in the queue (to now if the queue is empty), with no final close; the drain then applies the queue as usual and writes once, stamping `reset_at`. Stopping at the oldest queued event means no queued event is counted twice. If the catch-up errors, or the backup's `saved_at` is more than 30 days old (the change log keeps only 30 days), the drain logs, writes nothing, keeps the queue, and sets an in-memory flag that empties `read()` until a successful state write; for a live persona the recovery rebuild then runs. A `saved_at` ahead of now is held to now.

**The recovery job.** A new gated job of the central cadence, `emotion_recovery_rebuild`, first in `GATED_JOB_ORDER`, with no interval: `has_work=needs_recovery` (the switch exists and the state does not parse or validate with nothing readable in the backups, or a catch-up failed in this process), `run=recovery_job_run`, which catches and logs (once per active day) and returns `JobOutcome.SKIPPED` on failure. It runs only at a quiet moment (no reply in flight and the lull passed). Readers show absent until it succeeds. A queue file that does not parse never starts a rebuild: it is discarded with a log line.

**`reset_at` and the self-model.** The engine never writes the self-model's state. Instead every rebuild and every catch-up stamps `reset_at`, and the self-model tick reads it (3.4) to avoid comparing a saved gap with flags from a different history, which could otherwise close an open gap and write a false importance-9 "resolved" memory.

### 2.12 Start-up and the switch-over

`ensure_engine_ready(persona_dir)`:
1. If `state/emotion_switch.json` exists, return at once (no lock, no transaction). This is the normal case for a live persona.
2. Run the **x10 fix** (below), once per persona.
3. Take the lock, and return if the switch file now exists (two processes starting one persona cannot both rebuild).
4. Call `rebuild` in the first mode with the lock held; it writes the switch as its last step. Release.

It raises `EngineStartupError(what, cause)` on a failed check, an error, or an x10 error or rollback, and the failure stops start-up; no switch is written, so the next start runs the step again. From the CLI the error escapes `main` and lands in the existing `cli-crash.log`. A persona created under phase C runs the same step over zero memories and gets its switch at once.

**Where it runs.** In the bridge `lifespan`, between the vocabulary load and the backlog migration, before the supervisor starts and before its own drain; and in each CLI handler that opens a persona, after its own `load_persona_vocabulary`: `_chat_direct_mode`, `_heartbeat_handler`, `_dream_handler`, `_reflex_handler`, `_research_handler`, `_soul_review_handler`. On a persona with no switch file a `--dry-run` skips the step, runs no reader, writes nothing, prints a plain message to stderr asking for one normal start, and returns non-zero; on a live persona a dry run works as today. The per-turn tool process is always spawned from a session that has run the step.

**The x10 fix** (`nudge_scale_fix.py`). The maker, reach-out, note and peer writers stored small nudge values (about 0.08 to 0.25, peers up to a 0.6 window total) on a scale that runs to 10; hanamorix's writer work moves them onto the 0 to 10 scale behind the switch. The fix brings old rows onto the same scale:
- rows of type `initiate_outbound`, `kindled_peer` or `making` whose `peak_emotion_intensity` is below 1 (one threshold for all three; the peak test keeps out rows the emotion backfill re-tagged on the 0 to 10 scale, and decay never lowers the peak);
- `emotions_json`, `score` and `peak_emotion_intensity` multiplied by 10 (a fixed row then looks like a new-scale row; the peak is below 1, so every channel stays below 10);
- the done-check, the row writes and the marker in one `BEGIN IMMEDIATE` transaction; the marker is `PRAGMA user_version` = 1 (a later migration takes 2); rolled back if the number of rows changed differs from the number selected;
- no cutoff time: the fix runs before the switch exists, and every writer change is gated on the switch, so at that moment every row of those types is on the old scale. This relies on hanamorix's nudge-scale change being gated (asked in S6's PR body).

Why a `user_version` marker: it is transactional and adds nothing to the schema. A file marker leaves a crash window that would scale rows twice; a per-row flag adds a column to every row; and a value test cannot tell a fixed row from an unfixed one (0.05 × 10 is still below 1). The file carries a comment flagging it for the backwards-compatibility cleanup (#388).

### 2.13 Failure behaviour, in one place

| What fails | What happens |
|---|---|
| A hook raises, or its append fails | Logged; the memory write is unaffected; that event is lost (accepted, 6) |
| A queue event is invalid (no memory id on a showing, no emotions on a commit or assigned event, an unknown kind, a missing time, a non-finite amount) | Dropped with a log line saying why; the others fold |
| The queue file does not parse | Discarded with a log line; no rebuild |
| A drain fails (disk, permission) | Logged; the queue stays and the next drain retries |
| The state is unreadable | The walk tries the backups; a backup is caught up; with nothing readable a live persona gets a recovery rebuild, a phase B persona starts empty |
| A catch-up fails | Logged; `read()` is empty until a state write; recovery rebuild |
| The first rebuild or the x10 fix fails | Start-up stops with `EngineStartupError`; the next start retries |
| A recovery rebuild fails | Logged once per active day; retried at each quiet moment; readers show absent |
| A change-log file is missing or corrupt | Those changes are lost at a rebuild only |

The memories are the truth; the engine is derived from them. A failed engine update never loses a memory.

### 2.14 How the engine is tested

- **Fold tests:** the worked numbers above, out-of-order folds giving the same result, null against 0, a change on an emotion with no history, the future clamp, the fade with the 1/3 floor, and the climax order.
- **State and rebuild tests:** the validator, the backup walk, the cache, `read()` immutability and the registry filter, the half-life lookup, the refresh call (a name registered by another process appears; a missing refresh function does nothing; a refresh that raises or has another signature is logged and the read still returns names), the rebuild window, any-row days, change-log replay with missing files, first against recovery fades, a failed check keeping the old state and writing no switch, and a failing start-up step stopping start-up.
- **Queue tests:** the hook cases (never raises, which writer fires which hook) and the drain and its call sites.
- **One simulation of the shift rule end to end on a known history**, instead of piece-by-piece tests, plus the live test of section 8. Its cases include the start gate: a persona whose oldest memory is under 30 days old is gated; one whose first rows are of an uncounted type opens 30 days after the oldest of them; at a first rebuild an existing persona whose oldest memory is 90 days old with entries on its last 5 days only is not gated, and one whose oldest memory is 10 days old is.
- **Start order:** the call order of the start-up step, the dry-run cases, and a first start on a fresh persona that completes (so the lock is taken once).
- **x10 fix tests**, and one test that a peer at trust 1.0 takes modifier 1.0 once trust is wired.
- **Sizing check:** the shift rule as built must reproduce the sizing numbers of 2.6 on the synthetic database (up to the effect of the oldest-day edge in section 6); the slice that completes the shift (S4) states any difference.
- New tests in the store go in `tests/unit/brain/emotion/` and `tests/unit/brain/memory/`; a shared settable engine stub (`emotion_engine_stub`) and an autouse fixture that resets the engine registry go in `tests/conftest.py` (S3).

---

## 3. The other parts

### 3.1 Gate work (slice S5)

**The gate merge keeps emotions.** When the consolidation gate merges a candidate into an existing memory (`consolidation.py`, the merge branch of `_dispatch`) and the engine is live, it writes the merged content and merged emotions in one `store.update(target.id, content=merged, emotions=..., fold=False)`: an emotion on both memories takes the average of the two (joy 5 and 7 give 6), an emotion on one side is copied unchanged. A merge is not a new memory, so it folds nothing. Today the merge writes content only and drops the candidate's emotions. Not live, the merge stays as it is. The merge target is now pre-read with `bump=False`, live or not, so a merge does not count as an access of the target.

**The temp-gate write-back.** Remembering a memory refreshes it: you remember the last time you remembered it. When a recalled memory comes back through the temp gate (`_handle_reappraisals`) and the engine is live, after its importance update succeeds the gate computes the memory's faded values from the loaded row (its importance before this re-appraisal) and calls a new store method, `refresh_faded_emotions(memory_id, faded, now)`: one `UPDATE` of `emotions_json` and `created_at` (written as `created_at.isoformat()` in `+00:00` form) together, so a crash leaves both old or both new. `peak_emotion_intensity` is untouched, no hook fires, and the method raises `KeyError` if no row changed. It covers every memory type, and resets `created_at` even when the memory has no emotions. Later fades read the row's current (re-appraised) importance from the reset date. A failing write-back is caught at its call, logged, leaves the row unchanged and is not requeued (requeueing would repeat a model call).

Every reader of `created_at` now sees a recalled memory as new: its shown date in search, read-full-memory results and journal entries is the date she last remembered it; a recalled dream counts as a new dream in the maker's charge; a recalled conversation memory touches her interests again; a recalled grief event blocks a new grief touch for that person. These are accepted as in keeping with a refreshed memory, with no code change.

**Tests:** a live merge averages and calls no hook; a merge that is not live keeps today's behaviour; the merge leaves the target's `recall_count` and `last_accessed_at` unchanged; the write-back writes faded values and a new `created_at`, also for empty emotions; a raising write-back is logged and not requeued; and the store method's own cases.

### 3.2 Days since the user last spoke (slice S6)

Because the write-back moves `created_at` forward, memory dates can no longer say when the user last spoke. A new function, `brain/ingest/buffer.py: last_user_turn_ts(persona_dir)`, reads the active conversation files backward with a doubling window until it finds a row with `speaker == "user"`, falls back to the archive only when no active file has one, and returns None when there is none. `days_since_human` uses it, returns 999.0 for None, and reads no memory and no presence file. `_latest_user_turn_in_buffers` and `_latest_conversation_memory_ts` are deleted, and the stale comment in `extraction_backfill.py` is fixed. In S7 the now-unused `store` parameter of `days_since_human` goes, with its eight call lines and the store opens that only fed it.

**Tests:** the existing tests that built a "last user turn" from memory dates are rewritten (`test_memory_utils.py`, `test_buffer.py`, the reflex, extraction-backfill and migrated tests that depend on it).

### 3.3 The switch-over and the removals (slices S6 and S7)

**S6** lands the switch, the live fade and the end of the old decay pass together, so no memory is faded twice or not at all. It adds `rebuild`, `ensure_engine_ready`, the x10 fix, the recovery job and their call sites (2.11, 2.12); turns on change-log lines from `fold_change` and the fade in `faded_value`; wires peer trust to hanamorix's accessor; and adds `last_user_turn_ts` (3.2).

It removes: `heartbeat._apply_emotion_decay` with its `protected` check, `emotion/decay.py`, and what they leave unused (`HEARTBEAT_DECAY_BATCH_BUDGET_S`, `_DECAY_PAGE_SIZE`, `DecayCursor` and its resume path, `memories_decayed` and its readers including the heartbeat prompt segments built around it and the system segment's "decay applied", `update_emotions_batch`, `list_active_since`, and their tests), each name grepped again first. The prompt segments are edited as working text (section 7). The two safety-net test files that pass `memories_decayed=0` drop it, keeping their test ids. The dev-constants snapshot is re-approved.

**Timing in the S6 PR.** The safety net's timing script (3.6) runs on the parent commit and on the S6 commit, same host and same database copy, and both sets of numbers go in the PR description. On the S6 commit it starts the copy through `ensure_engine_ready` (untimed) after its vocabulary step, confirms `is_live()`, skips the decay timing (the pass is gone), and states its rebuild count. The first rebuild is also timed in this PR, before it is flipped: one cold run and warm repeats on a local copy of the synthetic database on the low-end host, with the first-load count of registered names. If the cold run is materially above the roughly 1.2 s the design assumed, it goes to ThinkerOfThoughts before the flip.

**S7** deletes what is left of the old design, after hanamorix's old `aggregate_state` reader branches are gone (4.2):
- `brain/emotion/aggregate.py` last, with `test_aggregate.py`, `test_climax_reset.py` and the importers of `_warned_unregistered` (the harness sandbox, its registry-leak test, the conftest registry-restore fixture and its test), after `git grep -E 'emotion\.aggregate|emotion import aggregate|_warned_unregistered'`;
- `protected` out of `store.py` (the dataclass, the INSERT, `update`, the row mapper; the column stays for old databases);
- `monologue_emotion` and `self_model_reconcile` out of `GATE_BYPASS_TYPES`, with the safety net's pin edited to `{journal_entry, initiate_outbound}` in the same step, and a dead-code check on `SALIENCE_ELIGIBLE_TYPES` and `MONOLOGUE_FAMILY_TYPES`;
- the emotion-only row writes in `chat/extractor.py` and `self_model/reconcile.py` (their changes go through `fold_change` instead);
- the unused `store` parameter of `days_since_human` (3.2).

### 3.4 The readers (phase B and phase C)

**Phase B: one PR.** Every reader of "how does she feel" takes its number from the engine behind the switch, in the shape `if engine.engine_for(persona_dir).is_live(): <new> else: <today, unchanged>`. An in-memory store reads as not live. Rules at every site:
- sites take `short`, `long`, `difference` and `shift` by name; they never filter names and never clamp (the engine's validator is the one boundary);
- an absent emotion is left out of thresholds, sums, summaries and top-N lists, never read as 0; a present 0 is real data;
- no threshold is retuned; the existing ones are measured instead (below);
- an unreadable engine shows absent at every site (the summary is left out, the body uses baselines, no arc fires); the existing fail-soft `except` blocks stay;
- one shared helper, `body_state_from_short(short, ...)` in `brain/body/state.py`, builds the body state from the short averages and strips the body emotions the engine did not return (the old builder fills absent names with 0.0);
- new constants in `dev_constants.py`: `EMOTION_SUMMARY_COUNT: int | None = 3` and `EMOTION_SUMMARY_MODE = "long"`, with the snapshot re-approved.

| Site | Takes | Notes |
|---|---|---|
| Prompt body block | short | via the body helper; body thresholds 5, 7 and 8 unchanged |
| Prompt mood summary | long | top `EMOTION_SUMMARY_COUNT` by long average, ties by name; modes `long`, `long_short`, `long_difference`; `""` when nothing is present. It now differs from the body block on purpose, and the comment that says they cannot disagree is rewritten |
| Monologue frame | the same string as the summary | no code change |
| Self-model block | the saved gap | `render_block(state, live=...)`; live, it sorts the saved channels by absolute difference, cuts to 3, and states position and direction; strings in `prompt_strings.toml` `self_model.ambient` |
| Persona panels (emotions, body) | short | absent stays absent |
| Tools `get_emotional_state`, `get_body_state` | short | |
| Rest gate | short, via body energy | closes its store in `finally` |
| Reflex | short | an absent emotion fails its trigger; a template that names an emotion absent from the read does not fire (`ArcSkipped(reason="template_emotion_absent")`); non-emotion placeholders keep "0"; a present 0 renders "0.0"; these rules apply live only |
| Research gate and its two prompt reads | short | `EmotionalState(emotions=short)` |
| Maker | short | |
| Soul review top 3 | short | |
| Felt-time `body_strain` | short body | |
| Felt-time `emotional_intensity` | nothing | unplugged at 0.0 with a comment for #386; the old computation stays in the else-branch |
| Heartbeat status summary | short | empty read: no daemon-state write (the line keeps its previous text); otherwise `_peak_emotion(short)` |
| Spike resonance | nothing | returns `(0.0, {})`; the emitter stays in the code but never fires until it is rebuilt (#569) |

**Per-memory readers** call `faded_value` unconditionally (it returns the stored value before the switch): the salience `_emotion_input` (the engine resolved once per sweep), the graveyard's `emotion_at_ingest`, the arc-close live read, the journal block's top 2, the heartbeat's `_peak_emotion` on fired rows, and `_mem_to_result`, which gains a required `persona_dir`. `_peak_input` and values read back from the graveyard are not faded again. The two safety-net pins that call `_mem_to_result` and `salience._emotion_input` directly get the new arguments in the same PR (a temporary persona dir, and `eng=None` for the stored value); they keep their test ids and the behaviour they pin.

**The self-model gap.** On the live path the self-model's own 72-hour and 720-hour windows and its noise floor are replaced by the engine:
- `gap_from_read(read, *, unnamed_pressure)` makes one channel per feeling whose `shift` is not 0, valued at its difference (size and position), with its direction in a new `Gap.directions` field. `compute_unnamed_pressure` (renamed) uses presence in the read and the short body inputs.
- `Gap` also gains `reset_seen` (with a sentinel `UNSET` for "never stored"); both new fields are serialised in `state.py`, and a missing `directions` loads as empty.
- The tick entry, `_self_model_engine_reads`, reads `status()`, then `read()`, then `status()` again. If the read is empty or the state is not caught up, it returns False and touches nothing (no gap, cadence or counter change), because an empty read is not "nothing shifted".
- If the two `reset_at` values differ from each other, or from the value stored with the saved gap (a missing stored value counts as different), a rebuild or catch-up has happened, so the tick applies the **spare rule** instead of comparing flags: a gap she acknowledged or dismissed with `sustained_ticks >= 3` is kept, any other is cleared without the resolution path (load, clear, save, as `self_model_repair.py` does). It writes no resolution that tick and stores the latest value. Otherwise it is a normal live tick.
- The reconcile cooldown filter prunes `directions` together with the channels. The articulation prompt carries each feeling's direction label. The articulation threshold stays 0.4.

With the shift rule, nothing is shifted for a persona's first 30 days, so no gap opens, no block renders and no articulation call or reconcile offer happens then; this replaces the earlier "silent on day one" rule.

**Threshold measurement**, run in the phase B PR: a script (not a pytest) on a copy of the synthetic database runs the engine with the live fold rules, quiet-day zeros included, twice: once with every day active and once with some days off. It reports, old against new, how often each threshold fires: body 5, 7 and 8; love 8 and the arousal 7 temperature bump; reflex arcs; the research peak 7.0; the rest gate; `body_strain`; and the maker's discharge rate (the maker feeds the short peak into `charge.accumulate`). The numbers go in the PR description for ThinkerOfThoughts to read before S6 merges; a large shift is raised straight away.

**Phase B tests:** reflex, heartbeat, the persona panels, the summary, the tools, a new `test_body_state_from_short.py`, felt-time, the faded-value readers, a narrowed same-numbers test for the summary and the monologue frame; and for the self-model, cooldown pruning, `gap_from_read`, the round trip of `directions` and `reset_seen`, the live ambient render, and the articulation text. `update_baseline` and `Baseline` stay in `felt_time/weather_shift.py`; moving `update_baseline` into `chat_log.py` is left to the felt-time project.

**Phase C: one PR**, after S6. It deletes every else-branch, `is_live` test, old query (the nine copied "newest 200 memories with emotions" queries), import used only by the old path, the reflex `.get(name, 0.0)` defaults, the heartbeat sums, the self-model windows, floors, `compute_gap`, `compute_derived`, `DerivedRead`, `_CHANNEL_FLOOR`, the `live` parameter, and the old-path comments. It keeps the old `emotional_intensity` computation as a private function that is never called, commented "dormant until #386", with its own 200-row query and its `update_baseline` import from `weather_shift.py`, and retargets the four felt-time tests that cover it. It fixes the prompt comments that describe the old paths. `_build_emotion_summary` and `_current_emotional_summary` take `persona_dir`. It re-points the safety net's same-answer cells at the engine read (3.6): the capture is each site's derived map, the reference the stub's short averages, the exact-window check is deleted, all 12 `xfail` marks and the old-path imports go, and the `no_base_failure` reason is reworded. It deletes the safety net's old-branch pin files (found with `git ls-files '*test_old_branch_*'`), edits `test_body_panel_smoke.py`, and deletes the tests of removed windows, floors and the old render. Acceptance: `git grep aggregate_state brain/` lists only hanamorix's callers and comments.

### 3.5 The access fixes (one PR, three commits)

A memory shown to her in full counts as an access; a read that shows her nothing does not.

**The door.** `brain/memory/recall_open.py`, the one door every full open goes through (the recall block's full renders and `read_full_memory`), loses its `deliberate` argument and does the same for every open: one `store.get(mem.id, bump=True)`, which adds 1.0 to `recall_count` and sets `last_accessed_at` to now in one `UPDATE`, plus the existing reappraisal enqueue. After the reappraisal branch, inside `if persona_dir is not None:`, one `try` imports the engine module and calls `engine.engine_for(persona_dir).on_memory_shown(mem, datetime.now(UTC))`, logging any exception. The call is placed after both branches of the reappraisal if/else because the recall block always passes a collector; a call in only one branch would miss every recall-block render. The engine's own type test discards uncounted types, and its repeat rule halves repeats within 24 hours. The legacy `persona_dir=None` path bumps and reports nothing. The four callers drop `deliberate=`. A comment at the `try` records the accepted edge that a crash between the bump and the append loses that showing, and the commit that adds it says so.

**Background reads that stop counting** (`bump=False`, which skips both fields): soul review's two context reads (`get` of the source memory and its `search_text` for related memories), and the recall-resonance staleness read in `initiate/resonance.py`, which today bumps up to 50 hebbian-active memories about every 15 minutes while showing her none of them, and carries a short comment. None of these edits is gated on the switch: they fix what the forgetting pass sees now, and nothing reads the engine before the switch.

Effects she will notice: a memory shown in full restarts its freshness, so it is slightly harder to fade; and the recall-resonance 7-day staleness gate now sees the real time since a memory was shown, so recall-resonance reach-outs can rise from today's near zero (bounded per memory by the anti-flood window). Neither field feeds a ranker. The stale docstrings and comments are updated (no em-dashes).

**Tests:** `test_recall_open.py` (the old two-mode cases flipped, renamed or deleted; a new test that every full open is reported to the engine once, four cases, using the engine stub); `test_review.py` (a fixed stub signature; the context reads do not tick); `test_resonance.py` (the tick reads without ticking). None of these new tests passes on base, so none needs `no_base_failure`. The related reads in hanamorix's files are in her issue (1.3); vocab repair's search and the `add_memory` read are her vocabulary work.

**GitHub:** the tracking issue first, then hanamorix's issue (assigned to her, `severity:high`), then the tracking issue edited with her issue's number. Phase C deletes nothing in this part.

### 3.6 The reader-side safety net (one PR)

Before any reader changes, tests pin what the readers do today, so a reader edit that changes behaviour unnoticed fails. Two permanent test files, seven `test_old_branch_*` files, a timing script with its shell wrapper, and one inventory entry in `scripts/README.md`. No `brain/` file and no existing test file is edited.

- **Same answer on every surface** (`tests/unit/brain/test_every_surface_same_answer.py`). It compares the emotion-to-strength map each strength reader pools (captured by wrapping `aggregate_state` at each site, and `_peak_emotion` for the heartbeat summary) against one reference, `aggregate_state` over the full dataset. Rendered text is not compared. Two datasets in file-backed stores: one (231 rows) with an emotive memory older than the newest 200 emotion-bearing rows, which every 200-row window misses; one (62 rows) with more than 50 newest memories without emotion, which soul review's 50-row window misses. 13 surfaces, 24 cells, one test per cell; each cell first checks the capture happened, that the dataset shows the case and that the window is exact (each failing with `pytest.fail`), then makes its one assertion. 12 cells are strict `xfail(raises=AssertionError)` (they disagree today) and 12 are plain. At phase C the readers' PR points the cells at the engine read (the reference becomes the stub's short averages, the stub giving distinct short, long and difference values and a nonzero gap since the last fold), removes the marks and the old imports, and the test stays as the one cross-surface test.
- **Bypass pin** (`tests/unit/brain/memory/test_gate_bypass_types.py`): `GATE_BYPASS_TYPES == frozenset({"journal_entry", "initiate_outbound", "monologue_emotion", "self_model_reconcile"})`, exactly. S7 edits it.
- **Old-branch pins**: every old path the reader edits keep until the switch is pinned; existing tests that already pin one are named in the PR's table, and 17 new pin tests fill the gaps (including exact boundary cases for the `>=` comparisons and the 0.5 floor). Phase B edits two of them for the new call shapes (3.4); phase C deletes the files.
- **Timing script** (`scripts/time_old_emotion_paths.py` and `.sh`): it refuses a database path under the app home, aborts if a `-wal` file has content, and copies only the main database file. It registers the database's feeling names through `load_persona_vocabulary` (printing the first-load count), prints a host line (including AVX2), and times three old paths in order: the current-state read (`get_emotional_state`), the trace write and the decay pass. Each is one labelled cold run plus 9 warm, reported as minimum, median and maximum with no pass/fail; the decay timing is 10 consecutive passes with the rows changed per pass. It reads the decay interval with `inspect`, and reports its rebuild count (0 in this PR). S6 extends it (3.3).
- **The PR description** has four parts: what is added, the pin table, the cell table with observed results, and the timing numbers with host, database checksum and commit. Commits run from the bypass pin first to the timing script last. Every new test function carries `@pytest.mark.no_base_failure(reason=...)` (one mark on a parametrised function), and `scripts/new_tests_fail_on_base.py --base origin/main --workers 0` must list exactly these functions as exempt.

### 3.7 Alternatives that were considered

| New part | Considered instead, and why not |
|---|---|
| One engine per persona behind `engine_for` | module-level functions; an engine owned by the store |
| `read_json_walk` | the existing heal path (writes on read); its private picker; #385's felt-time reader (felt-time specific, unmerged) |
| The queue file | the existing `PendingQueue` (a different job); the pass-2 queue pattern is reused |
| The change-log folder | the log-rotation helpers (none deletes whole per-day files); one file pruned by rewriting (rewrites a growing file); a table in `memories.db` (emotion-only entries in the memory database) |
| The state file and cache | a `memories.db` table or a separate emotions database |
| The recovery job | a new thread or timer; the central cadence already runs quiet-moment jobs |
| The x10 marker | a one-row table; a file marker; a per-row flag (2.12) |
| `last_user_turn_ts` | the presence file (not needed); a new field; a memory fallback; a throttle helper (one row, fails closed) |
| The HTTP drain via `BackgroundTasks` | draining before `return` (delays the reply); the supervisor loop only |
| Reader helpers, `Gap` fields, the tick helper, the reflex skip, the dormant intensity function, the summary constants | no new store, process, lock, queue or protocol |
| The timing script | a pytest marker, `pytest-benchmark`, a harness under `tests/`, `timeit`, a recipe in PR text |
| The access fixes | no new mechanism: `get(bump=True)` and the engine stub |

---

## 4. Build order

### 4.1 The PRs

| PR | Contents | Owner |
|---|---|---|
| Plan PR | the coordination draft PR (4.5) | ThinkerOfThoughts |
| S0 | the engine module as a stub with the real half-life lookup | ThinkerOfThoughts |
| S1 | the pure functions and the rest of the engine constants | ThinkerOfThoughts |
| S2 | the files, the read-only walk and the validator | ThinkerOfThoughts |
| S3 | the hooks, the `fold` flag and the test stub | ThinkerOfThoughts |
| S4 | the drain, `read()` with the shift, the catch-up and the shadow engine | ThinkerOfThoughts |
| S5 | the gate work (3.1) | ThinkerOfThoughts |
| Safety net | 3.6, one branch and one draft PR | ThinkerOfThoughts |
| Access fixes | 3.5, one PR, three commits | ThinkerOfThoughts |
| Readers phase B | 3.4 | ThinkerOfThoughts |
| S6 | the switch-over (3.2, 3.3) | ThinkerOfThoughts |
| Readers phase C | 3.4 | ThinkerOfThoughts |
| S7 | the removals (3.3) | ThinkerOfThoughts |
| #562, #563 | fixes for #555 (`close_session` unlinks without archiving) and #556 (the weekly rollover gap) | bug-fix PRs |
| #565 | `canonical_name` pooling in `aggregate.py` now | bug-fix PR |
| Her writer, vocabulary, cleanup and weather-shift PRs, and her access-fix issue | 1.3 | hanamorix |

**What each engine slice contains.**
- **S0.** `brain/emotion/engine.py` with every call of 2.7 as a stub: hooks and `fold_change` record their arguments in `recorded_calls`; `read()` returns `{}`; `is_live()` False; `status()` `(True, None)`; `faded_value()` the stored value; `oldest_entry_at()` None. `half_life()` is real from the start, because hanamorix's vocabulary work builds blend half-lives from it and can land before S1: S0 adds the half-life constants (`EMOTION_HALF_LIFE_DAYS`, `EMOTION_DEFAULT_HALF_LIFE_DAYS = 14.0`) and the persona-vocabulary-file cache in `engine_files.py`, with the half-life tests.
- **S1.** `engine_utils.py`: `day_of`, `fold_entry`, `fold_event`, `close_days`, `apply_climax_reset`, `run_tracker`, `shift_for`, `fade`, `counted_kind`, `source_modifier`; and the rest of the engine constants (`EMOTION_SHORT_HALF_LIFE_H = 24.0`, `EMOTION_LONG_HALF_LIFE_D = 30.0`, the climax numbers, `EMOTION_VALUE_CAP = 10.0`, `EMOTION_FADE_FLOOR_FRACTION`, `EMOTION_REPEAT_WINDOW_H = 24.0`, `EMOTION_REPEAT_FACTOR = 0.5`, file and folder names). `HORIZON_IMPORTANCE_GAIN` is imported inside `faded_value` to avoid an import cycle. The quiet-day zero's value 0 and weight 1 are written once in the module, not as tunables. The 30 days of the rebuild window, the change-log retention, the snapshot and the start gate are all the long half-life, not separate constants; the lock, queue, drain and day close have no constants. S1 edits none of hanamorix's files.
- **S2.** The rest of `engine_files.py`, `read_json_walk`, `state_dir`, the validator, the stat-key cache, and the queue, change-log and switch I/O.
- **S3.** The hooks and the `fold` flag on `create` and `update`; `fold=False` in recovery's restore branch and the two migrators; the peer-trust stand-in; the conftest fixtures.
- **S4.** The drain and its call sites, `read()` with the snapshot, the shift and the refresh call, `status()`, `engine_rebuild.replay` and `catch_up`, and three store methods: `list_engine_window(since)`, `creation_stamps_since(since)` and `oldest_created_at()` (plain SELECTs, explicit columns, no bump).
- **S5, S6, S7.** Sections 3.1 and 3.3.

### 4.2 What waits on what

- **S0** needs nothing and lands first, so hanamorix's writers build against it.
- **S1** needs S0; **S2** needs S1; **S3** needs S1 and S2; **S4** needs S3; **S5** needs S1 and S3 (independent of S4; it may land beside S3). **S3 and S4 land in the same window**: S3 alone queues events that nothing drains until S4.
- **The safety net** needs nothing from the engine and lands before the first reader edit in phase B.
- **The access fixes** need S0 to S4 (the stub, the hooks and fixtures, the drain), and must not land between S3 and S4.
- **Readers phase B** needs S0 to S5 and the safety net's pin for each old branch, and rebases onto the access fixes for the files both touch (soul review and its test, `read_full_memory.py`, `search_memories.py`). Nothing in it waits for #385 or for hanamorix's weather-shift work.
- **S6** needs: S4; readers phase B (including the self-model tick and the threshold measurement, whose numbers are read before S6 merges); hanamorix's writer work (her `peer_trust` accessor); the fixes for the reads that stamp last-accessed without a showing: the recall-resonance read (access fixes), the merge-target pre-read (S5), `add_memory` (her vocabulary work), and the ingest and research reads (her issue); and #555 and #556 fixed. There is no date wait after those fixes land: a fix being merged does not mean every install has it, so a persona's first rebuild may still see old false stamps, which is accepted (section 6). The dream read is not a prerequisite; it waits for the dream rework.
- **Readers phase C** needs S6.
- **S7** needs S6, readers phase C, hanamorix's removal after the switch of her five old `aggregate_state` reader branches (in the dream, monologue trace, growth scheduler, creative-DNA and reflex-crystallizer code), her cleanup that takes `protected` out of her files, and her writer change for the extractor and reconcile. Her plan has no item that deletes those branches, so S6's PR body asks which of her PRs does it and when. `aggregate.py` is deleted last.
- **The phase C PRs (S6, readers phase C, S7) are flipped to ready only after hanamorix has answered** on the PR that introduced each interface change they depend on (4.7). Work and draft PRs go ahead before that.
- **#565 and her writer-side pin** "spelling variants pool as separate channels": #565 flips that pin, so whichever lands second edits the pin in its own PR and says so in its body.

### 4.3 Merge order

S0, then S1 to S5, then (the safety net and the access fixes, anywhere before what needs them), then readers phase B, then S6 (with the decay-pass removal), then readers phase C, then S7. hanamorix's writer PRs can land any time after S0 and before S6, in any order.

### 4.4 How each PR is run

- Each PR targets `main`. A prerequisite that has not merged is merged into the new branch with `git merge origin/<branch>` (never rebase or force-push) and work starts now; it is merged again when it changes, and `origin/main` is merged when it lands. The PR body names which PRs merge first.
- Before each PR: a collision check (`gh pr list`, `gh issue list`); the item's sub-issue of #240, assigned to ThinkerOfThoughts (created first if missing, listing what, why, files and dependencies); then an empty first commit, push, and the DRAFT PR opened before any code.
- A PR is flipped (`gh pr ready`, reviewer hanamorix) only when every CI job is green, including the new-tests-must-fail job, with nothing pending, on a branch that has merged `origin/main` after its prerequisites landed. hanamorix merges.
- One `Closes #N` line per issue in the PR description.
- A PR that adds, changes or removes a constant in `brain/dev_constants.py` re-approves the dev-constants snapshot in the same PR (`uv run python tests/unit/test_dev_constants_snapshot.py --approve`): S0's half-life constants, S1's engine constants and phase B's summary constants add entries; S6 removes some.
- Every new test that passes on the base code carries `@pytest.mark.no_base_failure(reason=...)` on the test function, with the reason it passes on base; `scripts/new_tests_fail_on_base.py --base origin/main --workers 0` lists the exempt tests.
- A Discord heads-up at each point where a PR's "needs" line names another PR that must land first.
- Every interface change is written in the body of the PR that introduces it, with its question (4.7).
- GitHub text (issues, PR bodies, comments) is shown to ThinkerOfThoughts before it is posted.

### 4.5 The plan PR

One draft PR whose description is the plan as a checklist of ThinkerOfThoughts's twelve PRs in build order (S0, S1, S2, S3, S4, S5, the safety net, the access fixes, readers phase B, S6, readers phase C, S7), each ticked with a link to the PR that did it. It lists only this half and links hanamorix's branch and documents; the other lanes' work an item needs is in that item's "needs" line (4.2), not an item. It carries this document, `docs/DESIGN-INVARIANTS.md`, and the synthetic database with its build and check scripts in `test-data/synthetic-db/`, all scrubbed of personal details and shown to ThinkerOfThoughts as exact files before the push; ThinkerOfThoughts also sends hanamorix the database directly. #240 points to it. No build step waits for it to merge (the timings use a local copy of the database), and when it merges is not set.

### 4.6 Answers to hanamorix's questions about the engine interface

1. The hook is in `create` and the gate's promotion path, dated by the row's `created_at`: yes, as designed.
2. Backfilled emotions fold in: yes, as the "assigned" event.
3. `update_baseline` is kept: yes, in `felt_time/weather_shift.py`, where her plan keeps it; nothing in this rework moves it.
4. The interior-continuity block gets digests after the switch: accepted; the monologue systems are reworked in a later phase anyway.
5. `read()` calls `persona_loader.refresh_if_changed` before its registry filter: yes (2.7).
6. `canonical_name` pooling in `aggregate.py` now: patched by the bug-fix work (#565).
7. The x10 fix uses the switch time, with no recorded cutoff: yes (2.12).

#125 was reassigned to hanamorix with a note.

### 4.7 Interface changes and questions, by PR

Each item is written in the body of the PR that introduces it. A question here is what that PR asks hanamorix; a phase C PR is flipped only after her answer.

- **S0.** The engine is `brain/emotion/engine.py`; calls take the receiver form `engine.engine_for(persona_dir).<call>` (the earlier interface listed them without a persona); import the module so one test patch reaches every site. `read()` returns four values per feeling, `Reading(short, long, difference, shift)`, only for names in the vocabulary registry. **Question: does any caller of hers unpack `read()` as three values?** New: `status()`, `note_active(now)`. `oldest_entry_at()` has no reader any more and stays. The stub records calls in `recorded_calls`. `half_life()` is the real lookup from S0, so a blend her vocabulary work mints gets a real half-life.
- **S1.** The built-in half-life table is `EMOTION_HALF_LIFE_DAYS` in `dev_constants.py` from S0, so the table's literals in her vocabulary files (`_BASELINE`, `_canonical_personal_emotions.py`) leave with her vocabulary work; S1 edits none of her files. The line in `cli.py` that prints a growth-log event's own `decay_half_life_days`, and the persona-entry field `decay_half_life_days` in `persona_loader.py`, are hers and stay in phase B because the old decay pass still reads the field. **If the build finds the table move forces an edit in either, S1's PR body asks her for it.**
- **S2.** The engine's files are under `<persona_dir>/state/`; the switch is `state/emotion_switch.json` written with `write_json` (not `save_with_backup`); the state is read with the no-heal walk `read_json_walk` (not `attempt_heal`).
- **S3.** `create(..., fold=True)` and `update(memory_id, *, fold=True, **fields)`; hooks fire after the commit and never raise. **Question: is she content with the `fold=False` edits in her files (`recovery/engine.py` and the two migrators)?** Ingest passes `fold=False` only if she departs from the shared plan (the default folds). Peer trust is read through `engine_files.peer_trust`, 0.0 until S6 wires it to her accessor, expected to open `KindledLinkStore(kindled_db_path(persona_dir), integrity_check=False)` and return `get_relationship_state(...).trust_score`, 0.0 for no database or no row. `tests/conftest.py` gains `emotion_engine_stub` and the registry-reset fixture. From S3 every counted `create` in a file-backed test writes `state/emotion_pending.json`, and an uncounted one writes a start-only event there when the stored start date is null or later.
- **S4.** Hooks only append; folding happens in drains after each reply and in the supervisor pass. **The per-reply drain in the CLI chat without the bridge is ThinkerOfThoughts's edit in `cli.py`: is that fine?** `read()` calls `persona_loader.refresh_if_changed(<path of the persona vocabulary file>)`, expecting one path argument and a None return, in its own `try`. **Question: does her function match that signature, and does it re-read only a vocabulary file this process has already loaded?** `store.py` is shared: S3's `fold` flag, S4's three methods, S5's `refresh_faded_emotions` and her `list_recent_emotions` are plain additions, and whichever lands second rebases.
- **S5.** The gate merge now writes averaged emotions (`fold=False`); the temp-gate write-back resets `created_at` and writes faded values for every recalled type, so a stored value means "as of the last time this memory was written or remembered".
- **S6.** `ensure_engine_ready` runs in `_dream_handler` (her dreams) and five other CLI handlers, with the dry-run rule. The x10 fix multiplies her writers' old rows with no cutoff, which relies on her nudge-scale change being behind `is_live()`. **Question: is it?** The decay pass is gone. **Question: will her per-memory emotion readers (grief, and the dream average over member memories) read through `faded_value`?** The peer-trust stand-in is wired to her `peer_trust(persona_dir, peer_id)`. **Question: which of her PRs deletes her five old `aggregate_state` reader branches after the switch, and when?** (S7 waits on it.)
- **S7.** `aggregate.py` is deleted (her old branches must be gone first); the emotion-only row writes in her `chat/extractor.py` and `self_model/reconcile.py` are removed; `protected` leaves `store.py` (after her cleanup takes it out of the migrator transform and the recovery source reader); `GATE_BYPASS_TYPES` loses two entries (ingest imports it).
- **Readers phase B.** `_mem_to_result` gains a required `persona_dir`; `tools/impls/_common.py` is shared with her `_write_gate_check` and `_calc_importance_from_emotions`. **Question: how is it ordered with her writer PRs?** `faded_value` is available for her per-memory readers. `update_baseline` and `Baseline` stay where her plan keeps them.
- **Access fixes.** The one door reports every full open to the engine; the background reads in her files are in her issue; vocab repair's search and the `add_memory` read are her vocabulary work.
- **Safety net, readers phase C.** No interface change for her.

---

## 5. Design invariants check

Checked against `docs/DESIGN-INVARIANTS.md`.

| Invariant | How this design meets it |
|---|---|
| I1 one memories database | No new database: the engine's state is files under `state/`. |
| I2 per-memory attributes on the row | Emotions stay on memory rows; the write-back updates the row; the snapshot holds no memory id; the repeat map and the queue are keyed by memory id, not by content. |
| I3 derived thresholds | The shift rule has no tolerance or threshold, only day counts and the long half-life. The new constants are chosen design values (the half-lives, the climax numbers, the repeat factor, the x10 "below 1"), not fitted thresholds. Existing reader thresholds are unchanged and measured. The per-emotion half-lives are seed values, with #564 tracking half-lives that change over time. |
| I4 semantic first | Recall ordering is unchanged; the access fixes' two fields feed no ranker. |
| I5 opened memory gets the full bump and reappraisal | Every full open goes through the door: one `get(bump=True)` plus the reappraisal enqueue and the engine report. |
| I6 bulk work off the hot path | Drains run after the reply, off the event loop; the recovery rebuild runs at a quiet moment; up to 8 showing hooks per reply is a bounded per-turn step. |
| I7 constants and strings outside code | Engine and reader constants in `dev_constants.py`; new strings in `prompt_strings.toml`. |
| I8 no emotions store; crash-recovery log only | The state holds averages, not entries; the change log is read only by a rebuild or a catch-up, and phase B writes none. |
| I9 legacy import and run-once verified migration | The x10 fix has a run-once marker, one transaction and a row-count rollback; the first rebuild checks before it replaces anything; migrators and recovery keep working with `fold=False`; the `protected` column stays. |
| I10 persona strings | No persona wording is final; every string is listed in section 7. |
| I11 live persona never a fixture | Tests use temporary persona dirs; the timing script refuses paths under the app home; live tests use the synthetic user. |
| I12 no added user burden | The start-up step, the x10 fix and the first rebuild run automatically; a CLI dry run on a never-started persona asks for one normal start. |
| I13 cross-platform | The stat key works on all three platforms; `day_of` drops events Windows cannot convert; tests use local civil dates without `tzset`; the timing script runs through `uv run`; CI runs on Ubuntu, macOS and Windows. |
| I14 support machinery | The queue and drain, the rebuild and start-up, and the gate work each have slightly more support lines than design lines (7/8, 13/16, 4/5); they are ordinary unit tests of new code and were kept. The shift rule's support was cut to one simulation and a live test. The safety net (12 design to 8 support) and the access fixes (6 to 4) are under parity. Any new support line at build goes through the same check. |

---

## 6. Accepted edge cases

Each gets a code comment at the place named and a note in the commit message that adds it. "Issue" means an issue is also filed when the PR that introduces the edge merges (its text shown first).

| Edge | Where the comment goes | Slice |
|---|---|---|
| An install's first rebuild may count memories whose `last_accessed_at` was stamped without a showing (old resonance stamps, the reads in hanamorix's issue until fixed, dream dry-run or denied-slot reads). A fix being shipped does not mean it is installed, so a guard period cannot help. | directly above `rebuild`'s `replay(...)` call | S6 |
| A lost queue entry: a failed append, a raising hook, a crash between commit and append, a dropped invalid event; at most one turn's worth, and the memories from that turn still fold when committed | the hook append | S3 |
| An unparseable queue file is discarded, with no rebuild | the queue read | S2 |
| x10: a backfilled row whose strongest new value is below 1 is multiplied once (one odd day at most, then it sinks back) | `nudge_scale_fix.py` | S6 |
| x10: a channel the old decay pass deleted at 0.01 stays deleted | `nudge_scale_fix.py` | S6 |
| x10: `memories.db` restored alone under a surviving switch keeps unfixed rows | `nudge_scale_fix.py` | S6 |
| Readers of `created_at` see a recalled memory as new | no comment | S5 |
| The oldest snapshot day is a pending first day against the run and the next day is also against: that day's kind cannot be read (it has no previous day), so it is read as a move the other way, and the new run starts on that oldest day with length 2 where a zero first day would have started it a day later with length 1; a bar can move by 1. No issue | `run_tracker`, at the seeded-pending branch | S1 |
| An absent comfort_seeking or rest_need gets no climax +2 | `apply_climax_reset` | S1 |
| A clock set forward: day closes freeze until real time catches up, and change-log files pile up meanwhile. Issue (telling a corrected clock from a long absence needs a new rule or a chosen number) | `close_days` | S1 |
| An entry dated in an already-closed day: the day holds both its zero and the entry (a late-gated memory; a hook still resolving or waiting for the lock at midnight) | `close_days` and `fold_event` | S1 |
| A rollover is noticed only at the next supervisor pass; a read in between lacks the closed day's zeros | the supervisor drain call | S4 |
| Events with the same time fold in read order; catch-up and live can differ in a tie | `replay` | S4 |
| Catch-up: a late-dated queued event narrows the replay window; an append that straddles a save is missed. Issue | `catch_up` | S4 |
| A catch-up holds the lock while it replays, so a post-reply drain may wait | `catch_up` | S4 |
| A crash between the state write and the queue delete applies those events twice | `note_active` | S4 |
| A hook waiting during a rebuild or catch-up folds its row a second time (and a commit's climax reset again) | the hook append | S4, S6 |
| A day whose only rows were refreshed by the write-back counts as off in a later rebuild | `replay`'s past-day scan | S6 |
| Missing or corrupt change-log parts lose those changes at a rebuild | the change-log replay | S6 |
| A change still queued at a rebuild with no log line (a failed append, or from phase B) is lost | `rebuild`'s queue discard | S6 |
| The first rebuild has no chat-change history (phase B writes no change log) | `rebuild`, at the change-log replay | S6 |
| A rebuilt persona's snapshot is a fresh 30-day history; flags after a rebuild or catch-up may differ | `rebuild` | S6 |
| Repeat showings beyond the first within 24 hours are not restored by a rebuild | `rebuild` | S6 |
| A recovery rebuild with no readable old state has only `MIN(created_at)` for `persona_start_at`, which write-backs move forward: the gate can only stay shut longer, by at most 30 days. No other source exists (ids are random; the switch time would open a new persona's gate early). No issue | `rebuild` | S6 |
| The spare rule's clear drops a cleared gap's reconcile cooldowns, so a just-reconciled channel can come back next tick. Issue | `_self_model_reflect` | readers phase B |
| Readers show absent while a recovery rebuild is pending | the reader sites | readers phase B |
| A hand-edited reflex arc with threshold 0 no longer fires on a missing emotion | `reflex._trigger_met` | readers phase B |
| A crash between the door's bump commit and the showing append loses that showing | `recall_open.py`, at the `try` around the engine call | access fixes |

A new persona whose first rows are of an uncounted type would have opened its gate late; that is fixed, not accepted (the start-only event, 2.8).

---

## 7. Persona-facing wording left to the end

Working text only while building; the final wording is ThinkerOfThoughts's byte-exact call at the end of the build, with no LLM tells.
1. The mood summary: the `current emotions:` label (prompt and monologue frame), the `name:value` form, and the `long_short` and `long_difference` renderings.
2. The body block header, the energy, temperature and exhaustion line, and the `body emotions:` label; it differs from item 1 on purpose.
3. The self-model block: the four (position, direction) fragments, the zero-difference and missing-direction forms (`self_model.ambient`), and the block's framing and invitation text, including its quieter-than-usual, stronger-than-usual, rising and fading wording.
4. Articulation: the direction labels (`self_model.articulate`) and the prompt segments, which must say how to read the labels.
5. The resolution memory text: whether it still fits now that it fires when a feeling's run ends.
6. Tools: the `get_emotional_state` description and empty-return text, `format_emotion_summary`'s line form, and `get_body_state`'s "the six body emotions".
7. Research: the "How you feel right now" header and "Your current emotional state:".
8. The maker and notes `peak intensity` line; soul review's "neutral" and "unknown".
9. The heartbeat prompt segments built around `memories_decayed` and the system segment's "decay applied", which S6 edits when it removes them.
10. The live-test variants of the summary and the block framing (section 8).
11. The research rating line: hanamorix writes the rating code; the wording of the prompt line that asks for the rating is ThinkerOfThoughts's.

Not persona-facing: the CLI dry-run message, the soul-review docstring sentence, and code comments.

---

## 8. Live tests after the build

Run with the synthetic test user and a test persona, never a live persona, once the code exists:
1. **Mood summary variants:** top 3 against top 5; long against long and short against long and difference; all emotions, both ways. Switched by `EMOTION_SUMMARY_COUNT` and `EMOTION_SUMMARY_MODE`, with no code edit.
2. **The self-model block's framing and prompt**, decided once the machinery is built.
3. **The shift rule's effect on its readers:** how often the block renders, how often `reconcile_self_read` is offered, articulation calls per day (with direction), resolution memories, and how often a flagged feeling's rising or fading label disagrees with the average over the run; compared with the sizing numbers of 2.6. The maker's discharge rate is measured in the phase B threshold script instead.

---

## 9. Checks at build time

- Every file and line reference is re-read at the build's `origin/main`.
- Each item's #240 sub-issue exists before its PR opens.
- Which new tests pass on base and so carry `no_base_failure`, and the dev-constants snapshot in each PR that touches a constant.
- An import cycle check: `utils/memory.py` and `initiate/user_pattern.py` importing `ingest/buffer.py`.
- #555 and #556 fixed before S6.
- The first rebuild's timing in the S6 PR (3.3).
- A writer survey: every writer whose emotion constants are below 1. Another such writer changes the x10 type list.
- A record that the memory-database rework must keep `PRAGMA user_version`.
- A grep of readers of `score` for a "10 at most" assumption; a hit is fixed in the change.
- One grep of engine readers on the memory-backfill-extraction and interest-list handlers.
- The Ollama reachability check (no design effect).
- The shift-rule sizing rebuilt from its method and run with the run record as built, reproducing the numbers of 2.6 up to the oldest-day edge (S4).
- Before the felt-time edit in phase B: a collision check with #385, #386 and #387 and open PRs touching `prompt.py`, `supervisor.py`, `schemas.py` and `dev_constants.py`; the spike emitter's comment cites #569.
- Safety net: re-run the 24 cells, the 17 new pins and the bypass pin at the build commit (and again after #565 lands) and set the marks to what is observed; take the record timing run on the low-end host with its AVX2 line; confirm the cells pass under `pytest -n 2 --dist loadgroup`.
- Access fixes: whether the tracking issue already exists; the stub's `recorded_calls` entry shape at S3; a grep of `recall_count` and `last_accessed_at` readers for a ranker (if one exists, the "feeds no ranker" reasoning is redone); one timed loop of 8 `on_memory_shown` calls if the engine records no hook timing (PR text only); `new_tests_fail_on_base.py` clean on the branch after `origin/main` is merged in.
- S1 and hanamorix's files: read the `cli.py` line and the persona-entry field (4.7, S1).

These come back to ThinkerOfThoughts only if they happen: another sub-1 writer is found (the x10 list changes); hanamorix's nudge-scale change is not behind the switch (the "no cutoff" rule fails); the cold first rebuild on the low-end host is materially above about 1.2 s; or the half-life table move forces an edit in her files and she declines it on S1's PR.

---

## 10. Corrections to the shared plan

Where the shared emotion rework plan says otherwise, this design holds. Any correction that changes an interface hanamorix builds against is also stated in the PR that introduces it.
- The switch-over time is its own file, `state/emotion_switch.json`, not a field of the state file.
- The state file holds the average and weight pairs, `saved_at`, `reset_at`, `persona_start_at`, `oldest_entry_at` and the day-close snapshot. There is no fill marker: the switch file is the first rebuild's run-once marker. The 24-hour list of memories shown in full is the id-keyed repeat map.
- The averages do not hold still on a day the app ran: each present emotion with no entry that day gets a zero entry in both averages at the day close. Only whole off calendar days leave them as they are.
- There is no 24-hour rule: the shift is off until 30 days after the persona's oldest memory, which affects a new persona only.
- The first rebuild finishes before the persona starts; a recovery rebuild runs at a quiet moment.
- Hooks append events to `state/emotion_pending.json` and a drain after the reply folds them; one lock guards the engine's four files.
- `update_baseline` stays in `felt_time/weather_shift.py`; nothing in this rework removes it. The "prompt comments" the plan mentions are fixed in readers phase C.
- The phase C removals are the lists of S6, S7 and readers phase C, without `update_baseline`.
- There is no recorded cutoff for the x10 fix; it runs before the switch.
- The engine interface said `read()` returns three values per feeling; it returns four.
- hanamorix writes the research rating code; the wording of its prompt line is ThinkerOfThoughts's, at the end.

---

## 11. Not in this rework, and follow-up issues

Filed and related:
- #549 the recall count never fades; #550 memories surfaced by length.
- #555 `close_session` unlinks without archiving, and #556 the weekly rollover gap (fixes in #562 and #563; S6 waits for them); #565 `canonical_name` pooling.
- #557 (high), #558 (medium), #559 (low): audit bycatch. #571: the meaning-search candidate pool loads every memory in full on every chat turn. #572: the conversation-memory listing reads the whole table even when the caller wants ten.
- #548 wizard fixture names; #125 reassigned to hanamorix; #382 one integrity check at start-up and every 24 hours; #388 backwards compatibility in one place (the x10 file is flagged for it); #385 paused (it gets the note about `read_json_walk`); #386 the felt-time drivers (the unplugged `emotional_intensity` waits for it); #240 the umbrella.

Follow-up issues the plan called for, all filed:
- #564 a way for each emotion's half-life to change over time;
- #566 nudge values (and the chat nudge) calculated rather than hardcoded;
- #567 the body emotions and body state as a whole: the arousal tier and influence hints, climax as a build-up toward a threshold, and the bug that the climax reset sets climax to 0 so the avatar's climax expression and the body state's climax checks can never fire;
- #568 attunement falsification;
- #569 the journal rework and the spike-detector emitter;
- #570 moving the other persona state files into `state/`;
- a comment on #180 about the expression vector and custom sprites.

Not this build: the felt-time rework (#386); the pass-2 extractor review (a later phase); the dream read's stamp fix (the dream rework); switching the user-presence code to `last_user_turn_ts`, and a persona-dir cleanup (including the presence file's other fields). Small findings made while designing (a default-bump read in `initiate/review.py`, a fallback read bump in `heartbeat.py`, and others) are recorded for the hand-off and not fixed here.
