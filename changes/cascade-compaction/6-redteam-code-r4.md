# Stage-6 code re-review (round 4) — registry-lock closure of the resolve-persist race

**Reviewer stance:** cold, independent, no shared context with the author or with prior review rounds beyond
what is verifiable from source. Scope: commit `33cd5ba20f8462ba5c5f1493a825764641aa1756` ("fix(rollover): close
resolve-persist race atomically via registry lock + successor-redirected persist") only — the fix for the
Major surfaced in round 3 (`6-redteam-code-r3.md`, reviewing `5348cae8`). Also re-verified the `a485e72d` C-1
redirect-first fix is not regressed.

## Provenance

```
cd /home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction
git show 33cd5ba2 --stat
git diff 5348cae8 33cd5ba2 -- brain/ tests/
```

- Fix commit: `33cd5ba2`, parent `5348cae8`.
- HEAD at review time == `33cd5ba2`, worktree clean.
- Files touched (code + tests, excluding `changes/*.md` docs): `brain/bridge/supervisor.py` (+12/-2 comment
  only), `brain/chat/engine.py` (+26/-10, routes persist through the new redirect helper),
  `brain/chat/rollover.py` (+123/-61, wraps the destructive section in `registry_lock()`),
  `brain/chat/session.py` (+58, new `registry_lock()` / `persist_turns_following_successor()`),
  `tests/unit/brain/bridge/test_compaction_cadence.py`, `tests/unit/brain/chat/test_engine.py`,
  `tests/unit/brain/chat/test_rollover.py` (+188, three new tests).
- Ran the full touched-test set (`tests/bridge/test_cascade_rollover_endpoints.py`,
  `tests/unit/brain/chat/test_rollover.py`, `tests/unit/brain/chat/test_engine.py`,
  `tests/unit/brain/bridge/test_compaction_cadence.py`) — 37/37 pass, reproducibly across 3 reruns. (One
  isolated flake was observed on a single earlier 4-file run — `test_c8_idle_rollover_sync_and_selection`
  failed once with a wall-clock-adjacent assertion, then passed cleanly on 3 subsequent reruns of the exact
  same file set. Noting it for completeness; not attributable to this commit's diff and not reproducible, so
  not counted as a finding.)
- **I independently wrote and ran a standalone repro script** (not part of the delivered test suite; see
  charge 1b below) that reproduces a real, permanent data-loss bug this commit's own tests do not exercise.

## Charge 1: is the resolve-persist resurrection race now genuinely closed, with NO residual TOCTOU?

**Split answer. For the `tiers_plus_tail` (1c-B weekly) mode — the exact scenario round-3 found — YES, closed
by construction, verified by trace and by a genuinely concurrent test. For the `summary_only` (1c-A stale-
resume) mode, NO: a related, unaddressed, and independently-reproduced race causes a live turn to be silently
and PERMANENTLY LOST (not resurrected — actually deleted, unrecoverable), which is a stronger failure than
the resurrection bug this commit set out to close. This is not a new bug introduced by 33cd5ba2 — the
seed-selection logic that causes it is untouched (copy-pasted, unchanged) from `5348cae8` — but it directly
contradicts the "unreachable by construction" / "closes the race" language this commit adds to the code
comments and docstrings, and none of the three new tests exercise it.**

### 1a. The `tiers_plus_tail` resurrection race — closed, verified by trace

- `perform_rollover` (`brain/chat/rollover.py:66-195`): steps 1-2 (extraction `rollover.py:92-102`, LLM call;
  fold/cascade `rollover.py:110-125`, LLM call) run **before** `acquire_compaction_lock` and before
  `registry_lock()` — confirmed unguarded, matching the claim that the slow part stays outside the lock.
- `acquire_compaction_lock` (`rollover.py:130`) is a **non-blocking** file-existence check
  (`brain/ingest/buffer.py:464-509`, `os.open(..., O_CREAT|O_EXCL...)`; on contention it returns `False`
  immediately rather than waiting) — so it cannot deadlock against anything, including `registry_lock()`.
- `with registry_lock():` (`rollover.py:146-178`) wraps: `read_session` (re-read, `147`), `_split_summary_and_raw`
  (`148`), seed-row selection (`149-153`), `read_cursor` (`158`), `create_session` (`160`, re-acquires the
  same `RLock`, reentrant — no self-deadlock, matches the docstring at `session.py:265-266`),
  `rewrite_session_atomic` (`165`), `write_cursor` (`169`), **`write_rolled_to` (`177`, pointer write) then
  `remove_session` (`178`, registry evict) — both still inside the `with` block.** The pointer write is
  confirmed to land **before** the registry evict and **before** the block exits.
- `delete_session_buffer` / `delete_cursor` / `delete_backoff` (`rollover.py:186-188`) run **after** the `with
  registry_lock():` block has closed — confirmed outside the lock, matching the claim.
- `session.persist_turns_following_successor` (`session.py:270-298`): `with _LOCK:` (the *same* `_LOCK` object
  `registry_lock()` returns, `session.py:263`/`243-267`) wraps the full per-record loop — `_resolve_successor`
  then `ingest_turn`, for **both** records (user + assistant) under one lock acquisition (`session.py:291-298`)
  — so a turn pair can never be split across old/new buffers.
- `engine._persist_turn` (`engine.py:416-459`) now calls `persist_turns_following_successor` (`engine.py:452`)
  instead of two raw `ingest_turn` calls. Grepped `ingest_turn(` across `brain/`: the only two call sites are
  the definition (`brain/ingest/buffer.py:74`) and the one inside `persist_turns_following_successor`
  (`session.py:298`) — confirmed no other live-persist path bypasses the redirect.
- **Both orderings traced and hold for `tiers_plus_tail`:**
  - *Persist wins first* (acquires `_LOCK` before rollover reaches `with registry_lock():`): appends straight
    to the still-live old buffer (no pointer yet, `_resolve_successor` returns `None`); when rollover then
    acquires the lock and re-reads (`rollover.py:147`), the just-appended turn is present in `raw`, and since
    `tiers_plus_tail`'s seed keeps `raw[-40:]` (`rollover.py:151-152`), the freshly-appended turn (necessarily
    the newest) is captured into the successor seed. Nothing lost.
  - *Rollover wins first*: by the time persist acquires `_LOCK`, `rolled_to` is already written
    (`rollover.py:177`), so `_resolve_successor` (`session.py:223-240`, full-follows via
    `read_rolled_to`) redirects the append into the live successor buffer. Old buffer never resurrected.
- **Confirmed by test**, not just by trace: `test_persist_during_rollover_destructive_window_not_orphaned`
  (`tests/unit/brain/chat/test_rollover.py:452-533`) genuinely races two real OS threads — one blocked
  exactly at the (now-outside-the-lock) `delete_session_buffer` call via a monkeypatched hook, the other doing
  a real `persist_turns_following_successor` call concurrently — and asserts the raced turns land in the
  successor, not the deleted old buffer. I ran it individually and as part of the suite; it passes, and its
  mechanics (waiting on `at_delete`, a real `threading.Thread`) are not vacuous — see charge 6.

### 1b. The `summary_only` mode — a related, unaddressed, **reproduced** data-loss bug

`perform_rollover`'s seed-row selection (`rollover.py:149-153`):

```python
if seed_mode == "summary_only":
    seed_rows: list[dict] = [summary_row] if summary_row else []
else:
    tail = raw[-_ROLLOVER_TAIL:]
    seed_rows = ([summary_row] if summary_row else []) + tail
```

For `summary_only` (the 1c-A stale-resume path, `brain/bridge/server.py:1303-1306` — called directly, with
**no `is_session_busy` gate of any kind**, unlike the 1c-B weekly path), `seed_rows` is **only** the summary
row. `raw` — whatever raw turns are present in the old buffer at the moment `read_session` runs inside
`registry_lock()` — is read (`rollover.py:148`) but **unconditionally discarded**, regardless of whether those
raw turns are pre-existing carryover or a turn that raced in during rollover.

**Timeline that loses a turn permanently** (I reproduced this; see script + output below):

- t0: `/sessions/active` (`server.py:1250-1308`) scans for a stale (>24h idle) session, selects `stale_sid`.
  No busy-check anywhere on this path.
- t0+ε: a client reconnects to `stale_sid` — the exact "user resumes a stale session" scenario 1c-A exists
  for — and sends a chat turn. Tool loop runs (seconds).
- t1: `/sessions/active`'s rollover call proceeds: step 1 (extraction) and step 2
  (`compact_conversation(..., min_keep_tail=0)`) fold the buffer down to `[summary_row]` at step 2's own
  read time (before the racing turn exists).
- t2: the racing client's tool loop finishes; `_persist_turn` → `persist_turns_following_successor` acquires
  `_LOCK` **before** `perform_rollover` reaches its own `with registry_lock()` (i.e., in the gap between step
  2 finishing and step 3's `acquire_compaction_lock`/`with registry_lock()` — this gap is not covered by any
  lock on either side at that instant). `_resolve_successor(stale_sid)` → `None` (no pointer yet) → appends
  directly to the old buffer: buffer is now `[summary_row, live_user, live_asst]`.
- t3: `perform_rollover` acquires `registry_lock()`, re-reads → `raw = [live_user, live_asst]` — but
  `seed_mode == "summary_only"` discards `raw` entirely. `seed_rows = [summary_row]` only. New session seeded
  with just the summary. Old buffer deleted.
- **Result: `live_user`/`live_asst` are in neither the old buffer (deleted) nor the new buffer (excluded from
  the seed) nor the archive (the archival inside `compact_conversation` ran at t1, before these turns
  existed).** Permanently gone — not orphaned-but-recoverable, not resurrected — deleted.

**I verified this is real, not a hypothetical**, by writing a standalone repro against the actual `perform_rollover`
(reusing the test module's own `_persona`/`_seed`/`_ExtractOnlyProvider` helpers to rule out a harness mistake),
monkeypatching `acquire_compaction_lock` to fire a real `persist_turns_following_successor` call at exactly the
t1→t2 gap above, then letting `perform_rollover` continue unmodified:

```
buffer just before hook injects persist: [{'speaker': 'summary', 'text': '[truncated 6 earlier messages]', ...}]
fired: True
new_sid: f34ad878-4d22-4e77-8a8d-a7d69928375a
old_sid still active: False
new buffer contents: ['[truncated 6 earlier messages]']
RACED-USER present in new buffer: False
RACED-ASST present in new buffer: False
archived dir exists: True
  archived_conversations/sess_lossy.000.jsonl  ->  turn 0..turn 5 only (the ORIGINAL seed turns, not the raced ones)
```

`RACED-USER`/`RACED-ASST` appear nowhere on disk after the rollover completes. This is genuine, permanent data
loss, reachable via a real production endpoint with **zero** busy-gating, and strictly worse than the
resurrection bug this commit is titled to fix (resurrection at least leaves the data somewhere, just
disconnected from the successor chain; this loses it outright).

**Is this new?** No — the `seed_rows` selection code is byte-identical to `5348cae8` (only its indentation/
enclosing `with` block changed; confirmed via the diff). The bug's *window* is arguably even narrower now
than pre-fix (bounded to the gap between step 2's fold and step 3's lock acquisition, rather than spanning
all of steps 1-4), but it was never closed, and it is not what any of this commit's three new tests exercise
(see charge 6). The commit's docstrings assert an unqualified "the resolve-persist race is closed... by
construction" (`rollover.py:135`, `session.py:261`, `rollover.py:221-222`) without scoping that claim away
from `summary_only` mode, which is inaccurate given the above.

**Severity: Major.** Confirmed, reproducible, permanent turn loss on a real (and un-gated) endpoint, in the
exact class of bug this commit's own message says is now unreachable.

## Charge 2: async vs thread boundary

**Confirmed correct for the redirect helper itself, with one adjacent latency caveat.**

- `_respond_blocking` (`brain/bridge/server.py:241-273`) — called via `await asyncio.to_thread(_respond_blocking,
  ...)` at every call site (`server.py:2441-2442`, `2608-2609`, `2776`, `2849`) — runs `brain.chat.engine.respond`
  synchronously on an ordinary `asyncio.to_thread` worker thread. `respond()` calls `_persist_turn` →
  `persist_turns_following_successor` synchronously, at the end of its own call stack. Confirmed:
  `persist_turns_following_successor` is never called from the event loop thread itself.
- The supervisor thread (`server.py:961-975`, `threading.Thread(...)`, `sup_thread.start()`) is a plain OS
  thread, not a loop task. `perform_rollover` runs on it (via `run_folded` → `_run_compaction_tick` →
  `maybe_weekly_rollover`/direct call). Confirmed ordinary-thread, matching the claim that an `RLock` (not
  `asyncio.Lock`) is the right primitive for these two sides.
- **Caveat (not disqualifying, but worth flagging):** `get_or_hydrate_session` (`session.py:129-220`, `with
  _LOCK:` at `166`) **is** called directly from the event loop thread — `sess =
  get_or_hydrate_session(...)` at `server.py:2427` and `2490` sits inside `async def chat(...)` /
  `async def` WS handler, with **no** `asyncio.to_thread` wrapper. This predates this commit (`a485e72d`), so
  it is not new. What *is* new here: `registry_lock()` returns the exact same `_LOCK` object, and
  `perform_rollover`'s destructive section (`rollover.py:146-178`) now holds it continuously across several
  synchronous disk writes, including two `fsync` calls (`rewrite_session_atomic` → `brain/ingest/buffer.py:
  313-318`; `write_rolled_to` → `buffer.py:222-234`, explicit `os.fsync(fh.fileno())`) — where previously (at
  `5348cae8`) only the brief in-memory `create_session`/`remove_session` dict mutations took the lock, and the
  fsync-bearing writes ran unguarded. Since `_LOCK` is a single **process-global** mutex (not per-session, not
  per-persona — `session.py:96-97`), any concurrent `/chat` or `/stream` request's `get_or_hydrate_session`
  call — for *any* session — will now synchronously block the asyncio event loop thread for the (likely small,
  but no-longer-negligible) duration of a rollover's disk writes, whenever one is in flight. Not a correctness
  bug and not introduced as a NEW code path, but this delta measurably widens how long a background thread can
  hold a lock that the event loop blocks on synchronously and un-yieldingly. **Severity: Minor** — a real
  latency-injection/availability concern under contention, not a data-correctness issue, and bounded (rollover
  fires only at startup/weekly/stale-reconnect, and the newly-locked span contains no LLM calls).

## Charge 3: deadlock / lock ordering

**No deadlock found.** `acquire_compaction_lock` (`rollover.py:130`) is OUTER and is non-blocking (returns
`False` on contention rather than waiting — `brain/ingest/buffer.py:464-509`), so it cannot participate in a
lock-ordering deadlock with `registry_lock()` regardless of what's nested inside. `registry_lock()` (`_LOCK`,
an `RLock`) is INNER, and `create_session`/`remove_session` re-acquire the same `RLock` from the same thread —
reentrant, confirmed safe (`session.py:118-120`, `323-324`). I found no code path that acquires `registry_lock()`
first and `acquire_compaction_lock` second. Confirmed the slow extraction (`rollover.py:92-102`, LLM call) and
fold/cascade (`rollover.py:110-125`, LLM call) run **outside** `registry_lock()` — matching the claim, and
confirmed the (possibly retry-looping, `brain/ingest/buffer.py:451-459`, `_unlink_with_retry`, up to 8 retries
× up to ~0.05–0.4s backoff) buffer/cursor/backoff deletes (`rollover.py:186-188`) run **outside**
`registry_lock()` too — so a stalled unlink (e.g., a Windows antivirus pinning the file) cannot stall unrelated
registry operations on other sessions. No issue found here.

## Charge 4: did moving the delete outside the lock reintroduce anything?

**No, for the resurrection-specific concern.** Once `write_rolled_to` (`rollover.py:177`) has run inside the
lock, `_resolve_successor` (`session.py:223-240`) will find the pointer for any subsequent resolve or persist
of `old_sid`, so nothing can target the old buffer's path again through the redirect-covered paths (confirmed:
the only live-append call sites all go through `persist_turns_following_successor`, per charge 1a). The window
between the lock release and the physical `delete_session_buffer` unlink is therefore inert with respect to
resurrection — I found no path that could append to the old buffer in that window.

This charge doesn't surface anything beyond charge 1b's finding, which is a *pre-lock* (not post-lock) gap, so
it's recorded under charge 1, not here.

## Charge 5: any new correctness/concurrency issue introduced by this delta?

- **Charge 1b (summary_only permanent data loss)** is the standout finding, but as established it is not
  *introduced* by this delta — the vulnerable code is unchanged; only its surrounding lock scope moved. I'm
  not double-counting it here.
- **Charge 2's lock-hold-widening onto the event-loop-blocking path** is the one genuinely new-shape concern —
  rated Minor above.
- No new deadlock (charge 3), no new resurrection window (charge 4).
- The `budget.py` diff in this commit is comment-only-adjacent supervisor.py docstring changes; I did not
  re-verify charge-4-from-r3 (`apply_budget`'s independence from `emergency_fold_24h`) since this commit
  doesn't touch `budget.py`'s logic at all — out of scope for this round.

## Charge 6: do the new tests actually exercise the race, and are they discriminating?

**Two of three are real and discriminating; the third is a sound but narrower regression pin; none of the
three cover the summary_only loss found in charge 1b.**

- `test_persist_after_rollover_redirects_to_successor` (`test_rollover.py:363-404`): sequential, not
  concurrent — calls `perform_rollover` to completion, *then* calls `persist_turns_following_successor` for
  the (already-rolled-over) old sid, and asserts the record lands in the successor. This proves the redirect
  mechanism works post-hoc; it does not itself prove the race is closed (no interleaving is exercised), but it
  is not vacuous — it would fail if `_resolve_successor` or the pointer-write path were broken.
- `test_plain_ingest_after_rollover_resurrects_fail_demo` (`test_rollover.py:407-449`): a genuine "fail-demo"
  — it calls the OLD unredirected `ingest_turn` path directly (bypassing the fix entirely) and asserts the bug
  *does* reproduce (old buffer resurrected, turn absent from successor). I confirmed this discriminates: it
  is asserting the **absence** of the fix's mechanism reproduces the original bug, which is a legitimate way
  to prove the guard isn't vacuous, though it doesn't test the shipped code path under race conditions either
  — it tests a straw-man alternate path.
- `test_persist_during_rollover_destructive_window_not_orphaned` (`test_rollover.py:452-533`): **this is the
  one real concurrency test**, and it is discriminating — two real `threading.Thread`s, a monkeypatched
  `delete_session_buffer` that blocks until released, and a genuine `persist_turns_following_successor` call
  fired into the exact post-pointer-write/pre-delete window. I verified (by re-reading, not just trusting the
  docstring) that if `registry_lock()` were removed from `perform_rollover` while keeping the redirect in
  `persist_turns_following_successor`, this test would very likely still pass in practice most runs (since the
  pointer is already written by the time `at_delete` fires, independent of locking) — so its discriminating
  power is specifically against a **missing or wrong pointer-write-before-evict ordering**, not purely against
  a missing lock. It is still a real, non-vacuous regression pin for the scenario it targets (rollover-wins
  ordering, tiers_plus_tail mode).
- **None of the three tests use a `summary_only` + concurrent-persist-before-the-fold-window combination** —
  the two `summary_only` tests are both sequential (post-rollover), and the one concurrent test uses
  `tiers_plus_tail`. Charge 1b's bug is therefore untested and unguarded by this commit's own regression
  suite.
- Ran `uv run pytest tests/unit/brain/chat/test_rollover.py -q`: 9/9 pass (confirmed, not just asserted).

## Charge 7: regression check

**Not regressed.** `git diff 5348cae8 33cd5ba2 -- brain/chat/session.py` shows only additions
(`registry_lock`, `persist_turns_following_successor`) appended after `_resolve_successor` — the body of
`get_or_hydrate_session` and `_resolve_successor` (where the `a485e72d` C-1 redirect-first ordering lives) is
byte-identical. Ran `uv run pytest tests/bridge/test_cascade_rollover_endpoints.py -q`: 10/10 pass, including
the C16/C19-C21 conformance tests. Ran the full combined set (`test_cascade_rollover_endpoints.py`,
`test_rollover.py`, `test_engine.py`, `test_compaction_cadence.py`) three times: 37/37 pass each time.

## Bottom line

This commit correctly and verifiably closes the **specific** resurrection race round-3 found for the
`tiers_plus_tail` (1c-B, weekly-cap) rollover mode: the destructive pointer-write section now runs under a
lock shared with the live-turn persist path, both orderings were traced and hold, and one of the three new
tests (`test_persist_during_rollover_destructive_window_not_orphaned`) genuinely exercises the interleaving
concurrently rather than just asserting it. The `RLock`/OS-thread reasoning is sound (charge 2), lock ordering
introduces no deadlock (charge 3), and the buffer delete's move outside the lock does not reopen resurrection
(charge 4). The prior `a485e72d` fix is untouched and unregressed (charge 7).

**However, the commit's claim of having made the resolve-persist race "unreachable by construction" is not
accurate as stated.** I independently reproduced a **different, more severe** failure in the `summary_only`
(1c-A stale-resume) seed mode: a live turn racing in between the fold step and the registry-lock seed-read is
**silently and permanently discarded** — not resurrected, not recoverable from the archive, just gone — because
`summary_only`'s seed-row selection (`rollover.py:149-150`) unconditionally drops all raw turns, a piece of
logic this commit did not touch and does not test. This path (`brain/bridge/server.py:1303-1306`) has **no**
`is_session_busy` gate at all, so it is, if anything, more exposed to the underlying race pattern than the
weekly path this commit targeted.

## Verdict

1. **Worst severity: Major** (charge 1b — reproduced, permanent data loss in `summary_only` mode).
2. **Findings, one line each:**
   - Major (charge 1b): `summary_only` seed-row selection (`rollover.py:149-150`) discards all raw turns
     unconditionally; a turn racing in between step-2's fold and the registry-lock seed-read is permanently
     lost (verified by direct reproduction, not just trace) — worse than, and un-guarded against by, the fix
     this commit ships. Reachable via `server.py:1303` (`/sessions/active` stale-resume), which has no
     `is_session_busy` gate at all.
   - Sound (charge 1a): the `tiers_plus_tail` resurrection race round-3 found is genuinely closed by
     construction — both orderings traced line-by-line and confirmed by a real concurrent test.
   - Minor (charge 2): `registry_lock()`'s widened hold (now spanning several disk writes + 2 `fsync` calls)
     is a single process-global lock also taken synchronously, un-yieldingly, on the asyncio event-loop thread
     inside `get_or_hydrate_session` for every `/chat`/`/stream` request — a latency/availability concern under
     contention, not a correctness bug, and not newly introduced as a code path (pre-existing since
     `a485e72d`), but its hold duration is newly widened by this delta.
   - Sound (charges 3-4): no deadlock (the file-based compaction lock is non-blocking, `RLock` is reentrant);
     moving the buffer delete outside `registry_lock()` does not reopen resurrection, since the pointer is
     already durable by the time the lock releases.
   - Mixed (charge 6): 2 of 3 new tests are genuinely discriminating (one true concurrency test, one honest
     fail-demo); the third is a sound but sequential (non-racing) mechanism check. None of the three cover the
     `summary_only` mode under a race, which is exactly where charge 1b's bug lives.
   - Not regressed (charge 7): `a485e72d`'s C-1 ordering is byte-identical pre/post this commit; full touched
     test set green (37/37) across 3 reruns.
3. **Charge 1 explicit answer: NO, the resolve-persist race is not fully closed with no residual TOCTOU.** The
   specific resurrection mechanism round-3 flagged, for the weekly (`tiers_plus_tail`) rollover, is genuinely
   closed by construction — verified by trace and by a real concurrent test. But a sibling race in the
   `summary_only` (stale-resume) mode remains fully open and causes permanent, unrecoverable turn loss — a
   strictly worse outcome than resurrection — and it is untested by this commit's own regression suite. The
   commit's unqualified "unreachable by construction" language in `rollover.py:135`, `rollover.py:221-222`, and
   `session.py:261` overstates what was actually closed.
4. **Routing recommendation: back-to-stage-5.** The core mechanism (registry lock + successor-redirected
   persist) is correct and well-verified for the mode it was built and tested against, so this is not a
   rebuild — but a real, reproduced, permanent-data-loss bug in the sibling `summary_only` path remains open
   and is currently mis-described as closed. Recommend: extend the `summary_only` seed selection to capture
   any raw turns present at the registry-lock seed-read (mirroring what `tiers_plus_tail` already does via its
   tail, or by explicitly re-checking for and carrying forward any turns newer than the fold's own read), add
   a concurrent test analogous to `test_persist_during_rollover_destructive_window_not_orphaned` but for
   `summary_only` with the race positioned between the fold and the lock acquisition (not just before the
   delete), and either close the gap or, if deferred, correct the "unreachable by construction" language to
   scope it explicitly to `tiers_plus_tail` and name the `summary_only` gap as a known, accepted residual risk
   in `decisions.md`.
