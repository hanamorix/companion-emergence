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

## Round-4 re-review (after fix commits 33cd5ba2 + 2e1bcf6f)

**Reviewer stance:** cold, independent, no shared context with the author or with any prior review round
beyond what is verifiable from source. Scope: the complete delta from `5348cae8` (the commit round-3
examined) through `2e1bcf6f` (HEAD), i.e. both `33cd5ba2` (registry-lock + successor-redirected persist) and
`2e1bcf6f` (carry residual raw turns in `summary_only`). This round verifies that `2e1bcf6f` actually closes
the `summary_only` data-loss Major this file's own round-4 review (above) found in `33cd5ba2` alone.

### Provenance

```
cd /home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction
git diff 5348cae8 2e1bcf6f -- brain/ tests/
git show 33cd5ba2 ; git show 2e1bcf6f
uv run pytest tests/unit/brain/chat/test_rollover.py tests/bridge/test_cascade_rollover_endpoints.py -q   # 20 passed
uv run pytest tests/unit/brain/bridge/test_compaction_cadence.py tests/unit/brain/chat/test_engine.py \
    tests/bridge/test_endpoints.py tests/unit/brain/chat/ -q   # 416 passed, 1 xfailed
```

### Charge 1 — is the r3 resurrection race now closed with no residual TOCTOU (both orderings, both seed
modes)?

Yes, for the resurrection mechanism specifically. Traced both orderings against the actual code:

- `perform_rollover` (`brain/chat/rollover.py:146-206`) holds `registry_lock()` — the module `_LOCK` `RLock`
  defined at `brain/chat/session.py:104` — across the ENTIRE destructive section: seed re-read
  (`read_session`, `rollover.py:147`) → seed-row construction (mode-dependent, `rollover.py:149-163`) →
  `create_session` (`rollover.py:170-171`) → `rewrite_session_atomic` (`rollover.py:174`) →
  `write_cursor` (`rollover.py:177-178`) → `write_rolled_to` (`rollover.py:190`) → `remove_session`
  (`rollover.py:191`). The pointer write (`write_rolled_to`) happens strictly before `remove_session` and
  both are still inside the `with registry_lock():` block — confirmed by direct read of the block, not by
  trusting the comment.
- `persist_turns_following_successor` (`brain/chat/session.py:269-298`) — the ONLY other production writer to
  a session buffer (verified: `grep -rn "ingest_turn(" brain/` outside tests shows exactly two production
  call sites — the function definition itself in `brain/ingest/buffer.py:74` and this one call site at
  `session.py:298`; `brain/chat/engine.py`'s old direct `ingest_turn` calls are gone, replaced by a single
  call to this function at `engine.py:449`, confirmed via `git diff 5348cae8 2e1bcf6f -- brain/chat/engine.py`)
  — takes the SAME `_LOCK` for its entire resolve+append body.
- Because both critical sections use the same non-reentrant-across-threads `RLock`, the two can never
  interleave partially: a persist for `old_sid` either (a) fully completes BEFORE the rollover's seed re-read
  acquires the lock — in which case its turn is already on disk when `read_session` runs and is captured into
  `raw`/`seed_rows` for BOTH modes — or (b) fully executes AFTER the rollover releases the lock, at which
  point `write_rolled_to` has already run, so `_resolve_successor` (`session.py:220-236`) returns the new sid
  and the persist redirects there via `ingest_turn(persona_dir, {**rec, "session_id": successor})`
  (`session.py:296-298`). There is no ordering in which a persist can land on `old_sid` AFTER
  `remove_session`/`write_rolled_to` but write straight to the old buffer — every append path routes through
  `_resolve_successor` under the same lock.
- This reasoning is mode-agnostic: the lock scope and pointer-write timing are identical for `summary_only`
  and `tiers_plus_tail` — only the seed-row CONTENT differs (`rollover.py:149-163`). So the resurrection
  closure applies to both, not just `tiers_plus_tail` as round-3/round-4-original found.
- Confirmed concretely by `test_persist_during_rollover_destructive_window_not_orphaned`
  (`tests/unit/brain/chat/test_rollover.py:603-684`) — a genuine two-thread test that pauses
  `delete_session_buffer` (monkeypatched) mid-rollover and races a real `persist_turns_following_successor`
  call against it; it passes. And by `test_persist_after_rollover_redirects_to_successor`
  (`test_rollover.py:514-555`), which exercises ordering (b) directly. Both ran green in this session.

### Charge 2 — is the r4 `summary_only` data-loss now closed?

Yes, verified by trace and by re-deriving the exact scenario the original round-4 review reproduced.

- `rollover.py:149-163`: when `seed_mode == "summary_only"` and a `summary_row` was found at the T0 read,
  `seed_rows = [summary_row] + raw` (`rollover.py:163`) — `raw` here is whatever is STILL in the buffer at the
  T0 read, i.e. any turn present after the fold (step 2) and not yet folded/archived.
- Traced the exact race the original r4 review reproduced ("a live turn racing in between the fold step and
  the registry-lock seed-read"): the fold (`compact_conversation`, called at `rollover.py:117-121` with
  `min_keep_tail=0, older_than=timedelta(0)`) is NOT naive-overwrite — its own install step re-reads the live
  buffer immediately before its `rewrite_session_atomic` call and reconciles by turn identity
  (`brain/chat/compaction.py:612-628`, comment at 610-621: "Re-read the live buffer just before the rewrite
  and rebuild the retained set from CURRENT turns minus the archived ones"). A turn that lands on `old_sid`
  DURING the fold's multi-second summarize call is therefore never silently overwritten by the fold — it
  survives the fold's own rewrite as a retained raw turn, exactly the "residual raw" state
  `test_summary_only_carries_residual_raw_not_dropped` (`test_rollover.py:687-725`) sets up directly. When
  `perform_rollover` then does its own T0 `read_session` (`rollover.py:147`), it sees that same residual raw
  turn and — with the fix — carries it via `[summary_row] + raw` instead of dropping it.
- A turn that lands AFTER the fold's rewrite but BEFORE `perform_rollover`'s T0 read is even easier to
  handle: at that point neither lock is held yet (the fold runs entirely before `acquire_compaction_lock` at
  `rollover.py:130`), so the persist either completes and is visible at T0, or (per charge 1) is serialized
  against the T0 read by `_LOCK` and lands in one of the two safe orderings.
- `summary_row is None` branch (`rollover.py:150-158`): `seed_rows = []` → falls through to
  `if not seed_rows: return None` at `rollover.py:165-167`, which executes INSIDE the `with registry_lock():`
  block, before any of `create_session`/`rewrite_session_atomic`/`write_rolled_to`/`remove_session` run, and
  Python's `with` statement releases the lock cleanly on the early `return`. The old buffer is never touched
  — confirmed by re-running `test_sessions_active_skips_stale_over_24h`
  (`tests/bridge/test_endpoints.py:654-659`), which exercises exactly this abort path through the real
  `/sessions/active` endpoint and passed.
- `test_summary_only_carries_residual_raw_not_dropped` is a genuine fail-demo in the sense that matters here:
  it is structurally identical to the failing repro the original round-4 review wrote independently (residual
  raw present after a stubbed no-op fold), and it exercises the actual `rollover.py:149-163` code path, not a
  mock of it.

### Charge 3 — any new race/loss from carrying `+ raw` (e.g. duplication)?

None found. Because `raw` is a single point-in-time snapshot taken from the SAME `read_session` call
(`rollover.py:147`) used to determine `summary_row`, and because that read happens once, under the lock, the
carried raw turns are exactly "whatever `old_sid`'s buffer contained at the T0 read" — there is no second,
independent read that could duplicate or diverge from it.

Traced the specific duplication concern from the charge: could a turn be captured into the seed AND ALSO
redirected to the successor? For that to happen, the SAME `persist_turns_following_successor` call would have
to write to `old_sid` (to be captured into the T0 read) and separately write to the successor (to be
"redirected") for the SAME logical turn — but a single call processes each record exactly once
(`session.py:294-298`, one `ingest_turn` call per record, decided once by a single `_resolve_successor`
check). Two DIFFERENT calls (e.g. the user-record and the assistant-record from the same `_persist_turn`
pair) could in principle straddle the rollover boundary, landing one on each side — but `2e1bcf6f`'s parent
commit `33cd5ba2` already closed that: `engine.py:449` now makes ONE call to
`persist_turns_following_successor(persona_dir, [user_record, assistant_record])` for the pair, so both
records are resolved and appended under the SAME `_LOCK` acquisition (`session.py:294` loop runs inside one
`with _LOCK:` block) — they cannot straddle a rollover. (Pre-`33cd5ba2`, `engine.py` made two SEPARATE
`ingest_turn` calls for the pair with no shared lock at all; that path is gone.)

No loss scenario introduced either: `raw` is additive to the pre-existing `[summary_row]`-only seed, and the
common case (no race) is `raw == []`, a pure no-op — confirmed by the assertion in
`test_summary_only_carries_residual_raw_not_dropped` that the summary is still `texts[0]` (order preserved,
nothing reordered/dropped).

### Charge 4 — deadlock / lock ordering / event-loop hazard

No reverse-order acquisition found. `grep -rn "acquire_compaction_lock\|registry_lock(" brain/ --include=*.py`
(excluding tests) shows `registry_lock()`/`_LOCK` is acquired ONLY in `session.py` (`persist_turns_following_
successor`, plus the pre-existing `create_session`/`remove_session`/`get_or_hydrate_session`/etc., all
reentrant via the same `RLock`) and in `rollover.py:146` inside `perform_rollover`. `acquire_compaction_lock`
(`brain/ingest/buffer.py:464`) is a non-blocking, file-based, O_EXCL advisory lock — it never blocks the
calling thread (`FileExistsError` → return `False` immediately, `brain/ingest/buffer.py:474-497`) — so even
though `perform_rollover` acquires it BEFORE `registry_lock()` (`rollover.py:130` then `146`), and no other
path acquires `registry_lock()` first and then blocks on `acquire_compaction_lock`, there is no possible
deadlock: the only primitive that can actually block a thread is `_LOCK`, and nothing acquires `_LOCK` and
then waits on anything else that could itself be waiting on `_LOCK`.

Confirmed the slow fold (`compact_conversation`/`cascade_conversation`, `rollover.py:117-124`) runs BEFORE
`acquire_compaction_lock` is even called (`rollover.py:130`) — i.e. entirely outside both locks — and
`delete_session_buffer`/`delete_cursor`/`delete_backoff` (`rollover.py:209-211`, with `_unlink_with_retry`'s
bounded retry loop at `brain/ingest/buffer.py:141-150`) run AFTER the `with registry_lock():` block exits
(`rollover.py:203` closes it) but still inside the outer `try` (released via `finally: release_compaction_
lock` at `rollover.py:213-214`) — so the retry-looping unlink is outside `_LOCK`, matching the docstring
claim.

Event-loop hazard (noted per the charge, not blocked on, consistent with the original round-4 review's
"Minor"): `get_or_hydrate_session` (`session.py:129-137`) still acquires `_LOCK` synchronously and is still
called directly (not via `asyncio.to_thread`) from `async def` FastAPI route handlers — confirmed at
`brain/bridge/server.py:1317` (inside `async def chat`) and `server.py:2427`/`2766`/`2827` (other async
routes) — so a long hold of `_LOCK` by a rollover on the supervisor thread can stall the event loop for every
concurrent `/chat`/`/state`/etc. request. This is unchanged by `2e1bcf6f` (the only addition to the locked
section is a `+ raw` list-concatenation, negligible) and was already flagged as pre-existing/Minor by the
original round-4 review in this same file; not a new issue introduced by either commit under review here.
`persist_turns_following_successor` itself is only ever reached via `_persist_turn` → `engine.respond`, which
is invoked exclusively through `asyncio.to_thread(_respond_blocking, ...)` — confirmed at `server.py:2442`
and `server.py:2609`, with `_respond_blocking`'s own docstring stating as much (`server.py:249`) — so the
live-turn persist path itself never runs on the event-loop thread, matching the design claim; only the
(pre-existing) hydrate-on-lookup path does.

### Charge 5 — are comments/docstrings now accurate?

Yes, and demonstrably corrected from the r3-era overstatement the original round-4 review flagged.
`brain/bridge/supervisor.py`'s `_run_compaction_tick` docstring previously read "This makes the resolve-
persist race unreachable (the rollover cannot delete an active session's buffer out from under a mid-tool-
loop request)" — `git diff 5348cae8 2e1bcf6f -- brain/bridge/supervisor.py` shows this replaced with language
correctly demoting `is_session_busy` to "a best-effort efficiency/UX belt... NOT the race-safety mechanism"
and naming `registry_lock()` as the actual mechanism. `maybe_weekly_rollover`'s docstring
(`rollover.py:216-243`) states the same thing correctly. The one place an unqualified "closes ... by
construction" phrase remains (`rollover.py:135`, `session.py:261-266`) is now accurate as written per charge
1's trace: with `2e1bcf6f` in place, the closure genuinely applies to both seed modes, not just
`tiers_plus_tail` — so this is no longer the overstatement the original round-4 review caught (at that time
`33cd5ba2` alone had NOT closed it for `summary_only`, and the review correctly flagged the phrase as
inaccurate). No new overstated claim was introduced by either commit.

### Charge 6 — do the tests genuinely discriminate?

`uv run pytest tests/unit/brain/chat/test_rollover.py tests/bridge/test_cascade_rollover_endpoints.py -q` →
**20 passed.** Broader sweep (`test_compaction_cadence.py`, `test_engine.py`, `test_endpoints.py`, all of
`tests/unit/brain/chat/`) → **416 passed, 1 xfailed**, no regressions.

- `test_persist_during_rollover_destructive_window_not_orphaned` (`test_rollover.py:603-684`): REAL. Two
  actual `threading.Thread`s, a monkeypatched `delete_session_buffer` that blocks on an `Event` so the racing
  persist is guaranteed to land in the exact post-pointer-write/pre-delete window, asserting both no
  resurrection and the raced turn present in the successor. This is a true concurrency test, not a sequential
  simulation.
- `test_plain_ingest_after_rollover_resurrects_fail_demo` (`test_rollover.py:558-600`): REAL fail-demo. Calls
  the OLD unredirected `ingest_turn` path directly (bypassing the fix) and asserts the bug DOES reproduce
  (old buffer resurrected, turn NOT in successor) — proves the regression guard in the adjacent test is not
  vacuous, since the harness can demonstrably fail.
- `test_persist_after_rollover_redirects_to_successor` (`test_rollover.py:514-555`): REAL, sequential (not
  concurrent) but exercises the actual production call path (`persist_turns_following_successor`) after a
  real `perform_rollover` call, asserting the redirect fires and lands in the successor. Sound mechanism
  check, not a concurrency test — same characterization the original round-4 review gave the analogous test
  in `33cd5ba2`.
- `test_summary_only_carries_residual_raw_not_dropped` (`test_rollover.py:687-725`): REAL but sequential —
  the fold is monkeypatched to a no-op and the buffer is pre-seeded to look like a post-fold-with-race state,
  rather than driving an actual concurrent race. It does exercise the real `rollover.py:149-163` seed-
  selection code and would fail against the pre-`2e1bcf6f` `[summary_row]`-only logic (I confirmed this by
  reading the diff: `git diff 5348cae8 2e1bcf6f -- brain/chat/rollover.py` shows the old code path had no
  `+ raw` term at all in the `summary_only` branch, so this test is shown-able-to-fail on the parent commit).
  **Minor gap:** there is no true two-thread concurrent test for `summary_only` analogous to
  `test_persist_during_rollover_destructive_window_not_orphaned` (i.e. one that races a real persist against a
  real in-progress `summary_only` rollover, rather than pre-seeding the post-race state). Given the shared
  underlying mechanism (`registry_lock()` scope) is already proven correct under true concurrency for
  `tiers_plus_tail`, and the `summary_only`-specific piece being tested is pure seed-row selection logic (no
  additional concurrency-sensitive code), this is a coverage gap worth closing opportunistically, not a
  reason to doubt the fix.

### Charge 7 — regression check

`a485e72d`'s C-1 redirect-first resolution: `git diff 5348cae8 2e1bcf6f -- brain/chat/session.py` shows
`_resolve_successor` (`session.py:220-236`) and the successor-first ordering in `get_or_hydrate_session`
(`session.py:174-179`, consults `_resolve_successor` before the `_SESSIONS` cache and before the on-disk
buffer) are byte-identical to the base — the diff only ADDS `registry_lock()` and `persist_turns_following_
successor` after the existing code, touching nothing in the C-1 path. Idle-gate wiring
(`is_session_busy`, `5348cae8`): `_run_compaction_tick` (`supervisor.py:1640-1701`) and `maybe_weekly_rollover`
(`rollover.py:216-282`) still gate on the same two conditions (quiet-gap + `is_session_busy`) in the same
places; only the docstrings changed (see charge 5) plus one keyword-argument-position fix in
`test_compaction_cadence.py` (confirmed by re-running that file green). No regression found in either.

### Verdict

1. **Worst severity: none (no residual Blocker/Major/Minor defect found in the fix itself).** The one
   observation worth recording is a **Minor test-coverage gap** (charge 6): `summary_only`'s residual-raw
   carry is verified by a sound sequential test, not a true concurrent race test, unlike its `tiers_plus_tail`
   sibling. The event-loop `_LOCK` contention noted under charge 4 is a pre-existing characteristic
   (unaffected in magnitude by this delta) already logged as Minor by the original round-4 review in this
   file, not a new finding.
2. **Findings, one line each:**
   - The r3 resurrection race (`tiers_plus_tail`) remains closed by construction, and the closure is now
     shown to be mode-agnostic (applies to `summary_only` too) by direct trace of the shared `registry_lock()`
     section.
   - The r4 `summary_only` data-loss Major this file's own round-4 review reproduced is closed by
     `2e1bcf6f`'s `[summary_row] + raw` carry (`rollover.py:163`), verified by trace through the fold's own
     race-safe re-read (`compaction.py:612-628`) and by a real (if sequential) regression test.
   - No duplication or new loss from carrying `+ raw`: the seed is one point-in-time snapshot, and
     `33cd5ba2` already collapsed the user+assistant persist pair into one lock-protected call, closing the
     one plausible split-pair duplication path before it could interact with the r4 carry.
   - No deadlock: `_LOCK` is the only blocking primitive in play; `acquire_compaction_lock` never blocks; no
     reverse-order acquisition exists anywhere in the touched code.
   - Docstrings no longer overstate what's closed; the previously-flagged inaccurate "unreachable by
     construction" language is now accurate given `2e1bcf6f` closes the mode-agnostic gap it was flagged for.
   - `a485e72d` C-1 and the idle-gate wiring are both unregressed (diff-confirmed + green tests).
3. **Explicit answer:** **YES — both the r3 resurrection race and the r4 `summary_only` data loss are now
   closed, with no residual TOCTOU and no new issue found.** The only open item is a test-coverage
   nice-to-have (a true concurrent race test for `summary_only`, mirroring the existing `tiers_plus_tail` one)
   which does not indicate a live defect given the shared, already-concurrently-tested locking mechanism.
4. **Routing recommendation: advance to build-gate.** Optionally fold in a concurrent `summary_only` test
   (mirroring `test_persist_during_rollover_destructive_window_not_orphaned`) as a low-cost follow-up, but it
   is not a blocking condition for this change.
