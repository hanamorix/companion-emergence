# Stage-6 code re-review (round 3) — idle-gate fix for the resolve-persist race

**Reviewer stance:** cold, independent, no shared context with the author or with prior review rounds beyond
what is verifiable from source. Scope: commit `5348cae8c52d6d37baacb94e276d6361b40367cb` ("Gate weekly
rollover and compaction to idle periods") only — the fix for the "resolve-persist race" Major surfaced in
round 2 (`6-redteam-code-r2.md`, reviewing `a485e72d`). Also re-verified that `a485e72d`'s own fix (C-1,
redirect-first resolution) is not regressed by this delta.

## Provenance

```
cd /home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction
git show 5348cae8 --stat
git diff 5348cae8^ 5348cae8 -- brain/ tests/
```

- Fix commit: `5348cae8c52d6d37baacb94e276d6361b40367cb`, parent `a485e72d984ac2fe30f1d204b53bd5b402f01627`
- HEAD at review time == `5348cae8` (worktree clean except an unrelated in-progress edit to
  `changes/cascade-compaction/8-harness.md` adding an RP-1 harness row — not touched, not part of this
  review's file set)
- Files touched (code, excluding `changes/*.md` docs): `brain/bridge/server.py` (+11),
  `brain/bridge/supervisor.py` (+53/-6), `brain/chat/budget.py` (comment-only, +26/-16 lines of docstring),
  `brain/chat/rollover.py` (+27/-6), `tests/bridge/test_cascade_rollover_endpoints.py` (+82, two new tests)
- Ran `tests/bridge/test_cascade_rollover_endpoints.py` at HEAD: 10/10 pass, including the two new idle-gate
  tests (`test_owner_idle_gate_defers_busy_session`, `test_weekly_rollover_belt_defers_when_busy`).

## Charge 1 (the load-bearing check): does `in_flight_locks` span the full request lifetime?

**Answer: YES for the intra-request lock itself, but that is not sufficient — the gate built on top of it
(`is_session_busy`) is a point-in-time snapshot, not a held condition, and this reopens a narrower version of
exactly the race the commit claims to close. Net: NO, the fix does not make the resolve-persist race
unreachable; it substantially shrinks the window without closing it.**

**1a. The intra-request lock itself is correctly held end-to-end** (verified by direct trace, not assumption):

- `POST /chat` (`brain/bridge/server.py:2422-2472`): `sess = get_or_hydrate_session(...)` runs, then
  `sid = sess.session_id`, then `lock = s.in_flight_locks.setdefault(sid, asyncio.Lock())`, then
  `async with lock:` wraps `await asyncio.to_thread(_respond_blocking, ...)` — the entire blocking call.
- `_respond_blocking` (`server.py:241-273`) calls `brain.chat.engine.respond(...)` synchronously and returns
  its result directly — no early return before `respond()` finishes.
- `respond()` (`brain/chat/engine.py:86-315`): step 7 runs `run_tool_loop(...)` (the multi-second LLM
  tool-loop, line 272-284), and only **after** it returns does step 8 call `_persist_turn(...)` (line
  288-294, which persists via `ingest_turn`). The function returns `ChatResult` only after persistence.
- So the `async with lock:` block in the `/chat` handler covers session-resolution-adjacent code through the
  full tool-loop through full persistence, for the duration of one request. Same structure confirmed for
  `WS /stream` (`server.py:2576-2700`: `async with lock:` wraps `_guarded_respond()` → `_respond_blocking` via
  `asyncio.create_task` + `await respond_task`, i.e., the WS handler doesn't release the lock or leave the
  `async with` block until `_respond_blocking` (same respond()-then-persist) returns).
- `_is_session_busy` (`server.py:950-958`) reads `app.state.bridge.in_flight_locks.get(sid)` and calls
  `.locked()` — a plain bool read, correctly documented as best-effort but adequate for this purpose (GIL
  makes the read atomic; no torn state).

**1b. But `is_session_busy` is checked once, not held, on the rollover side — and the gap between the check
and the actual buffer delete is not negligible.**

`maybe_weekly_rollover` (`brain/chat/rollover.py:187-230`) checks `is_session_busy(session_id)` **once**,
immediately before calling `perform_rollover(...)` (rollover.py:228-230). `perform_rollover`
(`rollover.py:63-176`) does NOT accept or re-check `is_session_busy` anywhere internally. Between the check
and the actual buffer deletion, `perform_rollover` does real, non-trivial work:

1. `extract_session_snapshot(...)` (rollover.py:93-99) — an LLM call (`provider.generate` via
   `build_compaction_provider`) for final memory extraction. Best-effort, but not instant.
2. `cascade_conversation(...)` (rollover.py:118-124, `seed_mode="tiers_plus_tail"`) — another LLM-driven
   compaction pass over the buffer.
3. `acquire_compaction_lock` → re-read → `create_session` → `rewrite_session_atomic` → `write_cursor`
   (rollover.py:129-161).
4. **Only then**: `write_rolled_to(...)` → `remove_session(old_sid)` → `delete_session_buffer(...)` →
   `delete_cursor`/`delete_backoff` (rollover.py:163-168) — the actual destructive operations.

None of steps 1-4 re-checks `is_session_busy`. So the sequence that reproduces the exact bug this commit is
meant to close is:

- t0: `_run_compaction_tick`/`maybe_weekly_rollover` checks `is_session_busy(sid)` → False (genuinely no
  in-flight request at t0 — the session is weekly-eligible and quiet-gap-eligible, which is precisely the
  "user reconnects after being away long enough" scenario the round-2 review named as the trigger).
- t0+ε: a client sends a new request for `sid`. `get_or_hydrate_session` resolves it correctly to the *old*
  session (no `rolled_to` pointer exists yet — the rollover hasn't reached step 4), sets/acquires
  `in_flight_locks[sid]`, and starts its multi-second tool loop. `is_session_busy(sid)` would now report
  True, but nothing on the rollover side asks again.
- t1 (seconds later, after steps 1-3's LLM calls complete): `perform_rollover` reaches step 4, deletes the
  buffer, evicts the registry, writes the successor pointer — while the request from t0+ε is still running.
- t2 (request's tool loop finishes): `_persist_turn` → `ingest_turn` opens the buffer path in **append-create
  mode with no existence check** (`brain/ingest/buffer.py:101-103`, `open(path, "a", ...)`) —
  resurrecting the just-deleted file and orphaning the turn outside the successor chain. This is the
  identical symptom the commit's own message and code comments say is now "unreachable."

I confirmed there is no synchronization anywhere in the ingest/lock path that would prevent this: `ingest_turn`
takes no lock at all (grepped — plain `open(..., "a")`), and `perform_rollover`'s own
`acquire_compaction_lock`/`release_compaction_lock` (`brain/ingest/buffer.py:464-516`) is a *compaction-vs-
compaction* reentrancy guard (keyed off pid/staleness), not something a live chat request ever touches.

**This is not a hypothetical edge case invented for the review — it is the same trigger the round-2 reviewer
(`6-redteam-code-r2.md`, "Residual window — NOT closed by this delta") named for the pre-existing bug this
commit exists to fix**, just with the window narrowed from "any time during the session's idle-then-reattach
window" to "the duration of one `perform_rollover` call's extraction+cascade+seed steps" (order of seconds,
the same order of magnitude as the tool-loop window the fix is meant to close). The two new tests added in
this commit (`test_owner_idle_gate_defers_busy_session`, `test_weekly_rollover_belt_defers_when_busy`) both
pass a **static** `is_session_busy` lambda fixed for the whole call and never exercise a session that
transitions from not-busy to busy mid-`perform_rollover` — so the test suite does not, and structurally
cannot with the current test shape, catch this gap.

**Severity: Major.** Same failure class/consequence as the original C-1 and the round-2 residual finding
(silent turn loss, orphaned/resurrected buffer). The commit message and in-code comments
(`rollover.py`: "Together these make the resolve-then-long-generate-then-persist race unreachable"; `server.py`
docstring similarly) assert a stronger guarantee than the code delivers. The fix is a real, substantial
improvement (the old code had *no* busy-check at all — rollover could fire at any point regardless of
in-flight state), but "unreachable"/"closes the race by construction" (per `decisions.md`'s owner-ruling
writeup) is not accurate as implemented — a check-then-act gap remains around the single most
expensive part of the operation.

**Suggested minimal fix (not verified/implemented by me, out of scope for a review):** re-check
`is_session_busy(old_sid)` immediately before the destructive step (`write_rolled_to` / `remove_session` /
`delete_session_buffer` in `perform_rollover`), not only once at `maybe_weekly_rollover`'s entry. This would
not make the race fully impossible (a TOCTOU gap between a final check and the actual OS-level delete/registry
mutation is unavoidable without a lock spanning both sides), but it would shrink the window from
"several seconds of LLM calls" to "a few lines of pure Python," matching the tightness `a485e72d` achieved
for the resolve side. A structurally complete fix would need the rollover to hold (or coordinate with) the
same `in_flight_locks` entry across its full critical section, which is a bigger design change than this
commit attempts.

## Charge 2: is the idle-gating wiring itself correct (server → tick → rollover)?

**Yes, the wiring is correct and complete** — this is a distinct question from charge 1 (whether the
*concept* fully closes the race) and the wiring passes:

- `server.py:950-958` defines `_is_session_busy` closing over `app.state.bridge.in_flight_locks`, and threads
  it into `run_folded(..., is_session_busy=_is_session_busy)` (server.py:970).
- `supervisor.py:141` adds the `is_session_busy: Callable[[str], bool] | None = None` parameter to
  `run_folded`.
- `run_folded` passes it to `_run_compaction_tick` at **both** call sites: the new startup catch-up
  (`supervisor.py:255-259`) and the existing periodic cadence block (`supervisor.py:672-675`).
- `_run_compaction_tick` (`supervisor.py:1641-1710`) checks `is_session_busy(session_id)` once per session,
  before calling `cascade_conversation`, and skips (`continue`) both the cascade and the rollover for that
  session on a busy hit (supervisor.py:1686-1690). For a non-busy session it threads `is_session_busy` through
  to `maybe_weekly_rollover` (supervisor.py:1699-1706) as the documented "belt" second check.
- `maybe_weekly_rollover` (`rollover.py:187-230`) checks it a second time, right before calling
  `perform_rollover`, as analyzed in charge 1.

No path skips the wiring; no path fires rollover/compaction for a session it never checked. I found no typo,
no wrong-callable, no swapped argument. The wiring is sound *given* the point-in-time-check design; the design
itself is where charge 1's gap lives.

I also checked for a path that fires mid-exchange despite the gate: none found. `emergency_fold_24h` (via
`apply_budget`, see charge 4) runs *inside* the current request's own thread on its *own* buffer — it is not
a background job racing a different request, so it is out of scope for this "mid-exchange from a different
actor" concern, and does not bypass the gate (it isn't the gate's job to cover it — see charge 4).

## Charge 3: is the startup catch-up cascade correct and safe (idle by nature)?

**Correct in intent, inherits the same gap as charge 1 in a narrow edge case, but low-risk in practice.**

`supervisor.py:245-259` runs `_run_compaction_tick(..., is_session_busy=is_session_busy)` once, early in
`run_folded`, before the loop that later processes `daemon_state`/notes/etc. and before the periodic cadence
block. At the point this fires, `in_flight_locks` was just initialized to `{}` (`server.py:870`,
`BridgeAppState.in_flight_locks={}`) — no requests have been served yet in this process — so
`is_session_busy` trivially returns False for everything, matching "startup is idle by nature" as claimed.

The one caveat: `sup_thread.start()` (`server.py:975`) is non-blocking — it kicks off the supervisor thread
and returns immediately to the surrounding FastAPI startup/lifespan code, which then completes. If the ASGI
server begins accepting connections concurrently with (rather than strictly after) the supervisor thread's
startup catch-up section finishing its `_run_compaction_tick` call, a request landing on a weekly-eligible
session in that narrow window would hit the exact charge-1 gap (busy-check passes trivially at t0 because no
lock exists yet, then a real request registers and runs while the startup catch-up's `perform_rollover` is
still doing its extraction/cascade/delete). This is the same underlying gap as charge 1, not a separate defect
— I'm noting it here only because charge 3 specifically asked about startup safety. In practice this is very
low-probability (a client would have to connect and address a specific weekly-eligible session within the
first fraction of a second of a fresh process start, before any prior request has ever touched that session's
lock), and is not something I'd block on independently of the charge-1 finding.

## Charge 4: is the backstop reasoning in `brain/chat/budget.py` sound?

**Yes, confirmed against the code, not just the comment.** `apply_budget` (`budget.py:49-126`):

- Step 1 (`budget.py:78-99`): `emergency_fold_24h(...)` runs inside a `try/except Exception` that only logs
  on failure — its outcome (success, failure, or skip because `persona_dir`/`session_id` is None) has **no
  effect on whether step 2 executes**.
- Step 2 (`budget.py:101-126`): the deterministic `[truncated N earlier messages]` windowing always runs when
  the initial `_estimate_tokens(messages) > max_tokens` check (line 70) fired, unconditionally on step 1's
  outcome (the only early-return inside step 2 is `len(messages) < 2 + preserve_tail_msgs`, a genuine "nothing
  to truncate" case, not a step-1-dependent branch). It does not write to the buffer (no `ingest_turn`/
  `rewrite_session_atomic` call in this code path) — it only rewrites the in-memory `messages` list returned
  to the caller for this one turn.

So the claim holds: the within-session prompt-size bound is structurally independent of whether step 1 (the
persisted `emergency_fold_24h`) runs, succeeds, or is skipped. Idle-gating routine/weekly compaction (this
commit's actual change) therefore cannot cause unbounded prompt growth within an active session — the
windowing step is unconditional. `emergency_fold_24h` genuinely reads as a best-effort persistence nicety on
top of an already-guaranteed prompt-size bound, matching the documented "last-resort emergency net" framing.
No issue found here.

## Charge 5: any NEW correctness/concurrency issue introduced by this delta?

- **No deadlock risk**: `_is_session_busy` never acquires a lock itself — it does a dict `.get()` (GIL-atomic)
  and calls `asyncio.Lock.locked()`, which is a plain attribute read, not a blocking acquire, and is safe to
  call from a non-owning thread for this best-effort purpose. Called synchronously from the supervisor thread,
  never awaited, no cross-thread lock acquisition anywhere in the new code.
- **Starvation is theoretically possible but not newly introduced**: a session whose `in_flight_locks` entry
  is (for whatever reason) permanently `locked()` would never be compacted or rolled over. This requires a
  hung request that never returns from `asyncio.to_thread(_respond_blocking, ...)` — but such a session is
  already unusable for *any* further chat (the `/chat` and WS handlers both reject/eror on a locked
  session before this commit), so this is not a new failure mode this delta introduces; it's a pre-existing
  hung-request scenario now additionally suppressing compaction for that one session, which is a benign
  consequence, not a new bug.
- **The charge-1 residual gap is the one real "new correctness issue" territory**, but as established above
  it is not *introduced* by this delta — it *narrows* a pre-existing (round-2-documented) Major without fully
  closing it, and the commit's own comments overstate the closure. I am not double-counting it as a second
  finding; charge 1's writeup is the authoritative record.
- Nothing else in the diff touches shared mutable state in a new way. `budget.py`'s changes are pure comment
  edits (confirmed via `git diff` — the code inside `apply_budget` is byte-identical pre/post this commit;
  only the module docstring and step 1's inline comments changed).

## Prior fix (`a485e72d`, redirect-first resolution) — regression check

**Not regressed.** `git show 5348cae8 -- brain/chat/session.py` returns no diff — `session.py` (where the
`_resolve_successor`-first ordering lives) is untouched by this commit. I independently re-read
`get_or_hydrate_session` (`session.py:135-215`) at HEAD `5348cae8` and confirmed the ordering from `a485e72d`
is intact: `_resolve_successor` is still called unconditionally first inside `with _LOCK:`, before the
`_SESSIONS` cache check and before `read_session`. This commit's rollover-side changes (the `is_session_busy`
checks in `rollover.py`) sit *before* `perform_rollover` is called at all, so they cannot interact with or
weaken the C-1 fix's internal ordering — they only gate whether `perform_rollover` runs in the first place.
Ran the existing `test_c1_mid_rollover_window_redirects_not_resurrect`-covering test file
(`tests/bridge/test_cascade_rollover_endpoints.py`, which includes C16/C19-C21 conformance) at HEAD: all pass.

## Bottom line

The commit is a genuine, well-reasoned improvement — it correctly wires a busy-check through both the
periodic and startup-catch-up paths (charge 2: sound), the startup cascade is safe modulo a vanishingly
unlikely edge case shared with charge 1 (charge 3: sound), and the backstop-vs-idle-gate reasoning in
`budget.py` is verified correct against the actual code, not just the comment (charge 4: sound). It does not
regress the prior `a485e72d` fix (untouched file, re-verified).

**However, charge 1 — the load-bearing question — does not fully hold.** The in-flight lock does span a
single request's full lifetime (resolution-adjacent code through tool-loop through persistence) correctly.
But the gate this commit builds against that lock (`is_session_busy`) is a one-time snapshot taken before
`perform_rollover`'s multi-second, LLM-driven extraction/cascade/delete sequence, with no re-check at any
point during that sequence — and nothing in `ingest_turn`'s append-create path or `perform_rollover`'s
compaction lock provides any alternate synchronization. This reopens a narrower version of the exact
resolve-persist race the commit's message and comments claim to make "unreachable." The trigger (a
weekly-eligible, quiet-gap-eligible session that a client reconnects to right as the daily tick's rollover is
mid-flight) is the same sticky-session-reattach scenario this subsystem is explicitly designed around, so it
is not a contrived corner case.

## Verdict

1. **Worst severity: Major.**
2. **Findings, one line each:**
   - Major (charge 1): `is_session_busy` is checked once before `perform_rollover`'s multi-second
     extraction+cascade+delete sequence, with no re-check inside it and no synchronization in
     `ingest_turn`'s append-create path — a narrower but real residual resolve-persist race remains, contrary
     to the "unreachable"/"closes the race by construction" claims in the code comments and `decisions.md`.
   - Sound (charge 2): server→tick→rollover wiring of `is_session_busy` is complete and correct; no path
     fires the gate ungated.
   - Sound, with a noted low-probability shared edge case (charge 3): startup catch-up cascade is idle by
     construction (empty `in_flight_locks` at that point); theoretically shares charge 1's gap if a request
     arrives in the same instant the ASGI server starts accepting connections, before the catch-up completes.
   - Sound (charge 4): `budget.py`'s deterministic windowing (step 2) is unconditional on the persisted
     `emergency_fold_24h` (step 1)'s outcome — verified in code, not just comments — so idle-gating routine
     compaction cannot cause unbounded prompt growth.
   - No new issue (charge 5): no deadlock; a benign, pre-existing-class starvation edge case (hung request →
     that session skips compaction, but was already unusable for chat); `budget.py`'s diff is comment-only.
   - `a485e72d` (C-1 redirect-first fix) not regressed: `session.py` untouched by this commit; ordering
     re-verified at HEAD; conformance tests pass.
3. **Charge 1 explicit answer: the lock spans a single request's own lifetime correctly (server.py
   resolve→lock→tool-loop→persist, and engine.py's tool-loop-then-persist-then-return, traced line-by-line).
   But "does the in-flight lock close the resolve-persist race" — NO, not fully: `perform_rollover`
   (`brain/chat/rollover.py:63-176`) performs its extraction (line 93-99, LLM call) and cascade (line
   118-124, LLM call) and seed/rewrite (line 129-161) steps *after* the single `is_session_busy` check
   (`rollover.py:226-227`) and *before* the actual buffer delete (line 165), with no re-check in between and
   no lock held across that span. A request that resolves and registers its `in_flight_locks` entry inside
   that multi-second window reproduces the original orphaned-turn/resurrected-buffer bug.**
4. **Routing recommendation: back-to-stage-5.** This is the same failure class the commit was built to close
   (per `decisions.md`'s "OWNER RULINGS — idle-gating" entry, which states the fix "Closes the race by
   construction"), and the residual window, while much narrower than before, is not negligible (multi-second,
   LLM-call-bounded, triggered by the exact reattach scenario this subsystem targets) and is asserted as
   closed in both code comments and the design record. Recommend: add a re-check of `is_session_busy`
   immediately before `perform_rollover`'s destructive step (or a structurally equivalent close of the gap),
   update the "unreachable"/"closes by construction" language to reflect the actual (narrowed, not
   eliminated) guarantee if a full close is deferred, and add a test that transitions a session from
   not-busy to busy *during* a `perform_rollover` call (e.g., via a stubbed `is_session_busy` or a monkeypatch
   on the extraction/cascade step) so the gap either gets closed or is knowingly, explicitly accepted with a
   named residual-risk note in `decisions.md` rather than left implicit.
