# Stage-6 code re-review (round 2) — C-1 mid-rollover redirect race fix

**Reviewer stance:** cold, independent, no shared context with the author. Scope: the fix commit
`a485e72d` only (the delta that closes stage-6 Major C-1), and its interaction with the resolve/rollover
paths. The rest of the cascade-compaction change is explicitly out of scope for this pass.

## Provenance

```
cd /home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction
git show a485e72d --stat
git diff 76e5f798..a485e72d -- brain/ tests/
```

- Fix commit: `a485e72d984ac2fe30f1d204b53bd5b402f01627`
- Parent (pre-fix) commit reviewed against: `76e5f798`
- `git cat-file -p a485e72d | sha256sum` → `0f7771aaea394b654d8fffd57f45983db3b312c59f493341e5159c4fa4031602`
- Files touched (excluding `changes/*.md` docs, per instructions): `brain/chat/compaction.py` (-2),
  `brain/chat/rollover.py` (+13/-6), `brain/chat/session.py` (+14/-9),
  `tests/unit/brain/chat/test_rollover.py` (+68, new test)
- HEAD at review time == `a485e72d` (worktree clean, `git status --short` empty)

## C-1 ruling: CLOSED

`get_or_hydrate_session` now calls `_resolve_successor(persona_dir, session_id)` **unconditionally, first,
before** the `_SESSIONS.get` cache check and before `read_session` (`brain/chat/session.py:166-183`). All
three of the function's prior return paths (cache hit, buffer-has-turns, buffer-empty) sit strictly after
this check inside the same `with _LOCK:` block, so none of them can execute without the pointer having
already been consulted. From the instant `write_rolled_to` becomes durable (`brain/ingest/buffer.py:222-234`,
tmp+fsync+`os.replace` — atomic on POSIX, no torn read possible), every subsequent call for the old sid
resolves to the successor, regardless of whether the registry has been evicted yet or the buffer deleted
yet. That is exactly the window C-1 named.

**Verified, not just read:**
- Ran the new regression test at HEAD: `test_c1_mid_rollover_window_redirects_not_resurrect` passes
  (`tests/unit/brain/chat/test_rollover.py`, 6/6 tests in the file, 494 passed / 1 xfailed across
  `tests/unit/brain/chat/` + `tests/unit/brain/ingest/`).
- **Sanity-checked the test is discriminating, not vacuous**: temporarily reverted just the
  ordering inside `get_or_hydrate_session` back to the pre-fix shape (cache-check first, pointer-check only
  reached from the empty-buffer branch — i.e. reproduced the exact code `git diff` shows was removed), reran
  the single test. It failed exactly as the docstring claims:
  ```
  AssertionError: mid-rollover request resolved to the stale-cached OLD session —
  the C-1 race (resurrectable old buffer) is not closed
  assert '468eea2e-...' == '5e439dbe-...'
  ```
  Restored the file from a pre-edit backup immediately after; `git status --short brain/chat/session.py`
  is clean (matches HEAD exactly) — no residual edit left in the worktree.
- Ran `ruff check` on all four touched files: clean.

## Per-lens findings

### Factual
No issue. Every claim in the new code comments about mechanism (pointer-first order, RLock reentrancy,
atomic `os.replace`, cursor-guard no-op) checks out against the actual code read.

### Logical
No issue in the fix itself. See Concurrency lens below for a residual gap in the *surrounding* request
lifecycle that the fix's logic does not (and structurally cannot, by itself) address.

### Missed-opportunity
**Nitpick.** `brain/chat/rollover.py:145-159`: the comment justifying the L2 revert states the daily tick's
cascade "cascades BEFORE any extraction (cursor guard → folds nothing)." That's only literally true the
*first* time a session is ever extracted (cursor starts `None`, and both `cascade_conversation` and
`compact_conversation` hard-no-op on `cursor is None` — confirmed at `brain/chat/compaction.py:869-871` and
`:472-479`). On a session's *second and later* weekly cycles the cursor already carries a value from an
earlier silence/finalize extraction (`brain/ingest/pipeline.py:507-521`, `:557-571`), so the tick's
pre-rollover cascade call is not literally a no-op — it folds whatever was extracted as of the *old* cursor.
The revert's actual justification (verified independently below) is the more general fact that extraction
strictly *advances* the cursor, so the post-extraction fold in `perform_rollover` step 2 always has a
cursor ≥ the tick's pre-extraction cursor, and therefore always has at least as much (usually strictly more)
eligible material — not that the tick-level fold is always a hard no-op. The conclusion (load-bearing, not
redundant, revert is correct) still holds; only the stated mechanism is overbroad for the general case.
Doc-only, no functional impact.

### Assumptions
No issue with the fix's own assumptions. The one assumption worth naming: the fix (and the original C-1
report) implicitly scope "a request for the old sid" to mean *the resolve call itself* landing in the
mid-rollover window. That assumption is correct for what it covers, but it does not cover a resolve that
happened *before* the window and is *used* after it — see Concurrency.

### Fidelity (position/render)
N/A — no render/positional change, confirmed no UI-facing diff in this delta.

### Concurrency (explicit ask)

**1. Resolve-path re-enumeration (C-1 itself): CLOSED, no residual window within `get_or_hydrate_session`.**
Confirmed above by direct fix-then-revert-then-refix test execution. `(a)` all three original return paths
now sit behind the unconditional pointer check — no path bypasses it. `(b)` `_LOCK = threading.RLock()`
(`session.py:97`), so the one level of self-recursion (`get_or_hydrate_session` → `get_or_hydrate_session`)
is safe. `(c)` No infinite loop: `_resolve_successor` already fully walks the chain to its terminal successor
internally (its own `visited` set aborts a cycle to `None`), so the *outer* function only ever recurses
**once** regardless of chain length — the recursive call's own `_resolve_successor` on the terminal node
finds no further pointer and returns `None` immediately. A cyclic pointer aborts to `None` via the
visited-set guard as documented, not a depth cap. `(d)` Hot path: for a session with no pointer,
`_resolve_successor` costs one `Path.exists()` stat (`read_rolled_to` short-circuits on the file not
existing) — cheap, and it is a real behavior change (previously a cache hit paid zero disk I/O; now every
call, hit or miss, pays one stat) but not a correctness regression. Minor performance note, not worth
blocking on.

**2. Rollover ordering: no new window opened by the reorder.** `write_rolled_to` → `remove_session` →
`delete_session_buffer` → `delete_cursor`/`delete_backoff` (`brain/chat/rollover.py:152-158`). The
pointer-first write invariant (old sid never resolves to nothing) is unchanged — `write_rolled_to` is still
the first of the four operations, same as pre-fix. Moving `remove_session` earlier (before the buffer
delete instead of after all four) is genuinely belt-and-braces as the comment says: because
`get_or_hydrate_session` now checks the pointer unconditionally first, the *order* of registry-evict vs.
buffer-delete no longer affects correctness — either one, or neither, being done yet is equally safe once
the pointer is down. I checked for a new race the reorder itself might introduce (e.g., a request landing
between `remove_session` and `delete_session_buffer` now vs. before) and found none: `remove_session` takes
the same `_LOCK` as `get_or_hydrate_session` (`session.py:266`), so a concurrent resolve is serialized
against it, and since the resolve's outcome no longer depends on registry state at all (pointer wins
regardless), the eviction timing is inert to correctness.

**3. Residual window — NOT closed by this delta (Major, pre-existing, not introduced by this commit).**

The C-1 fix closes the race *at the point of `get_or_hydrate_session`'s internal check*. It does not — and
by its nature as a single-call reorder, cannot — close a wider version of the same race: **a session
resolved successfully *before* a rollover starts, whose resulting turn is persisted *after* that same
rollover completes.**

Traced end-to-end from an actual request:
- `brain/bridge/server.py:2416` (`POST /chat`) resolves once: `sess = get_or_hydrate_session(...)`, then
  captures `sid = sess.session_id` and holds it for the entire request (used as the lock key, and passed
  into `_respond_blocking` → `brain.chat.engine.respond(..., session=sess, ...)`).
- Inside `respond()`, `session` is never re-resolved. `session.session_id` is threaded through
  `run_tool_loop` (`brain/chat/engine.py:270-283`) — the actual LLM call, which can involve multiple tool
  round-trips and take real wall-clock seconds — and is still the value used at the *end* of the function
  when the turn is finally written: `_persist_turn(..., session_id=session.session_id, ...)`
  (`brain/chat/engine.py:288-294`).
- `_persist_turn` calls `ingest_turn(persona_dir, {"session_id": session_id, ...})`
  (`brain/chat/engine.py:433-436`), and `ingest_turn` opens the buffer file in **append mode with no
  existence check** (`brain/ingest/buffer.py:101-103`, `open(path, "a", ...)`) — the exact
  append-creates-resurrects semantics the original C-1 report named.

So the sequence: (1) a client's request resolves `old_sid` at t0, correctly getting the live old session
because no pointer exists yet; (2) `maybe_weekly_rollover` fires concurrently for that same session (it is
gated on the session being idle ≥ `quiet_gap` — `brain/chat/rollover.py` — which a session sitting idle
long enough to be weekly-rollover-eligible plausibly satisfies right up until the moment a client reconnects
and sends a new message, i.e. exactly the sticky-session-reattach scenario this module exists to support);
(3) the rollover fully completes (pointer written, registry evicted, buffer deleted) while the client's
`run_tool_loop` call is still in flight; (4) `_persist_turn` at t1 > rollover-completion appends to
`old_sid`'s buffer, resurrecting the just-deleted file and orphaning the turn outside the successor — the
same failure mode C-1 describes, via a different trigger (temporal ordering of resolve-vs-persist across an
async request, not a check-ordering bug inside one function call).

This is **not introduced by commit `a485e72d`** — it predates the fix and would exist even with a
hypothetically perfect `get_or_hydrate_session`, because the gap is between *when* the session is resolved
and *when* the resulting turn is durably written, not in how the resolve itself is computed. It is also
**not claimed to be closed** by anything in this commit's comments once read literally (the session.py
comment says "Checking it first closes the mid-rollover window in which a stale cache entry... OR a
not-yet-deleted old buffer would otherwise shadow the redirect" — true for a resolve landing *inside* that
window, silent on a resolve landing *before* it whose write lands *after*). I'm flagging it because the
charter asked me to re-enumerate the resolve path against the rollover write ordering for "a residual window
or a new race the reorder introduced," and this is a residual window in the same failure class, discovered
while doing that enumeration — not a reason to consider C-1 itself unclosed.

- **Severity:** Major (same consequence as C-1 — a turn silently vanishes from the successor's history and
  resurrects a supposedly-deleted buffer — via a plausible real-world trigger: a user returning after being
  away long enough to be both quiet-gap- and weekly-age-eligible, racing the daily tick).
- **Not a blocker for THIS delta**: closing it needs a structurally different mechanism (e.g. re-validating
  `rolled_to` for `session.session_id` immediately before `_persist_turn`'s `ingest_turn` calls, or holding
  some lock across the full request lifetime) that is out of scope for a single-function reorder and belongs
  in its own change.

### Nitpick — unused import removal (`brain/chat/compaction.py`)

Confirmed correct. `_install_cascade_row` (`brain/chat/compaction.py:784-836`) never references `Path`
anywhere in its body — the removed line (`from pathlib import Path  # noqa: F401 — parity with module's
lazy-import style`) was genuinely dead, and its own comment admitted as much (present only for stylistic
parity, not because anything used it). The module's other lazy `from pathlib import Path` imports (lines
64, 519, 860, 974) are each inside functions that do use `Path(...)` (confirmed by grep). Clean removal, no
behavior change.

## Bottom line

C-1, as scoped by the stage-6 finding (the cache-check-before-redirect-check ordering bug inside
`get_or_hydrate_session`), is **genuinely closed**. The fix is correct, minimal, well-commented, and proven
both by the new regression test and by my own independent revert-and-reproduce check against the pre-fix
ordering. The rollover.py reordering is safe, correctly-reasoned belt-and-braces that introduces no new
race. The L2-revert decision (keeping the in-rollover `cascade_conversation`/`compact_conversation` fold) is
correct, though its code-comment justification slightly overstates the "folds nothing" claim for
already-extracted sessions — nitpick only. The unused-import removal is clean.

One Major finding surfaced during the concurrency re-enumeration: a wider, pre-existing
resolve-then-long-generate-then-persist race (`brain/bridge/server.py:2416` →
`brain/chat/engine.py:270-294` → `brain/chat/engine.py:433-436` → `brain/ingest/buffer.py:101-103`)
produces the identical orphaned-turn/resurrected-buffer symptom via a different trigger than C-1, and is not
addressed by this commit. It is not a regression introduced by this delta and does not reopen C-1 — but it
is real, plausible in the sticky-session-reattach scenario this module is designed around, and worth its own
follow-up.
