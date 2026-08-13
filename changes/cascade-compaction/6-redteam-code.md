# 6-redteam-code.md — Cascade Compaction: stage-6 cold code red-team

## Provenance

- Frozen criteria read: `changes/cascade-compaction/1.5-criteria.md`
  sha256 `6a12be35e71558237412e8faad5956960c0117bc45ebe3fed42fdee50166a914` (matches the value pinned in the
  charter — confirmed via `sha256sum`).
- Frozen plan read: `changes/cascade-compaction/2-plan.md`
  sha256 `72afd8d4d4074c752e78922a7f1d3f6b846533ba144be8df6d0de7887184e21a`.
- Reviewed diff generated mechanically (ST6d):
  `cd /home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction && git diff cd29bc61..HEAD`
  (29 files changed, 6834 insertions(+), 97 deletions(-); code files only reviewed — `changes/*.md` docs
  excluded per charter).
- Method: no shared context with the build author. Five parallel cold research passes (one per module
  cluster: `compaction.py`+`budget.py`; `rollover.py`+`session.py`; `server.py`+`supervisor.py`;
  `buffer.py`+`pipeline.py`+`compaction_migration.py`+`turn_logger.py`; the test suite) gathered
  file:line-cited facts only, no verdicts. I then independently spot-verified a sample of the highest-stakes
  citations by reading the source directly (see "Spot-verification" below) before ruling on anything, and ran
  the actual test suite + ruff myself rather than trusting the sub-passes' claims that tests pass.
- Spot-verification performed directly (not delegated): `compaction.py:675-685` (`_bucket_of_age`),
  `compaction.py:220-270` diff hunk (single-fold fallback, confirmed pre-existing/unchanged),
  `rollover.py:65-163` (`perform_rollover`, full read), `session.py:160-234` (`get_or_hydrate_session` +
  `_resolve_successor` + `remove_session`), `buffer.py:74-113` (`ingest_turn`/`list_active_sessions`),
  `budget.py` full diff, `tests/bridge/test_cascade_rollover_endpoints.py:395-491` (C21 structural check +
  pre-fix fail-demo), `tests/unit/brain/chat/test_rollover.py:140-215` (C10 interleave test),
  `tests/unit/brain/ingest/test_pipeline.py` full diff (judgment call 2). All matched their citations.
- Executed myself: `uv run pytest tests/unit/brain/chat/test_cascade_compaction.py
  tests/unit/brain/chat/test_rollover.py tests/unit/brain/chat/test_compaction_migration.py
  tests/unit/brain/ingest/test_archive_segments.py tests/bridge/test_cascade_rollover_endpoints.py
  tests/unit/brain/chat/test_compaction.py tests/unit/brain/ingest/test_pipeline.py -q` →
  **94 passed**. `uv run ruff check` on all ten touched source files → **clean**.

---

## Per-lens findings

### Lens 1 — Factual (does the code do what the plan/criteria say?)

No-issue-found items, each independently confirmed by direct read, not just cited:
- Sectioned row shape, static labels, coarse-span render, no-clock-in-render — `compaction.py:295-310`
  (`_render_sections`), `:191-195` (`_SECTION_LABELS`). Confirmed no `now()`/nonce reachable from this
  function or its `_coarse_span`/`_coarse_stamp` helpers.
- Tolerant legacy reader, unconditional 96h old-floor, never falls back to `covers_until_ts` —
  `compaction.py:313-341` (`_read_sections`). `covers_from_ts` is set from `_LEGACY_AGE_FLOOR`
  unconditionally on the fallback branch; `covers_until_ts` has its own independent fallback chain, which
  is a different field and does not affect the age classification.
- Migration mirrors the same mapping — `compaction_migration.py:271-310` (`_migrate_one_session_sections`),
  marker `.sections_migrated`, `"72h"` key literal (not `"24h"` — the historically-wrong doc value),
  idempotent via marker + already-sectioned check. Startup order confirmed at `server.py:919-935`:
  `run_sections_migration` runs before `run_backlog_migration`.
- Atomic single 3-tier write — `compaction.py:846-957` (`cascade_conversation`) reads the pre-pass snapshot
  once, computes all three tiers, calls `_install_cascade_row` exactly once (line 942), which calls
  `rewrite_session_atomic` exactly once (line 842). No loop issuing per-tier rewrites.
- Archive segmentation — `buffer.py:407-424` (roll condition, `_ARCHIVE_SEGMENT_MAX_BYTES = 5*1024*1024`),
  `:363-378` (`_fsync_dir`, same open/fsync/close/try-except-finally shape as
  `brain/health/attempt_heal.py:254-265`), `:337-361` (`_archive_segments`/`_archive_read_order`, legacy file
  read first then numeric-ordered segments, glob pattern requires two literal dots so it cannot
  double-match the legacy file).
- Five `get_or_hydrate_session` call sites, exactly five (grep-confirmed), each rebinding to a resolved
  `sid`/`session_id` before any downstream load-bearing use — `server.py:1309`/`1314`, `:2416`/`2422`,
  `:2479`/`2487`, `:2755`/`2761`, `:2816`/`2823`. Verified exhaustively that no `req.session_id` reference
  survives past each rebind point in any of the five handlers.
- Full-follow visited-set cycle guard (not a depth cap) — `session.py:216-233` (`_resolve_successor`),
  confirmed a genuine `while True` + `visited: set[str]` loop, no arbitrary bound.
- `rolled_to.json` written atomically, under the lock, before the buffer delete — `buffer.py:222-234`
  (`write_rolled_to`, tmp+fsync+`os.replace`), called at `rollover.py:151`, strictly before
  `delete_session_buffer` at `:152`.
- Finalize is extraction-only on both the success path and the empty-buffer branch; poison-quarantine path
  unchanged — `pipeline.py:558-589` (confirmed via direct diff read, not just citation).
- Terminal tier-3 multi-input fold matches all four plan sub-steps (ordered lossless-leaning join → 20%
  target → sentence-boundary cap truncation → same-join fallback on double-reject, never dropping either
  input) — `compaction.py:733-774` (`_fold_into_section`).

Discrepancy found (Minor, Factual):
- **F1 — Plan's `(24h,48h]`/`(48h,72h]` bucket notation vs. the code's actual boundary.** Plan §1.3 step 2
  writes `G24=(24h,48h]`, i.e. inclusive at 48h. The code's `_bucket_of_age` (`compaction.py:675-685`) uses
  strict `<` at both boundaries, so an item at exactly age==48h lands in `G48`/the `"48h"` bucket, not
  `G24`/`"24h"` as the bracket notation would literally imply; similarly age==72h lands in `"72h"`, not
  `"48h"`. This is fully addressed under Judgment Call 1 below — ruled sound, but it is a real deviation from
  the plan's literal prose that a naive reader would not expect, and the plan itself never edited that bucket
  notation to match the shipped `<` semantics.

### Lens 2 — Logical (bugs, edge cases, sequencing)

**Major — L1 (concurrency, see Lens "Concurrency" below for full detail; cross-referenced here as a logical
sequencing gap).** `rollover.py:150-155` writes the successor pointer, deletes the buffer/cursor/backoff, and
*only then* evicts the in-memory `_SESSIONS` registry entry (`remove_session`, line 155) — three synchronous
disk operations elapse between "buffer gone from disk" and "registry entry gone from memory." A concurrent
`get_or_hydrate_session(old_sid)` call landing in that window hits the in-memory-cache short-circuit at
`session.py:167-169` (`existing = _SESSIONS.get(session_id); if existing is not None: return existing`) and
returns the **stale** cached `SessionState` for the just-deleted buffer, **without ever consulting
`rolled_to.json`**. See Concurrency lens for the full trace and impact.

Minor — L2 (redundant work, not incorrect). `perform_rollover`'s `seed_mode="tiers_plus_tail"` branch
(`rollover.py:112-117`) calls `cascade_conversation` a second time on the same session within the same daily
tick — `_run_compaction_tick` (`supervisor.py:1652`) already called it once, unconditionally, moments earlier
in the same loop iteration, before `maybe_weekly_rollover`/`perform_rollover` runs (`supervisor.py:1659`).
Because `cascade_conversation` is idempotent when nothing new has aged (`compaction.py:908-912`, C7), the
second call is very likely a fast no-op in practice, but it is a second full lock-acquire/compute/release
cycle per weekly-rollover-eligible session per day, not mentioned as deliberate anywhere beyond the generic
"M2: seeds from just-updated tiers" comment. A plain re-read (`read_session`) would satisfy the plan's stated
"re-read the committed row" requirement without redoing the fold. Not a correctness defect — flagged as a
missed-opportunity-adjacent logic smell.

Nitpick — L3. `_install_cascade_row` (`compaction.py:796`) has an unused `from pathlib import Path  #
noqa: F401` local import — genuinely dead in that function body (confirmed: no `Path(...)` call anywhere in
`_install_cascade_row`), unlike the same import at `compaction.py:862`/`976` which are used. Harmless,
ruff-silenced.

No issue found: idempotence (C7), the archive-before-rewrite gate on cascade failure (`_install_cascade_row`
returns `None`/`CascadeResult(False, ..., reason="archive_failed")` on a zero-byte archive write, buffer left
untouched — `compaction.py:814-828`/`944-947`), the quiet-gap defer logic (`rollover.py:165-201`, both age
and quiet-gap conditions independently gate, confirmed by direct read matching the C9 test's own fail-demo
that flips `_ROLLOVER_QUIET_GAP` to zero and shows the swap would otherwise have fired).

### Lens 3 — Missed opportunity

- The redundant `cascade_conversation` call noted under Lens 2/L2 above is the clearest missed opportunity:
  threading the already-computed tier state from `_run_compaction_tick`'s cascade call through to
  `perform_rollover` (or reducing the rollover's own step to a plain re-read) would remove a full duplicate
  fold-compute + lock cycle from the hot weekly-rollover path with no loss of the interposition-safety
  property the plan cites.
- The C21 structural check (`tests/bridge/test_cascade_rollover_endpoints.py:411-451`) is a source-text regex
  over `server.py`, not an AST-based check. It correctly catches the current 5 sites and is proven non-vacuous
  against the pre-fix commit, but a differently-styled correct fix (rebind folded into a compound expression,
  a differently-named local, or `sess.session_id` used inline without an assignment) would false-negative
  past it. Given C21's whole stated purpose is "so a future 6th caller can't silently regress the class," a
  regex tied to one specific rebind-statement shape is a narrower guarantee than the criterion's stated intent
  ("static/structural check... a future 6th caller cannot silently regress"). Not a defect in the reviewed
  code today — a missed opportunity to make the guard itself more robust (e.g. an `ast`-based check that finds
  variable definitions/uses rather than matching literal source lines).

### Lens 4 — Unstated assumptions & risks

- **A1 — The single-fold (`compact_conversation`) double-reject fallback is *pre-existing*, unchanged by this
  diff, and behaves differently from the new cascade fallback.** Confirmed via diff: the
  `new_part = f"[truncated {len(removable)} earlier messages]"` placeholder-note fallback
  (`compaction.py:249-253`) was already present before this change; the diff only replaced the direct
  `provider.generate(...).strip()` call with the new validated/retried `_generate_validated_fold` wrapper
  feeding into that *same, pre-existing* fallback. So the single-fold path (used by 1c-A's full-conversation
  idle-rollover fold, and by the unmodified `run_backlog_migration`) now routes *more* trigger conditions
  (refusals, non-first-person output — not just provider exceptions) into a fallback that discards the actual
  text of the folded raw batch and keeps only a count, whereas the brand-new cascade fallback
  (`_fold_into_section`) preserves the actual joined text of its inputs. This is not a correctness defect per
  C4's own oracle wording ("(a) single-fold: ... assert not-stored + one retry + prior value unchanged on
  repeat" — which this satisfies: `prior_text` is never overwritten with garbage) and raw content is not lost
  system-wide (it was already archived before the fold, per the unchanged archive-before-rewrite gate at
  `compaction.py:600-612`). But it is an unstated asymmetry: the plan's C4 prose ("source material is
  preserved, never dropped... never stored via bare `.strip()`") reads as a single unified requirement, and a
  reader could reasonably expect the single-fold fallback to also preserve the actual raw text now that it
  goes through the same validator, when in fact it doesn't (and never did) — worth an explicit note in the
  plan/decisions record rather than leaving it implicit. Minor.
- **A2 — apply_budget's `provider=` kwarg is silently ignored, both before and after this change** (judgment
  call 4, ruled below) — a caller reading the signature would reasonably assume passing a custom provider
  changes which model performs the emergency fold; it does not. Pre-existing, out of scope, but the risk of a
  future caller relying on it is real and unflagged in code (no deprecation comment on the parameter itself).
  Nitpick/Minor — a one-line comment on the parameter would close this cheaply.

### Lens 5 — Fidelity (does the code implement the owner mechanism, or a proxy?)

- **Terminal tier-3, multi-input fold**: real mechanism, not a proxy. Confirmed by direct read of
  `_fold_into_section` (`compaction.py:733-774`) — genuine ordered join of persisting-tier3-prose +
  graduated-tier2-prose + G72-raw, oldest-first, both on the success path and the double-reject fallback path.
  Loaded terms this pins: "terminal" = the prior tier-3 section always reclassifies back into the 72h band and
  is folded again (never evicted, no 4th tier, no evict leg — confirmed no such branch exists anywhere in
  `cascade_conversation`); "multi-input" = `classified["72h"]` can and does hold two section inputs
  simultaneously (verified via the C14 multi-input test, which independently confirms both markers survive).
- **Oldest-edge graduation**: real mechanism. `bucket_of` classifies by `covers_from_ts` age
  (`compaction.py:900-902`), never by `covers_until_ts`/newest edge, and the prior-24h section is never
  co-folded with fresh raw (confirmed: `_bucket_of_age` reclassifies a section by its own `covers_from_ts`
  before it's ever combined with a raw group of the SAME target band, and the two are only combined once
  already co-classified — no re-fold-with-fresh-raw path exists for a section that hasn't aged into the target
  band).
- **Full-follow redirect at the real chokepoint**: mechanism is real for the steady-state case (verified: a
  genuine visited-set loop, multi-generation 3-hop test passes, cyclic-pointer aborts to 404). **However, the
  claim that `get_or_hydrate_session` is "the real resolution chokepoint" that *always* consults
  `rolled_to.json`** is not fully true — see the Major concurrency finding below: the in-memory `_SESSIONS`
  cache check (`session.py:167-169`) is checked *first* and can short-circuit past the pointer-consultation
  logic entirely during a narrow post-delete, pre-registry-evict window. The mechanism is real when it runs,
  but the code does not guarantee it always runs before returning a session. This is a partial-fidelity gap,
  not a full proxy — most of the time (no in-flight rollover racing the call) the mechanism is exactly as
  specified.
- **Old-floor migration**: real mechanism, not a proxy — confirmed unconditional 96h floor, "72h" not "24h",
  both the migration function and the runtime tolerant-reader independently implement (not literally
  duplicate, but structurally mirror) the same mapping, and the self-heal-on-flatten interaction (MO-3) is
  exercised end-to-end by `test_c12_delayed_backlog_flatten_self_heals_to_tier3`.

---

## Position lens (fires — head render + budget re-parse)

No issue found. `_render_sections` (`compaction.py:295-310`) takes only a `sections` dict, no `now` parameter,
and neither it nor its `_coarse_span`/`_coarse_stamp` helpers call any clock function — confirmed by direct
read of all three functions, not just the C6 test's claim. Two renders of the same `sections` object are
therefore byte-identical by construction (also independently verified by the C6 test's own noise-injection
fail-demo, which breaks equality when a live timestamp is spliced in). Labels are a static dict
(`_SECTION_LABELS`, `compaction.py:191-195`), not computed. `budget.py`'s diff (`git diff cd29bc61..HEAD --
brain/chat/budget.py`) touches only which compaction function is called (`compact_conversation` →
`emergency_fold_24h`) and drops one now-unused `older_than`/`fold_existing_summary` pair of arguments — it
does **not** touch `_COMPACTION_SUMMARY_PREFIX` or any re-parse logic, so the prefix contract is genuinely
untouched by this change. C6 and C22's tests exercise the real `apply_budget` re-parse against a sectioned row
and both pass under direct execution (confirmed by my own pytest run).

## Concurrency lens (fires)

**Accessor enumeration performed independently** (cross-checked against the plan's own §5 table): session
buffer (disk), `_SESSIONS` registry (memory), `in_flight_locks`, ingest cursor, archive segments,
`rolled_to.json`. All writers found match the plan's enumeration **except one gap**, below.

### MAJOR — C-1: `get_or_hydrate_session`'s in-memory registry check bypasses the `rolled_to.json` redirect during a real, unguarded post-delete window in `perform_rollover`

**Mechanism.** `perform_rollover` (`rollover.py:150-155`) executes, in order, under the file-based compaction
lock:
```
151: write_rolled_to(persona_dir, old_sid, new_sid)      # pointer written
152: delete_session_buffer(persona_dir, old_sid)         # buffer file deleted from disk
153: delete_cursor(persona_dir, old_sid)
154: delete_backoff(persona_dir, old_sid)
155: remove_session(old_sid)                             # in-memory _SESSIONS entry evicted
```
`remove_session` (`session.py:250-259`) acquires session.py's **separate** in-memory lock (`_LOCK`, a
`threading.RLock`) only for the instant of the dict `.pop()` — it is not held for the duration of
`perform_rollover`. Between line 152 (buffer gone from disk) and line 155 (registry entry gone from memory),
`old_sid` — if it was already resident in `_SESSIONS` from a prior request, which is the expected case for
1c-B's explicitly-stated target population ("the continuously-used client that never re-attaches", plan §0) —
remains present in `_SESSIONS`.

`get_or_hydrate_session` (`session.py:166-169`) checks the in-memory registry **first**, under `_LOCK`
(a *different* lock from the compaction lock rollover holds):
```python
with _LOCK:
    existing = _SESSIONS.get(session_id)
    if existing is not None:
        return existing
```
A concurrent call to `get_or_hydrate_session(old_sid)` landing in the line-152–155 window acquires `_LOCK`
uncontended (rollover never holds it), finds the stale entry, and returns it immediately — **without ever
reaching the `rolled_to.json`-consulting branch** at `session.py:171-181`. The returned `SessionState` still
reports `session_id == old_sid`, pointing at a buffer file that no longer exists.

**Impact, traced to the actual write path.** `ingest_turn` (`buffer.py:74-104`) opens its target path in
append mode (`open(path, "a", ...)`, line 102), which **creates the file if absent**. A `/chat` request that
resolved to the stale `old_sid` state in this window (`server.py:2416`, `sid = sess.session_id` at line 2422
would be `old_sid`) will silently **resurrect** `active_conversations/<old_sid>.jsonl` with a single fresh
turn — a turn that is now orphaned from the successor timeline the client's other requests are landing in,
invisible to `rolled_to.json`'s chain (which points *from* `old_sid`, not into it), and will sit undetected
until some future sweep (e.g. a fresh weekly-rollover cycle, if `old_sid` accumulates enough age) reaps it.
This is exactly the class of finding the charter's concurrency lens asks for: a resolution gap producing a
lost/misrouted write, not merely a lock contention slowdown.

**Why this wasn't caught.** The plan's own §5 concurrency table has a row for "rollover swap... (extract→
archive→delete→seed→rolled_to) | W | compaction lock (both paths acquire it) | yes — both rollover paths hold
the compaction lock across the whole swap" — this claims full coverage, but it is evaluated against the
**session buffer** accessor only. The table's `_SESSIONS` registry rows cover `/sessions/close`'s
`remove_session` call and finalize's `remove_session` call (with an explicit "M1: harmless... a still-live
session evicted here re-hydrates from disk... **no buffer effect (finalize no longer deletes buffers)**"
justification) — but there is no row for rollover's *own* `remove_session` call, whose safety argument would
need to be different from M1's, precisely because in rollover's case **the buffer has already been deleted**
by the time the registry entry is evicted — the one condition M1's own reasoning explicitly relies on being
false to call itself harmless. None of the C16/C19/C20 tests target this window; all of them run rollover to
full completion first, then issue a request against the old sid, which never races the delete-then-evict
gap.

**Severity: Major, correctness/mechanism (concurrency), not test-quality.** Narrow window (three synchronous
disk operations, likely low-single-digit milliseconds), and the consequence (one orphaned turn, eventually
reaped) does not corrupt other sessions or crash the process — hence Major rather than Blocker. But it is a
genuine, previously-unflagged violation of the plan's own "no window where the old sid resolves to nothing"
invariant (worse, in fact: it resolves to *something* that silently accepts writes against a deleted buffer,
which is a worse failure mode than resolving to nothing). Routes back to stage 5 per the severity table.
**Suggested direction for the fix** (not prescribing the implementation): either hold `session.py`'s `_LOCK`
across rollover's delete+evict sequence (risk: cross-lock-type nesting with the compaction lock — needs its
own ordering analysis), or evict the registry entry (`remove_session`) *before* deleting the buffer/cursor so
a racing lookup that hits the stale cache still resolves to a buffer that, if briefly still present, is at
worst momentarily inconsistent rather than silently write-resurrected; or make `get_or_hydrate_session`'s
registry hit re-validate against `rolled_to.json` even when the id is cache-resident. Any of these needs its
own red-team pass — not resolved here.

No further concurrency issues found. Specifically confirmed sound: the two-separate-acquire-release-cycles
lock model (`compaction.py:865`/`957` for the fold, `rollover.py:124`/`162` for the rollover, never a single
continuous hold — matches plan §2.2's corrected round-6 F3 description); the re-read-under-lock at
`rollover.py:127` (`read_session` executes after the lock acquire, not before, so an interposing `apply_budget`
call is safely picked up); `in_flight_locks` uniform keying by resolved sid at all five sites
(`server.py:2423`, `:2488`, `:2762`, `:2833`, and the `/state` in-flight read at `:1315`); `rolled_to.json`
written-before-delete ordering; archive segment roll-and-fsync under the same compaction lock as all other
archive appenders (append-during-roll genuinely lock-precluded, matching plan §3's claim); the crash-mid-roll
simulation in `test_c11_archive_segments_reader_crash` (directly read, confirmed it writes a zero-length
next-segment file plus a torn trailing line and shows the reader still recovers the full prior chain).

---

## Criteria-gating check (sample: C14, C16, C21, C12 — plus notes on C10)

- **C14** — `test_c14_graduation_and_terminal_persistence` + `test_c14_multi_input_and_long_inactivity`
  (`tests/unit/brain/chat/test_cascade_compaction.py`). Drives the real `cascade_conversation` with a
  marker-preserving (non-tautological) fake provider; runs 5 consecutive daily passes and explicitly asserts
  the marker is **absent** from every non-expected tier at each pass (not just present in the expected one),
  covering terminal persistence on passes 4 and 5; a separate 4-day steady-state loop confirms both the
  persisting-tier3 and newly-graduated-tier2 markers co-survive within cap. **Gap**: the criterion's
  "multi-day-gap sequence → markers land by true age" sub-clause is covered only by a single-pass
  long-inactivity fixture, not a genuine multi-pass sequence with a gap *between* cadence ticks — this
  sub-clause's coverage is thinner than the rest of C14 but not absent. Minor test-quality gap, not a
  correctness defect (the underlying `_bucket_of_age`/age-partition mechanism is age-pure and doesn't
  distinguish "gap" from "steady-state" in its own logic, so the missing scenario is unlikely to hide a
  real bug, but it is a literal gap against the criterion's own stated oracle).
- **C16** — real `TestClient` throughout (confirmed by direct read of the multi-generation test), not the
  disqualified bare `ingest_turn` proxy. Three successive `write_rolled_to` hops confirmed, resolved via both
  a direct `get_or_hydrate_session` call and a `POST /chat` through `TestClient`; cyclic pointer aborts to a
  genuine 404, not a hang (confirmed via direct read of the C21 section's neighboring C16 cyclic test). Real
  path, real assertions, no proxy found.
- **C21** — the structural check is a real, non-vacuous mechanism: verified by direct read that
  `test_c21_flags_all_five_sites_on_pre_fix_base_commit` loads the actual pre-fix `server.py` blob via
  `git show cd29bc61:...` and re-runs the identical check function against it, asserting all 5 named handlers
  are flagged. This is the H6 "shown able to fail" requirement satisfied *inside* the test suite itself, not
  merely asserted in prose. Caveat (already covered under Lens 3/Missed opportunity): the check is
  regex/text-based, not an AST/data-flow check, so it is narrower than "no future 6th caller can silently
  regress" in the fully general case — it catches the specific rebind idiom used today.
- **C12** — real `run_sections_migration`, real `_read_sections`, real `cascade_conversation` chained
  together across all four sub-tests, including the fallback-branch "delayed backlog flatten" self-heal
  scenario (`test_c12_delayed_backlog_flatten_self_heals_to_tier3`), which is the load-bearing MO-2/MO-3
  regression guard. No proxy found.
- **C10** — real `finalize_stale_sessions` and real `perform_rollover`, but the interleaving is **sequential
  (finalize fully completes, then rollover runs)**, not a literal injection into the middle of the rollover's
  write window as the criterion's oracle text asks for ("inject a finalize tick into the middle of its
  window"). The test's own docstring is honest about this ("the allowed 'before/around' interleaving").
  Given finalize is now *unconditionally* non-deleting (verified: no code path in the current
  `finalize_stale_sessions` deletes a buffer except the unchanged poison-quarantine branch), a literal
  mid-write injection would not exercise a materially different code path than the before/after ordering
  already tested — the decoupling makes the timing distinction largely moot for *this* implementation. Rated
  Minor/test-quality (oracle-literalism gap), not a correctness concern, since the property actually being
  protected (finalize never deletes) holds regardless of interleave timing and is proven both by direct
  assertion and by the H6 fail-demo against the real pre-change commit
  (`test_c10_pre_change_finalize_deleted_buffer_fail_demo`, confirmed via direct read to `exec()` the actual
  historical `pipeline.py` from `git show c2154a97^:...`).

---

## Judgment-call rulings

1. **Age-boundary `<` vs `≤` — SOUND.** `_bucket_of_age` (`compaction.py:675-685`) uses strict `<` at both
   the 48h and 72h boundaries; the top-level raw-turn eligibility gate (`compaction.py:886`) separately uses
   `<=` (age ≥24h eligible). Verified the bucket function is total and mutually exclusive across its three
   branches — at every exact boundary (age==48h, age==72h) exactly one bucket is chosen; nothing is dropped
   from all buckets and nothing double-counted into two. The effect of the boundary choice is that an
   exact-boundary item lands one tier *older* than the plan's literal `(24h,48h]` bracket notation would
   suggest — but this strict-`<` choice is not arbitrary: it is exactly what makes C14's oracle
   ("marker sown day 0 is in tier1 after 1 pass, tier2 after 2, tier3 after 3") hold at the literal pass
   boundaries, and I independently confirmed `test_c14_graduation_and_terminal_persistence` exercises and
   passes this exact mapping. The plan's own bucket-notation prose is the looser artifact here (never edited
   to match the shipped semantics — see Lens 1/F1), not the code. Ruling: correct and consistent across the
   classifier, the raw partition, and graduation; no cohort is dropped or double-counted at any boundary.

2. **Updated existing tests — SOUND overall, with one minor caveat.**
   - `test_c1a_over_cap_backstop_fires_once_not_per_turn`: the new assertion
     (`row["compaction"]["sections"]["24h"]["text"] == "SUM"` plus `"SUM" in row["text"]`) is a faithful, if
     anything *stronger* (exact-match on the structured field, not just substring on the flat text) adaptation
     to the new sectioned contract. Not loosened.
   - The two `test_pipeline.py` finalize tests: both flips (delete-assertion → `buf.exists()`) are a genuine,
     correct reflection of the new extraction-only contract (independently confirmed against
     `pipeline.py:558-589`'s actual code, which has no delete path outside poison-quarantine). Not loosened
     to paper over a bug. **Caveat**: in isolation, neither modified test independently re-asserts that
     extraction *succeeded* (no `extract_calls`/`store.count()` check — only `len(reports)==1` and
     `reports[0].session_id==sid`, both pre-existing and satisfied even if extraction internally raised and
     was caught). This gap is closed elsewhere in the suite — the dedicated
     `test_c10_finalize_no_delete_interleave` does assert `provider.extract_calls == 1` and
     `store.count() == 1` — so the criterion (C10) is genuinely gated, but these two specific renamed tests,
     read on their own, verify less than their new names imply. Minor, not defective.

3. **Resolved-sid echo — SOUND.** All five handlers echo the resolved sid, confirmed by direct trace of every
   response body (`server.py:1317`, `:2453`, `:2567`/`2651`/`2658`/`2668` for the WS frames, `:2792`, `:2916`).
   No test or code path found that expects the *raw* client-sent sid to be echoed back in a redirect scenario;
   to the contrary, C16's own oracle explicitly requires the response to report the successor sid
   (`session_id == sid2`), so echoing the resolved id is not merely harmless but required by the criterion
   itself. The plan's "may still echo the original where that's what the client sent" carve-out is not
   exercised by any handler today (every echo point is post-rebind) — consistent with, not contradicting, the
   plan (that clause was phrased as a permission, not a requirement).

4. **`apply_budget`'s `provider=` dead parameter — SOUND (correctly identified as pre-existing/out-of-scope).**
   Confirmed via `git diff` against baseline `cd29bc61`: the parameter was unused in the function body both
   before and after this change (before: passed to `compact_conversation` as `provider=` but the *bound*
   parameter itself was never referenced elsewhere in the body either — direct grep confirms `provider` never
   appears as a bare reference in `apply_budget`'s body in the pre-change source; after: the pattern is
   identical, just calling `emergency_fold_24h` instead). The cascade change does not newly wire this
   parameter through to anything and does not make it load-bearing. The `test_c22_apply_budget_sectioned_row`
   test independently rediscovered and documented the same fact (it has to monkeypatch
   `build_compaction_provider` rather than pass a stub through the public `provider=` kwarg). Correctly
   out of scope for this change; worth a follow-up nitpick (a one-line deprecation/dead-param comment) but not
   a defect in this diff.

5. **New `brain/chat/rollover.py` module — factoring is sound; the lock/ordering contract it documents is
   accurately implemented, EXCEPT for the registry-eviction race identified as the Major concurrency finding
   above.** The module split itself does not introduce any lock/ordering problem — the two-separate-cycles
   model, the re-read-under-lock, and `apply_budget` as a genuine concurrent lock caller are all confirmed
   exactly as the plan describes. The defect found is not attributable to "factoring into a new module" as
   such; it is a genuine gap in the concurrency design (the `_SESSIONS` in-memory registry was never brought
   under the same synchronization discipline as the on-disk buffer state it now needs to stay consistent
   with) that would exist whether or not `rollover.py` were a separate file. Flagging this judgment call as
   **defective as asked** ("does the new module... break the lock/ordering contracts") only insofar as the
   contract itself — as actually implemented — has the gap described above; the factoring decision is not the
   cause.

---

## Bottom line

**Worst severity: Major** — one concurrency/correctness defect (C-1: the in-memory `_SESSIONS` registry check
in `get_or_hydrate_session` can return a stale, already-deleted-buffer session state during a real, narrow,
previously-unflagged window in `perform_rollover`'s delete-then-evict sequence, silently bypassing the
`rolled_to.json` redirect and, via `ingest_turn`'s create-on-append semantics, resurrecting and writing into
the just-deleted old buffer). Everything else found is Minor or Nitpick (test-quality gaps, redundant work, a
regex-based structural check narrower than fully general, an unstated fallback asymmetry, a pre-existing dead
parameter correctly called out as such).

All 94 targeted new/modified tests pass under direct execution; `ruff check` is clean on all ten touched
source files. The mechanism fidelity is otherwise strong: terminal multi-input tier-3 fold, oldest-edge
graduation, old-floor migration, and the full-follow redirect chain are all genuinely implemented (not
proxied), and the criteria-gating sample (C10, C12, C14, C16, C21) shows real production-path tests for
19 of the 22 gating criteria, with only C10's interleave-timing and C14's multi-day-gap sub-clause showing
oracle-literalism gaps that are Minor, not correctness-hiding.

**Routing: back to stage 5** for the Major concurrency finding (C-1). The other findings (L2 redundant
cascade call, the C21 check's regex specificity, the C4 fallback asymmetry, the C10/C14 oracle-literalism
gaps, the two under-verifying pipeline.py tests, the dead `provider=` param) are Minor/Nitpick and may be
batched into the same stage-5 pass or deferred at the owner's discretion — none of them independently
gate.
