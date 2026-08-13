# decisions.md — cascade-compaction (gate log, append-only)

## Run-start path validation (CFG3) — 2026-08-13

`redteam_context` (from `guarded-change.companion.md`):
- OK   `~/Desktop/companion-emergence/brain` (source of truth clone)
- OK   `~/Downloads/Phoebe/chat_usage.jsonl` (stale read-only)
- OK   `~/Downloads/Phoebe/tool_invocations.log.jsonl` (stale read-only)
- OK   `~/companion-token-trace/` (priors)

Spec touched-files (validated in the worktree `.claude/worktrees/cascade-compaction`):
- OK   `brain/chat/compaction.py` (426 ln)
- OK   `brain/ingest/buffer.py` (424 ln)
- OK   `brain/bridge/server.py`  (NOT brain/chat/server.py — brief loci corrected; `/sessions/active` :1203, `_ATTACH_MAX_AGE_HOURS` :1224)
- OK   `brain/ingest/pipeline.py` (finalize tick — `finalize_after_hours` :532)
- OK   `brain/bridge/supervisor.py` (daily tick + finalize scheduling)
- OK   `brain/chat/engine.py` (448 ln)
- OK   `brain/chat/budget.py` (118 ln)
- OK   `brain/chat/compaction_migration.py` (233 ln)
- OK   `brain/monologue/ambient.py` (45 ln — interior-continuity read)

All paths readable. No dead paths. Recorded for gate 4.

## Gate entries

### Gate 4 (plan) — 2026-08-13 — worst finding: Major → route to stage 2
Stage-3 cold red-team (general-purpose, opus) returned a **Major** worst finding (tie: F1 tick-shift proxy /
L2 finalize buffer-leak). Route: **return to stage 2** for revision, then re-red-team before build.
Run-start path validation: recorded above (all OK). Criteria NOT frozen (freeze only on route-to-build).
Orchestrator routing on the two escalate-worthy findings: BOTH conform-to-brief, NOT escalated —
F1→restore age-gating (brief §Decisions-2 line 45); F2→drive weekly from daily cadence (brief §1c-B line 68).
A1 SYNC kept (owner-ratified brief §1c-A); resolve stale-selection + sync-bound sub-points in-plan.
Owner directive (usage): all subsequent cold reviewers run on **sonnet** (was opus); haiku still fine for
mechanical sub-agents. Applies to the stage-3 re-run and stage-6.
Revision targets: F1/L1, L2, A1-subpoints, F2/M1, L3, C11-crash, nitpick — see `3-redteam-plan.md` disposition.

### Gate 4 (plan) — round 2 — 2026-08-13 — worst finding: Blocker → route to stage 2 (narrow)
Stage-3 RE-RUN (general-purpose, **sonnet**) confirmed all 7 carried findings RESOLVED; found one new
**Blocker P-1** (the `rolled_to.json` weekly-swap redirect never reaches the real 404 locus
`get_or_hydrate_session`, and C16 was a proxy test) + 4 minors + a nitpick. Route: **return to stage 2,
narrowly scoped** — fix P-1 via option (a) (server-contained redirect in `session.py`) + the minors, then
re-red-team. NOT an owner escalation (brief specifies no redirect mechanism → author's design; option (a)
chosen per orchestrator guidance; a client-visible contract change (option b) would STOP-for-owner but is
avoided). Criteria NOT frozen. Iteration-cap note: this is a *different* finding class (P-1 redirect-locus)
than round 1 (F1/L2/A1) — not a repeat bounce; cap not implicated.
New/changed criteria: C16 re-pointed to the real path; +C17 (atomic cascade write), +C18 (seed cursor
prevents re-extraction). Touched-files expanded: +`brain/chat/session.py` (Hana zone — PR #58 coordination).

### Gate 4 (plan) — round 3 — 2026-08-13 — worst finding: Major → route to stage 2 (narrow)
Stage-3 re-run (general-purpose, **sonnet**) confirmed ALL round-2 carried findings resolved (P-1 genuinely
fixed at the correct locus incl. RLock reentrancy); found 3 new **Majors** (F1 redirect incomplete for
close/snapshot; F2 `in_flight_locks` key-split; F3 age-laundering under continuous use) + minors (F4 cursor
race, snapshot_stale enumeration, C4 wording, cap ack, remove_session no-op, dead code). Route: **return to
stage 2, narrow.** None an owner escalation (all conform-to-design or engineering fixes). Full verbatim record:
`3-redteam-plan-r3.md` (filed by orchestrator; disposition appended there).
Criteria: +C19 (`in_flight_locks` resolved-sid keying); C14 strengthened (graduation, non-tautological);
C16 broadened (4 handler sites). Touched-files: `brain/bridge/server.py` (4-site resolved-sid) already listed.
**ITERATION-CAP (SEV4) — AT THE CAP ON TWO CLASSES:** cascade-promotion (round-1 F1 + round-3 F3 = 2 bounces)
and rollover/redirect-lifecycle (round-2 P-1 + round-3 F1/F2 = 2 bounces). A 3rd bounce on either in round 4
triggers the human tie-break. Both were resolved at mechanism root this lap. Reported to orchestrator.
Context-economy change (owner, from round 4 on): the cold reviewer writes its OWN full review to
`3-redteam-plan-r<N>.md` and returns only a short structured verdict.

### OWNER DESIGN RULINGS — 2026-08-13 (tie-break; attributed to the owner via the orchestrator)
The owner resolved the iteration-cap tie-break and clarified the cascade design (the round-3 "F3/G2" concern
was a spec-comprehension gap, not a loop-convergence failure → **cascade-promotion cap-trip RETIRED**). Do NOT
split the PR (weekly-rollover stays). Authoritative cascade design:
1. **THREE tiers, human labels in the render:** tier1 (24h)="yesterday"; tier2 (48h)="day before yesterday";
   tier3 (72h)="a few days ago".
2. **Tier 3 is TERMINAL** — no 4th tier, no evict/archive-out-of-summary. Material graduates raw→1→2→3 and then
   **stays in tier 3**, re-compacted every cycle so the oldest content fades to an ever-briefer trace
   ("re-compacted forever" = the intended gradual fade, NOT a leak). The prior "falls off the bottom → archived
   out" leg is REMOVED.
3. **C14 corrected:** assert material graduates 1→2→3 across passes, then **PERSISTS in tier 3 (fading)** — NOT
   "gone/archived after pass 4"; AND after long inactivity only tier 3 is populated (tiers 1 & 2 empty). Keep
   the graduation-by-true-age (oldest-edge classify + no-remerge) from the round-3 fix.
4. **Tier 3 hard cap:** `_SECTION_72H_CHAR_CAP = 0.20 × _SECTION_24H_CHAR_CAP`, enforced like tier1
   (sentence-boundary truncation). Tier 2 stays bounded-by-input (transient; input is the tier1-capped section).
5. **Inactivity/catch-up:** after any inactivity the cascade rebuilds straight down to tier 3 (only tier 3
   populated) — the age-bucketing already handles this.
### Gate 4 (plan) — round 4 — 2026-08-13 — resolved by OWNER tie-break → revise + round 5
Round-4 sonnet re-review (full record `3-redteam-plan-r4.md`) surfaced the cascade-graduation question that
tripped the SEV4 cap on the cascade-promotion class. **The owner broke the tie** (rulings above): the design
is now pinned (terminal tier 3), the "F3/G2" concern was a spec-comprehension gap → **cap-trip RETIRED**. One
genuine Major (**G1**, close-cleanup leak) + M1/M2 minors remained, all engineering fixes I own. Route: revise
per owner rulings + G1/M1/M2, then ONE more sonnet re-review (round 5); if minors-only/clean → BUILD (owner
go-ahead, no bounce-back). Criteria: C3→tier1+tier3 caps; C5→labels; C14→terminal persistence; +C20 (G1).
Iteration count so far: **4 gate-4/stage-3 rounds** logged; both prior cap-at-2 classes are now cleared
(cascade by owner ruling; redirect-lifecycle by the round-4 fixes converging — round-5 confirms).

Also: **G1 (Major, real):** `/sessions/close` cleanup uses the pre-redirect id at `server.py:2835`
(`remove_session(req.session_id)`) + `:2836` (`in_flight_locks.pop(req.session_id)`) → successor registry+lock
leak; use the resolved `sess.session_id`. **M1:** enumerate `supervisor.py:1686` finalize `remove_session`
in §5 (harmless — registry-cache eviction, re-hydrates from disk). **M2:** specify same-tick ordering of
weekly-rollover vs cascade fold. **Go-ahead:** after ONE more sonnet re-review (r5), if minors-only/clean,
proceed to BUILD (stages 5-6) without bouncing back; report at the build-done+CI-green+cold-reviewed gate.

### Path validation (CFG3) — round-3 spawn — 2026-08-13
New touched-file since gate-4 round 2: `brain/chat/session.py` — OK (268 ln, readable). All prior paths still OK.

### Gate 4 (plan) — round 5 — 2026-08-13 — worst finding: Major → route to stage 2 (narrow)
Round-5 sonnet re-review (full record `3-redteam-plan-r5.md`) judged the owner design faithfully implemented;
found 2 new Majors + minors: **F1** (terminal tier-3 is a steady-state MULTI-INPUT fold — persisting tier3 +
graduated tier2 each cycle — which was unspecified → risk of dropping owner-protected content) and **F2**
(redirect whack-a-mole: a 5th `get_or_hydrate_session` caller `/state` missed — sweep ALL callers + prefer a
root/structural fix). Minors: N1 (stale 960-char Q8 figure), M-1/M-2/F3 (multi-input fold semantics doc,
apply_budget×sectioned-row test, same-tick lock-continuity mechanism). All engineering fixes I own.
Resolution: F1→explicit multi-input lossless-leaning tier-3 fold (§1.3) + C14/C3 force salient content from
BOTH inputs to survive within the cap. F2→enumerated ALL 5 callers (`/chat`,`/stream`,`/sessions/close`,
`/sessions/snapshot`,`/state`), uniform rebind to `sess.session_id` downstream + **new C21 structural guard**
(no raw id downstream of resolution) so round 6 can't find a 6th site; +C22 (apply_budget×sectioned row);
N1 fixed; M2 mechanism stated (single lock hold across fold-then-rollover). Then round-6 re-review; if
minors-only/clean → BUILD (standing owner go-ahead). Iteration count: **5 gate-4/stage-3 rounds.**
sha256 (round-5-revised): pending round-6 spawn record.

### Gate 4 (plan) — round 6 — 2026-08-13 — worst finding: Major → route to stage 2 (narrow)
Round-6 sonnet re-review (full record `3-redteam-plan-r6.md`) independently re-verified F1 (terminal
multi-input fold) + F2 (redirect: grepped whole worktree — exactly 5 sites, no 6th, C21 guard holds → the
whack-a-mole class is CLOSED). Two new Majors, both bounded, neither touching the owner-pinned design:
- **MO-1 (Major, migration):** the migration/tolerant-reader synthesized `covers_from_ts = covers_until_ts`,
  so the oldest-edge classifier would mislabel EVERY existing persona's months-old history as "yesterday" on
  first migration — reproducing #82 for 100% of production personas. Fix: legacy blob defaults to **tier 3**
  (derive oldest turn ts if available). C12 extended with the cascade-on-migrated-output #82-guard.
- **L-1 (Major, redirect chain):** inconsistent "one hop" vs "cap N" → pinned to **full-follow the
  `rolled_to.json` chain to its live successor** (visited-set cycle guard, not a depth cap). C16 gains a
  multi-generation test (sid1→sid2→sid3, client on sid1 resolves to sid3).
Minors: **F3** (corrected the "single continuous lock hold" claim — the compaction lock is non-reentrant/
self-contained per call, `apply_budget` is a real concurrent caller; correctness is by the rollover **re-read**
of the current row, not lock continuity); **UA-2** (re-pointed C18 from the speaker-filter-protected seed
summary row to the real risk — 1c-B's carried 40-msg raw tail extraction state).
All engineering fixes I own. Then round-7 re-review; if minors-only/clean → BUILD (standing go-ahead).
**HARD STOP-CONDITION (owner, new):** if round 7 surfaces ANOTHER new Major in the weekly-rollover/
successor-redirect class (NOT migration, NOT cascade), HALT + report — trigger to split weekly-rollover into
its own follow-up PR and ship the core cascade alone. MO-1 + cascade-side fixes proceed regardless.
Iteration count: **6 gate-4/stage-3 rounds.**

### Gate 4 (plan) — round 7 — 2026-08-13 — worst finding: Major (migration class) → route to stage 2 (narrow)
Round-7 sonnet re-review (full record `3-redteam-plan-r7.md`): the **redirect/weekly-rollover class is CLOSED**
— L-1 (chain full-follow + cycle guard + multi-gen), F3, UA-2 all resolved + independently re-verified; the
redirect stop-condition **never fired** (stays retired). One Major left, correctly in the **migration** class:
- **MO-2 (Major, migration):** deriving `covers_from_ts` with a `covers_until_ts` fallback lets the classifier
  reclassify a legacy blob to tier 1 next pass (#82 again). Structural fix: legacy blob → tier 3 with
  `covers_from_ts = now − _LEGACY_AGE_FLOOR (96h)` **unconditionally**, never `covers_until_ts`; optional
  primary path = **archive scan** (never buffer). Dropped the false "conservative floor / can only fade"
  claim; replaced with the structural argument. C12 gains the fallback-branch survives-next-cascade-pass test.
Then round-8 re-review; if minors-only/clean → BUILD (standing go-ahead).
**Migration-class stop-condition (owner, new, mirrors the retired redirect one):** if round 8 surfaces ANOTHER
new Major in the migration/tolerant-reader class, HALT + report (signals the migration approach needs a rethink
or scope carve-out). Redirect stop-condition stays retired. Iteration count: **7 gate-4/stage-3 rounds.**

### Gate 4 (plan) — round 8 — 2026-08-13 — MO-2 resolved; MO-3 = stale-doc cleanup (override) → route to stage 2 (narrow)
Round-8 sonnet re-review (record `3-redteam-plan-r8.md`): **MO-2 RESOLVED + verified** (old-floor migration
mechanism structurally sound). One finding MO-3, which I halted on per the migration stop-condition.

**ORCHESTRATOR OVERRIDE (attributed to the orchestrator / main):** "Orchestrator override of the
migration-class stop-condition for MO-3 — rationale: MO-3 is NOT a mechanism failure. The MO-2 old-floor
mechanism is cold-verified sound. MO-3 is a STALE leftover passage (plan.md:412-419, dated round 2, never
swept) that contradicts the authoritative fix at plan.md:142 — a documentation-consistency cleanup, not the
migration rabbit-hole the stop-condition guards against (the reviewer itself calls it 'narrow/mechanical, not
a design rethink'). Not escalated to owner; no design decision at stake."

Fixes (narrow): (1) correct plan.md:412-419 tolerant-read reference `{"24h"}` → `{"72h"}` old-floor;
(2) C12 sub-case — already-sections-migrated persona hit by a delayed `run_backlog_migration` retry → still
tier 3, not tier 1; (3) **one-time full internal-consistency sweep** of spec+plan+criteria against ALL current
authoritative decisions (terminal tier3 + labels + tier3 cap 20%, old-floor 96h migration, full-chain redirect
+ 5-site rebind + C21, oldest-edge classifier, multi-input terminal fold) — fix every stale leftover so round 9
cannot bounce on another stale sentence.

**REVISED stop-conditions (owner/orchestrator):** both redirect + migration MECHANISMS are closed/verified.
Going forward HALT+report ONLY on a new Major that is a genuine MECHANISM/design/correctness failure in any
class; a pure doc/text-consistency finding AFTER this sweep is fixed inline, not halted. Iteration count:
**8 gate-4/stage-3 rounds.**

**One-time full-consistency sweep (round-8, done):** swept spec+plan+criteria end-to-end against all
authoritative decisions (terminal tier3 + labels + tier3 20% cap, old-floor 96h migration, full-chain redirect
+ 5-site rebind + C21, oldest-edge classifier, multi-input terminal fold). Stale items found + fixed:
(1) MO-3 plan tolerant-read `{"24h"}`→`{"72h"}` old-floor; (2) spec §7 stale `ingest_turn`-follows-`rolled_to`
line → corrected (redirect lives in `get_or_hydrate_session`, P-1); (3) criteria footer "C16→4 handlers"→
"4 operation handlers". All other flagged greps were correct-context (rejected-mechanism descriptions,
fail-demos, prior-art loci, negated "no evict"). No placeholders remain. C12 gained the delayed-backlog-retry
sub-case.

### Gate 4 (plan) — round 9 — 2026-08-13 — MINORS-ONLY → ROUTE TO BUILD (criteria FROZEN)
Round-9 sonnet re-review (record `3-redteam-plan-r9.md`): **minors-only, no Major/Blocker**; all carried
findings reconfirmed resolved against live source; the consistency sweep held under an independent fresh grep.
Per the standing owner go-ahead → **PROCEED TO BUILD.** Two non-gating items folded into the build:
N1 (pin sections-migration BEFORE run_backlog_migration in the startup thread — plan §4), N2 (restore
`covers_until_ts` in the tolerant-reader shape at plan §4). 
**CRITERIA FROZEN at route-to-build:**
  1.5-criteria.md sha256 = `6a12be35e71558237412e8faad5956960c0117bc45ebe3fed42fdee50166a914`
  (spec `724ccd17…`, plan at freeze `72afd8d4…`). Stage 8 verifies 1.5-criteria.md still matches this hash.
Iteration count: **9 gate-4/stage-3 rounds** (converged).

### Stage 5 (build) — base for the stage-6 mechanical diff
Build base commit: cd29bc611a2d00aabe12c56a2dc47a5eeeaa914e (== diagnose base cd29bc61, branch ThinkerOfThoughts/cascade-compaction).
Stage-6 reviewed diff = `git diff <this base>`.

### Gate 7 (code) — 2026-08-13 — worst finding: Major (C-1) → fix-in-place at stage 5, re-review
Stage-6 cold code red-team (sonnet, record `6-redteam-code.md`): all FIVE flagged judgment calls ruled SOUND
(age-boundary strict `<`; updated existing tests; resolved-sid echo; apply_budget `provider=` dead-param
out-of-scope; new `rollover.py` factoring). One **Major C-1 (concurrency)** + minors. Dispositions:
- **C-1 (Major) — FIXED:** `get_or_hydrate_session` now consults the `rolled_to` successor pointer FIRST
  (before the `_SESSIONS` cache and the on-disk buffer), so the mid-rollover window (pointer written, registry
  not yet evicted / old buffer not yet deleted) can no longer return a stale-cached old session that
  `ingest_turn` would resurrect. `rollover.py` also reordered to evict the registry BEFORE deleting the buffer
  (belt-and-braces). **New regression test** `test_c1_mid_rollover_window_redirects_not_resurrect`
  (tests/unit/brain/chat/test_rollover.py) — shown-able-to-fail: it FAILS against the committed pre-fix
  session.py (verified by stashing the fix; resolved to old_sid) and passes with it.
- **L2 (minor) — REVERTED after test disproved the premise:** the reviewer proposed removing the in-rollover
  `cascade_conversation` as "redundant with the daily tick's cascade." FALSE: `_run_compaction_tick` cascades
  BEFORE any extraction (cursor guard → folds nothing), and `perform_rollover` extracts (step 1) THEN folds —
  so the in-rollover cascade (post-extraction) is what actually populates the tiers the seed re-reads. The C9
  real-tick test yielded 0 summary rows without it. Kept the call; documented why it's not redundant. (A case
  where a stage-6 minor rested on an incorrect premise; the conformance test discriminated.)
- **F1 (minor) — FIXED:** plan §1.3 age-boundary prose already reads strict-`<`-consistent with the shipped
  `_bucket_of_age` (Judgment Call 1 ruled the code sound). Nitpick unused `from pathlib import Path`
  (compaction.py) — removed. rollover.py unused import churn resolved.
- **A1 (minor) — ACCEPTED LIMITATION (named):** the pre-existing single-fold double-reject fallback
  (`compaction.py`, `[truncated N earlier messages]`) discards summary text (keeps a count) rather than the
  lossless-leaning join the cascade fallback uses, and the new validator makes it reachable more often.
  ACCEPTED, not changed: (a) it is pre-existing; (b) the raw turns are **archived before the rewrite**
  (lossless-before-lossy), so it is a degraded *summary* on a rare double-reject, NOT data loss; (c) making
  compact_conversation keep full raw text risks head bloat (the exact thing the caps prevent) + its own cap/test
  surface — not cheap. Follow-up: unify the single-fold fallback with the cascade's lossless-leaning join when
  the flat path is next touched.
- **Discretionary test-literalism (C10/C14 interleave/gap thinness, C21 regex-not-AST, the two pipeline.py
  extraction-ran checks) — ACCEPTED as-is:** the reviewer judged all non-vacuous / covered elsewhere; each
  gating criterion is verified by execution on its real path (stage-8 table). Not expanded (no gold-plating).
