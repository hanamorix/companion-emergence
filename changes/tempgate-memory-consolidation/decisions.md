# decisions.md — gate log (append-only)

## Run start — path validation (CFG3, required before gate 4)
Date: 2026-08-11. Branch: `ThinkerOfThoughts/memory-rework-groundwork` @ `cd29bc61`.
Reviewer `redteam_context` brain path is **this worktree's** `brain/` (per the gate brief — the
config's path note assumes main checkout == main and is stale for this matched-baseline run).

- Worktree `brain/` — **OK** (reviewer source of truth).
- All 21 spec touched-files under `brain/` — **OK** (validated by `test -f`).
- New file `brain/engines/consolidation.py` — **absent** (expected; created at build).
- Config extras: `~/Downloads/Phoebe/chat_usage.jsonl`, `.../tool_invocations.log.jsonl`,
  `~/companion-token-trace` — **OK** (present, but **not relevant** to this behavioral change:
  no telemetry replay workload; regression is advisory-only per config + brief. Reviewer told they
  are optional context, not required source.)

Result: all handed paths exist and are readable. Gate-4 path-validation precondition satisfied.

## Author design decisions (delegated to spec by the brief; flagged for red-team fidelity audit)
- **Separate `consolidation_state` column**, not an overload of the existing `state` (which owns the
  active/fading forgetting lifecycle). Rationale: orthogonal axes (a memory can be committed AND
  fading); avoids editing every forgetting call site.
- **`create()` defaults `consolidation_state='candidate'`; deliberate/import/restore writers pass
  `'committed'`.** Fail-safe toward the goal (a missed classification delays, never loses; a missed
  auto-writer under "default committed" would defeat the change). The brief explicitly left the
  deliberate-vs-gated boundary "to the spec."
- **Column SQL default `'committed'`** so pre-existing rows + migration are committed (no recall
  regression). New candidate status is applied at the `create()` call, not the column default.
- Magnitude knobs (salience floor, cluster threshold, merge sensitivity) parameterized with a
  conservative default, **pending L-B confirmation** — deliberately NOT blocked on per the brief.

(Gate entries appended below as the loop progresses.)

## Gate 3 (round 1) — plan red-team
Worst severity: **MAJOR** → route **replan** (stage 2). Reviewer: general-purpose / opus.
Two MAJOR verified against source by the author before acting:
- **A-1** `run_tick` is multi-threaded (background `supervisor.py:1085` + close-worker
  `server.py:651` + CLI, own connections) — my "single-threaded tick" accessor-table claim was
  FALSE; overlapping runs could double-fold a committed merge target.
- **F-1** reflex (`reflex.py:451`, `emotions={}`) and heartbeat (`heartbeat.py:931`) rows are
  content-bearing at `importance==0.0`; my `<=0.0` salience `hard_delete` would destroy them each
  tick.
Plus A-2 (grief-graveyard resurfacing drops as "lost"), CH8-1 (gate pruning interior-continuity
source rows), F-2 (`ensure_edge` no-ops over a prior weak edge), CH8-5 (recall-count churn).
Crux design (separate column, choke-point, recall seam, cache-prefix isolation) ruled CLEAN+earned.

**Replan adopted (round 2 artifacts):**
1. Gate is **non-destructive by default** — promote-or-leave; no `hard_delete`, no grief-graveyard
   entry. Recall filter alone cuts the flood fuel. Physical candidate-GC DEFERRED; `monologue_trace`
   never GC-eligible. (Resolves F-1, A-2, CH8-1.)
2. **Salience filtering OFF by default** (`SALIENCE_FLOOR=None`) — no numeric floor safe pre-L-B.
3. **Non-blocking persona gate file-lock** (reuse `graveyard.py` `file_lock`); contended run skips.
   (Resolves A-1.) Accessor table corrected.
4. Merge archives pre-image to a **plain append-only archive file**, not the grief graveyard.
5. **`hebbian.set_edge_weight`** (UPSERT to MAX(existing,5.0)) replaces bare `ensure_edge`.
   (Resolves F-2.)
6. Gate internal reads use **`search_text(bump=False)`** (resolves CH8-5).
7. New criteria: **C17** (overlapping-run lock), **C18** (no hard-delete/no grief-graveyard),
   **C19** (interior-continuity source rows not pruned); C5/C7/C8/C10 updated.
8. Within-session affected recall consumers named in spec; flagged for sign-off.

## Gate 3 (round 2) — plan red-team
Worst severity: **MAJOR** → route **replan**. Reviewer: general-purpose / opus. Record:
`3-redteam-plan-v2.md`. 4/6 round-1 findings verified RESOLVED (F-1, A-2, F-2, CH8-1, CH8-5).
Two MAJORs:
- **M1** the `file_lock` helper (`brain/utils/file_lock.py`) is BLOCKING-ONLY — verified by reading
  it — so "reuse as a non-blocking skip lock" is not deliverable and C17 was unsatisfiable.
- **A** non-destructive default → duplicates/merged-sources/low-salience candidates stay in the
  pool forever, re-classified every tick (per-tick work unbounded; "bounded" claim false) and a
  folded source re-mergeable → surgical-drift across ticks. No criterion observed multi-tick
  stability.
Plus advisories: non-exhaustive "reviewed-not-filtered" search_text list; classifier return
contract unspecified; gate-merge vs concurrent fade un-locked but non-corrupting.

**Replan adopted (round 3 artifacts):**
1. **Third `consolidation_state` value `held`** for gate-decided-not-promote (dup / low-salience /
   merged-source). `list_candidates` returns ONLY `candidate`, so per-tick work is bounded to new
   arrivals and a folded source is never re-merged. Recall shows `committed` only (held excluded).
   Held rows decay via normal forgetting → accumulation bounded. (Resolves A.)
2. **Non-blocking mode added to `file_lock`** (`blocking=False` → POSIX `LOCK_NB` / Windows
   `LK_NBLCK`, yields acquired-bool). Gate skips if not acquired. (Resolves M1; C17 now satisfiable.)
3. New criterion **C20** (multi-tick stability: decided rows not re-processed/re-merged);
   C2/C5/C7/C19 updated for the `held` value; classifier return contract pinned; search_text caller
   list made exhaustive + research.py:518/522 flagged as an out-of-scope secondary fuel path.

**Iteration note:** gate 3 has now bounced twice (rounds 1-2), both MAJOR. Findings differ per round
(assumption → mechanism → accumulation) and each round resolved the prior — this reads as
convergence, not a stuck tie. One confirmation round (3) will run; if it bounces MAJOR again on the
concurrency-guard class, STOP and escalate the iteration-cap situation to the owner rather than a
round 4.

## Gate 3 (round 3) — plan red-team → ITERATION CAP → ESCALATE
Worst severity: **MAJOR** (3rd consecutive at gate 3). Reviewer: general-purpose / opus. Record:
`3-redteam-plan-v3.md`. M1 RESOLVED (non-blocking lock sound; C17 satisfiable). `held` introduces
no new non-recall-consumer leak (verified: no non-recall consumer filters `consolidation_state`).
NEW MAJOR: a `held` deduped/merged row still rides the forgetting loss pass
(`forgetting/__init__.py:135`, no state filter; `is_exempt` ignores `protected`) → guaranteed
decay → graveyard + grief = round-1 A-2 reopened via forgetting. Two minors fixed this pass
(forgetting-runs-in-supervisor-thread accessor correction; merge-update try/except KeyError guard).

**Iteration cap reached** on the finding class **"terminal fate of a gate-rejected candidate"**:
- bounce 1 = round-1 A-2 (grief-on-drop via graveyard),
- bounce 2 = round-2 A (accumulation / re-merge),
- bounce 3 = round-3 (grief-on-decay via forgetting).
Each round resolved the prior surface and exposed the next — the axis is genuinely under-specified
in the brief ("drop … always safe" does not resolve the grief/forgetting/referential-integrity
interactions) and carries product/values weight (companion grief behavior). Per the cap + the
delegation stop-for-human rule, **the runner STOPS and escalates to the owner** rather than patch a
4th time. Options + author recommendation captured in 1-spec "Open owner-decision gates #1".

**Route:** HELD at gate 3 pending owner ruling on the terminal-fate axis. No build. No 4th red-team
round until the axis is ruled. On the ruling: write a RAT1 ratification record into 1-spec, add the
covering criterion (forgetting pass over reject rows), then resume the loop (re-red-team the
resolved spec, then gate 4 → owner sign-off → build).

## OWNER RULING (Roy, relayed via coordinator 2026-08-11) — ARCHITECTURE CORRECTION → RESPEC
Roy dissolved the terminal-fate axis at the architecture level: the pending/candidate queue is a
**separate store OUTSIDE memories.db**; a candidate is NOT a stored memory. REJECT = discard from the
queue (no graveyard/grief/hebbian — those are main-DB concepts); PROMOTE = store.create into main DB;
MERGE = archive lossless pre-image of the existing main-DB memory + surgical edit + discard candidate.
Recall reads main DB (genuine+promoted only) → the committed_only/consolidation_state recall-filter
apparatus is REMOVED. Forgetting walks only main DB → no exemption logic. Keep what survived red-team
(gate-first ordering, non-blocking gate lock for the 3-thread tick, weight-5 edges for
corrections/continuations, merge archive-before-edit + KeyError guard). Within-session recall behavior
RATIFIED as intended (do not re-ask). Ruling recorded as RAT1/RAT2 ratification records in 1-spec.md.

**Respec done (v2 artifacts):** 1-spec/1.5-criteria/2-plan rewritten for the separate-queue model. New
`PendingQueue` (JSONL + file_lock, mirroring soul_candidates.jsonl); automatic firehose writers
ENQUEUE instead of store.create; deliberate/import/restore writers unchanged; interior-continuity reads
recent traces from the QUEUE (else it starves — the one interaction Roy's terse ruling didn't spell out,
resolved faithfully). No recall/forgetting/cache-prefix edits at all (C13 asserts this).

**NEW SURFACE this architecture introduces — consumer migration (flagged for stage-4 sign-off, NOT
silently resolved):** pulling candidates out of memories.db changes what EVERY main-DB list_by_type
reader of a firehose type sees (only promoted now), not just recall. Material items: (1) bridge feed
(feed.py) shows consolidated-only dream/research/notes — a user-facing product change; (2)
initiate/memory.py:105 outbound dedup could double-send if outbound candidates sit un-promoted in the
queue — a potential bug. Also surfaced: scope option (full firehose = pinned, vs monologue-family-only
= smaller blast radius). These are the stage-4 owner-sign-off gates (C17 finalizes on the ruling).

**Route:** respec complete → one cold review of the settled spec (per owner instruction) → return to
main at stage-4 → owner sign-off (relaying the consumer-migration + scope items) before any build.

## Cold review of settled v2 (the one Roy scoped) → architecture SOUND, round-4 fixes applied
Reviewer: general-purpose / opus. Record: 3-redteam-plan-v4.md. Verdict: the separate-queue
architecture is fundamentally sound and needs NO respec; reject-fate axis GENUINELY DISSOLVED;
recall/forgetting/cache untouched by construction; queue + merge-target concurrency guards correct;
ratification records honest/hedged. One MAJOR (plan-level) + minors, all fixed on this pass:
- MAJOR reflex/journal straddle → **type-based routing** (`route_write` keys on
  `memory_type ∈ GATED_TYPES`, not call site); variable-typed sites self-classify; `journal_entry`
  excluded so prompt.py:1079 is unaffected. GATED_TYPES membership = the scope knob for sign-off.
- initiate double-send → resolved by default (initiate_outbound excluded from GATED_TYPES); C17 split
  into C17a (mechanism, gating, testable now) + C17b (feed product decision, stage-4).
- merge-vs-fade window → skip-if-fading/missing re-read guard; drain-crash durability named-accepted;
  recall.py keep-sharp no-op accepted; C5 gains a to_dict round-trip assertion; C10 wording fixed.

## Gate 4 (spec/criteria gate) — STOP for owner sign-off
Spec + criteria + plan are cold-review-clean at the architecture level with round-4 fixes applied.
Per the run's mandate, HOLD at the spec/criteria gate for owner sign-off before build. Owner-decision
items to relay: (1) GATED_TYPES scope — full firehose (default: monologue family + dream + research +
heartbeat + gated reflex/ingest types) vs monologue-family-only; (2) bridge-feed behavior
(dream/research consolidated-only, C17b); (3) confirm interior-continuity-cleared-at-drain is
acceptable (Open #3). No build until sign-off. No push/PR. LOCAL only.

## OWNER SIGN-OFF (Roy, relayed 2026-08-11) — PROCEED TO BUILD, with directives (R3)
1. Scope BROAD confirmed (type-based routing kept). 2. Feeds consolidated-only accepted (C17b
resolved, no code change). 3. Interior-continuity KEEP AS-IS — do NOT change _AMBIENT_LIMIT (5) or
semantics; only the read-source move; ≤2-thought rework deferred to Phase 4. 4. SALIENCE GUARD (real,
Roy-flagged): verified dreams derive importance=sum(emotions)/10 (≈0 flat), monologue_trace pinned
0.3, only monologue episodes carry 0..10 (extractor w.salience*10). Design call: salience-drop scoped
to SALIENCE_ELIGIBLE_TYPES = {monologue, monologue_emotion, monologue_soul_candidate}; dreams/research/
heartbeat/reflex/initiate/trace EXEMPT (dedup only). New gating criterion C18 proves legitimate
dream/research/trace survive. 5. Run one extra cold spec pass on round-4 fixes + salience change
before build. Recorded as ratification record R3 in 1-spec.md.

Applied to artifacts: spec Pass-1 salience scoping + deferred note + R3 + sign-off resolutions;
plan Step-4 salience scoping; criteria C18 added, C17b resolved→advisory. Next: cold spec pass, then
build (instrument→build→cold-review→CI→local commits), back to main at post-build/done gate.

## Stage 5 — BUILD (owner-approved; local only)
Production implemented per plan: file_lock blocking=False; hebbian.set_edge_weight; store.db_path/
persona_dir + search_text bump=False; NEW brain/memory/pending.py (PendingQueue + route_write +
GATE_BYPASS_TYPES={journal_entry,initiate_outbound} + SALIENCE_ELIGIBLE_TYPES={monologue,
monologue_emotion,monologue_soul_candidate}); NEW brain/engines/consolidation.py (drain + Pass1/2 +
promote/merge/reject + non-blocking gate lock); heartbeat run_tick wiring (gate first); 8 automatic
writers → route_write; interior-continuity (ambient.py, monologue/recall.py) read the queue.
ingest auto-hebbian guarded to bypass-only (gated candidate has no DB row → would dangle).

New gate tests: tests/test_consolidation_gate.py — 19 pass, covering C1-C14, C18.

**Existing-test migration (44 fail on old direct-to-DB contract → new queue contract).** Delegated to
a cold builder with strict anti-masking rules; result 4233 passed / 0 failed / 6 xfailed, ruff clean.
Category-B xfails (behavior genuinely removed, deferred to Phase 4, assertions left intact):
monologue_trace no longer fades/is-lost in DB (2), recall keep-sharp bump is a no-op (1), ambient
faded-summary render N/A (1). These are legitimate (traces are ephemeral queue items now).

**REAL REGRESSION found by the migration + FIXED (production).**
`HeartbeatEngine._write_daemon_state_for_dream` re-fetched the just-fired dream via store.get(id) →
None (dream now a gated candidate) → wrote nothing. Reflex/research daemon writers likewise degraded
to synthetic fallbacks. FIX: all three now read the just-fired memory from the pending queue
(store.persona_dir) with a store fallback. The migration's strict-xfail on that dream test is removed;
all 77 heartbeat tests pass. Heartbeat daemon AGGREGATE (emotional state across memories) intentionally
reads consolidated memories only — consistent with Roy's consolidated-only feeds ruling.

Migration also surfaced (flagged, not my regression): an unrelated latent timezone flake in
initiate/test_review (made deterministic); and :memory:-store tests leaking a pending_candidates.jsonl
to CWD (gitignored litter; production never uses :memory:).

NEXT: full-suite re-run (confirm green after the production daemon fix) → stage-6 cold review of the
BUILD (independent) → commit → back to main at the done gate.
