# 3-redteam-plan.md — verbatim stage-3 record

## Reviewer
- **Agent type / model:** general-purpose (Explore-class), **opus** — chosen deliberately: adversarial
  judgment over a large architectural design plus code-grounded factual checks; the loop's highest-stakes gate;
  the project config flags reviewer independence matters extra here.
- Spawned cold (no shared context with the author).

## Closed context set given to the reviewer
- Stage artifacts: `1-spec.md`, `1.5-criteria.md`, `2-plan.md` (this change folder).
- `redteam_context` (priority-ordered, from `guarded-change.companion.md`): `brain/` clone (source of truth) +
  the stale `~/Downloads/Phoebe/*.jsonl` snapshots + `~/companion-token-trace/` (priors, not authority).
- Spec touched-files (the closed set): `brain/chat/compaction.py`, `brain/chat/compaction_migration.py`,
  `brain/ingest/buffer.py`, `brain/bridge/server.py`, `brain/bridge/supervisor.py`, `brain/ingest/pipeline.py`,
  `brain/chat/engine.py`, `brain/chat/budget.py`, `brain/monologue/ambient.py`.
- Design authority for the fidelity lens: `~/.claude/plans/memory-dream-rework-p1-cascade-brief.md`,
  `/home/zero/Documents/New_mem_system.md` Part 1.
- Carried-forward findings: none (first review of this change).

## Charter given
The full red-team charter core (five lenses + evidence discipline + provenance + position lens + concurrency
lens) verbatim, plus the stage-3 additions (CH8 coverage challenge, CH9/CH10 label audit). Reproduced verbatim
in the reviewer prompt (see the spawn record). No "OWNER MUST RATIFY" finding is carried, so CH11/CH12 do not
fire — but the reviewer was tasked to flag any of the three "resolved open points" that is actually an
owner-values call mis-cast as an engineering default.

## Author-side artifact + source hashes (recorded at spawn, base `cd29bc61`)
Artifacts:
```
2c296c496faee52b665473dc84be81bac3fde5bf66d9ec4dc6ff6dce280da4b3  1-spec.md
8cb87538ac59014bf52e48deeff03e9983b7b98ddaa32fd7a8d1d60efbaf0e34  1.5-criteria.md
63853746645304d188279a4b2e2d5bef7e920416ec6ec8286b193a1f464db5da  2-plan.md
```
Touched files:
```
cc9b6e22ac3cf05aa1109e84abb5d8217619153d85cf172ae1b84644b1aad7fb  brain/chat/compaction.py
7d32a50b29f47b34645bd48c04b6cae3bb62e2dd15e031712b2b752fd59e1d77  brain/chat/compaction_migration.py
cefb079963884fbafea3a0d8125c74bdc3a9e889894731329f482f49f93da56b  brain/ingest/buffer.py
f0b0b715746bc9f8e27964ec7c24301a14b6b80dab4d38db8f8b533d1df62d7e  brain/bridge/server.py
ca6eeba8070959cf502e76177a3a635832ecc0377e12e43773b3cf9629c116b2  brain/bridge/supervisor.py
cfe8b63b3d642dabe998f52b52a087ccc4c0acbd8bab5b32bba177f0b309d331  brain/ingest/pipeline.py
d359be520046a66c502ca5f0b56a0c61e8f4a13fbef93868e6fe127bed0d1260  brain/chat/engine.py
b71ed34d8f0d50108739fad682a3698989cfb5bd1278f0e4a3c7a011d873c7ca  brain/chat/budget.py
74dd5ba03614872c430fd2f3d2e40f23d4f70349a69816309ea0a4e7b8ee054e  brain/monologue/ambient.py
```

---

## Reviewer verbatim output
Returned as text via the orchestrator (not written to disk by the reviewer); pasted verbatim below.
Reviewer-reported hashes (truncated in the return): 1-spec `2c296c49…da4b3`, 1.5-criteria `8cb87538…a0e34`,
2-plan `63853746…4db5da` — match the author-side hashes above. Brief + New_mem_system.md also read.

```
# Stage-3 Plan Red-Team — Cascade Compaction

## 1. Provenance header
Reviewer: cold independent stage-3 reviewer, general-purpose agent, model claude-opus-4-8.
sha256: 1-spec.md 2c296c49...da4b3; 1.5-criteria.md 8cb87538...a0e34; 2-plan.md 63853746...4db5da; brief c23de64f...1329d; New_mem_system.md ad4c76ff...ce64f9.
Source consulted: brain/chat/compaction.py (full), budget.py (full), engine.py (:44-56,:228-245,:360-409), ingest/buffer.py (:255-403), bridge/server.py (:905-944,:1203-1253), bridge/supervisor.py (:495-521,:635-665,:1619-1697), ingest/pipeline.py (:529-626), monologue/ambient.py (:22-45), chat/compaction_migration.py (:160-209).

## 2. Per-lens findings
Lens 1 Factual — earned clean. Verified: .strip()-only store compaction.py:338; archive-before-rewrite byte-count abort :382-394; race-safe re-read-by-identity rewrite :406-417; _split_buffer first-summary-wins :192-209; cursor guard None->no-op/clamp :263-273; min_keep_tail protected window :276; old-summary appended to archive on fold :380-381. budget.py _COMPACTION_SUMMARY_PREFIX :31, preserve-head re-parse :103-109; default max_tokens=190_000 budget.py:45 BUT engine.respond overrides to 80_000 engine.py:240 — plan §0-Q8 correct. engine.py:405-408 inserts summary at history index 0 -> assembled index 1 -> budget re-parses messages[1]; no contradiction. append_archive returns bytes, append-only, whole-file read_archive buffer.py:290-312; per-session O_EXCL lock :341-392; ambient.py:34 is a MemoryStore.list_by_type read (C13 preserved by construction). Nitpick: spec §6 cites summary row :362-374; actually :364-374.

Lens 2 Logical —
L1 (Major): "shift per tick" != age-gated promotion. compaction cadence advances exactly one interval unconditionally (supervisor.py:635-662, persisted_cadence.advance); after multi-day sleep is_due fires ONCE not once-per-missed-day; compact_conversation folds ALL turns ts<=now-24h in one pass (compaction.py:271-281). So 3 days of raw turns fold into "24h" section in one tick while the shift moves layers down one position -> labels stop matching age. This is the designed post-restart/sleep behavior (supervisor.py:642-647, #21).
L2 (Major): buffer cleanup removed from finalize (pipeline.py:557-559,:588-590) but not reassigned; only poison-move (:598-602) removes a buffer after. Plan §2.1/2.2 archive+new-sid but never delete the rolled-over old buffer, yet §2.3 says "rollover owns that lifecycle." Result: every rolled-over buffer persists in active_conversations/; list_active_sessions re-iterated every tick by _run_compaction_tick (supervisor.py:1636), /sessions/active (server.py:1227), finalize (pipeline.py:554) -> cost grows with all sessions ever created.
L3 (Minor): #77 "keep prior section" fallback defined for single-fold; under cascade shift the 48h->72h re-compact's "prior section" is the old 72h (being discarded). Double-reject keeps old 72h; 48h material neither promoted nor retained (48h overwritten by 24h same pass) -> dropped from tiers (still in archive). Specify fallback for the cascade case.

Lens 3 Missed opportunity —
M1 (Minor): weekly-cap executes only at /sessions/active (on re-attach); supervisor quiet-gap swap deferred. But trigger B targets the continuously-used conversation that never hits the 24h gap; a client staying attached never picks up the marker -> weekly cap never fires for its target population. Build the supervisor quiet-gap swap now under the same lock.

Lens 4 Assumptions/risks/concurrency —
A1 (Major): 1c-A runs multi-second sync Haiku full-fold inside a def (threadpool) GET handler (/sessions/active server.py:1203-1253). Risks not addressed: (a) blocks a threadpool worker for the full fold on every stale resume; (b) which stale session is rolled over when several exist is unspecified (endpoint has no client-supplied prior sid); (c) after L2, stale buffers accumulate, compounding (b).
A2 (Minor): weekly rollover assumes client re-attaches; load-bearing given trigger B's continuous-client purpose.
Concurrency: §5 accessor table is thorough; correctly IDs ingest_turn appends without the compaction lock (compaction.py:404-405), handled by re-read-by-identity rewrite. Residual: (i) segment-roll fsyncs the file but a new segment file's directory entry needs a directory fsync for crash durability — not mentioned; (ii) §5 asserts all archive writers hold the per-session lock, so C11's "append during a roll" is a lock-precluded interleaving, not the real risk (crash mid-roll).

Lens 5 Fidelity — terms pinned to New_mem_system.md Part 1 + brief:
"three age-labelled sections, one row" -> faithful (one summary row + compaction.sections + deterministic render §1.1-1.2). "60/40/20 ratios" -> matches New_mem_system.md:47-49 exactly. "#77 fold validation" -> faithful (§1.4 predicate replacing .strip() at :338), modulo L3. "#82 temporal markers" -> faithful (§1.2 coarse span headers from covers_from/until_ts). "dream-ordering slot reserved" -> faithful genuine no-op.
F1 (Major, escalation-worthy per reviewer): the tick-count shift is a proxy for age that holds only if the tick fires reliably every 24h with steady activity; under skipped/quiet ticks the "24h layer" ceases to hold recent material, breaking the owner's key insight (New_mem_system.md:8,51). Unflagged mechanism substitution (not one of the 3 authorized open points) -> untrusted until owner confirms proxy or age-gating restored.
F2 (Minor, confirm-with-owner per reviewer): the weekly fire-point resolution (mark-on-tick, execute-on-next-attach, defer quiet-gap swap) changes WHETHER trigger B fires for its target population -> behavioral-scope decision, not a value-neutral knob. [Orchestrator note: brief line 68 "checked on the daily cadence" already resolves this — conform, don't escalate.]

## 3. Coverage challenge (CH8) — behaviors no criterion observes:
1. Multi-tick/skipped-tick cascade correctness (High). C2 exercises ONE pass; C5 only asserts header span==covered ts; nothing asserts the "24h" section holds <=24h material. F1/L1 failure invisible. Add a multi-tick fixture for layer<->age correspondence.
2. Old-buffer cleanup / active-set boundedness after rollover (High). C10 checks seed survives finalize; nothing checks the OLD buffer is removed / active set stays bounded (L2 leak).
3. 1c-A stale-session selection with multiple stale buffers (Medium). C8 uses a single stale fixture.
4. New-session extraction-cursor continuity after rollover (Medium). Seed is a summary row (never extracted); new session starts with no cursor -> first compaction no-ops until extraction runs. Not asserted.
5. Dream-ordering-slot as a true no-op (Low). Genuine no-op; nothing enforces a later change won't fold-before-dream. Documentation-only.
6. 72h material falling off bottom (Low). Verified SAFE — cursor guard :281 only folds extracted turns; dropped material in archive + long-term memory. No data-loss finding.

## 4. Label audit — gating criteria governed paths: C1 representation writer (governs); C2 oldest-first wiring but ONE pass only; C3 cap enforcement (governs); C4 _validate_fold_output single-fold (governs; not cascade fallback L3); C5 temporal render label content (not label-vs-age); C6 position-lens byte-equality + budget re-parse (governs, well-constructed); C7 invariants (governs); C8 1c-A single-stale (multi-stale uncovered); C9 1c-B attach swap (never-re-attach unobservable); C10 finalize<->rollover concurrency oracle (governs; not old-buffer reaping); C11 segmentation/reader/provenance (governs; "append during roll" tests lock-precluded interleaving — re-point at crash); C12 migration+idempotence (governs); C13 ambient guard (governs). Advisory labels A1 regression / A2 CI-green both justified (no live workload; C6 covers cache-stability mechanism as gating).

## 5. Bottom line
Worst-severity: F1 (tick-shift proxy breaks "24h layer IS recent material" on skipped/quiet ticks, unobserved — CH8-1) ties with L2 (buffer-deletion removed from finalize, never reassigned -> leak inflating every tick — CH8-2). A1 (sync seconds-long fold in GET handler + unspecified stale selection) close behind.
Routing: Return to author (stage-2) before build. (1) escalate F1 or restore age-gating + add multi-tick fixture; (2) assign old-buffer deletion to rollover + boundedness criterion; (3) specify 1c-A stale selection + address worker-blocking sync fold; (4) confirm F2 weekly fire-point; (5) fix L3 cascade fallback + re-point C11 at a crash scenario. Core (C1,C3,C4,C6,C7,C12,C13) sound and faithful.
```

## Author disposition (orchestrator-routed; see `decisions.md` gate-4 entry)
Route: **Major → return to stage 2.** Findings addressed in the revised {spec, criteria, plan} then re-red-teamed:
- **F1/L1 (Major):** restore true **age-gated** promotion (partition by ACTUAL age, robust to skipped/quiet
  ticks); NOT escalated (brief §Decisions-2 line 45 mandates age-gating). New criterion **C14** (layer↔age).
- **L2 (Major):** rollover path now **owns old-buffer deletion** (extract → archive → delete → reseed); new
  criterion **C15** (old buffer reaped, active set bounded).
- **A1 (Major):** SYNC kept (owner-ratified, brief §1c-A). Sub-points resolved: stale-session **selection rule**
  (most-recently-active) + **sync-work bound** (§2.1). No de-sync.
- **F2/M1 (Minor):** weekly trigger B driven from the **daily supervisor tick** at a quiet-moment boundary +
  a **successor pointer** so a continuous client's stale-sid append redirects (brief §1c-B line 68 "checked on
  the daily cadence"). NOT escalated. New criterion **C16** (post-rollover redirect).
- **L3 (Minor):** #77 fallback specified for the **cascade** case (never drop source material on double-reject).
- **C11:** re-pointed at a **crash / partial-write** scenario + directory fsync (append-during-roll is
  lock-precluded).
- **Nitpick:** spec §6 summary-row cite corrected to `:364-374`.

---

# Stage-3 RE-RUN (round 2) — revised {spec, criteria, plan}

## Reviewer
- **Agent type / model:** general-purpose, **sonnet** (per owner usage directive 2026-08-13: all cold
  reviewers on sonnet; cold-independence + lens discipline unchanged).
- Carried-forward findings: F1/L1, L2, A1, F2/M1, L3, C11, nitpick (from round 1) — reviewer tasked to confirm
  each resolved + hunt new issues.

## Revised artifact hashes (author-side, base cd29bc61)
```
4df4d6b56b673a034b6583c6efe644f2c86ffcb8236ebfc0bb81740c0cef7072  1-spec.md
abec34f4ac17dfce960ab041640c97fe875db9f53c4f27536f838a1cdf96c7df  1.5-criteria.md
60ec2e7953ba5fb8645c9d7ae9144bec754772ea6389236572869cb66ea249f6  2-plan.md
```

## Reviewer verbatim output (round 2)
_(pending re-review completion)_

Returned as text via the orchestrator; pasted verbatim below. Reviewer-reported full-file hashes match the
author-side revised hashes above (1-spec `4df4d6b5…`, 1.5-criteria `abec34f4…`, 2-plan `60ec2e79…`).

```
# Stage-3 Plan Red-Team RE-RUN — Cascade Compaction (Cold Review)

## 1. Provenance
Agent: general-purpose subagent, model claude-sonnet-5, cold independent reviewer.
sha256 (full-file): 1-spec.md 4df4d6b5...f7072; 1.5-criteria.md abec34f4...6c7df; 2-plan.md 60ec2e79...49f6; compaction.py cc9b6e22...ad7fb; buffer.py cefb0799...da56b; server.py f0b0b715...62d7e; supervisor.py ca6eeba8...116b2; pipeline.py cfe8b63b...09d331; engine.py d359be52...0d1260; budget.py b71ed34d...73c7ca; compaction_migration.py 7d32a50b...e1d77; ambient.py 74dd5ba0...054e. Also read: cascade-brief (1-100), New_mem_system.md Part 1, brain/chat/session.py (full), brain/bridge/persisted_cadence.py (full), brain/health/attempt_heal.py:235-268.

## 2. Carried-findings resolution
F1/L1 (Major) age-gating: RESOLVED — spec §2b + plan §1.3 age-partition raw turns (G24/G48/G72 by now−ts) + age-classify/merge-down by newest covered ts each pass; persisted_cadence.advance = next_at=now+interval (verified: fires once after multi-day sleep, not per-missed-day); C14 + fail-demo target it.
L2 (Major) finalize delete → rollover: RESOLVED at design level — plan §2.3 finalize becomes extraction-only; both rollover paths call delete_session_buffer after seed (matches pipeline.py:557-559,587-590); C15 gates active-set boundedness. (See P-1 residual on empty/corrupt sub-path.)
A1 (Major) stale selection + sync bound: RESOLVED — plan §2.1 most-recently-active selection + deterministic tie-break, bounds sync to one _SECTION_24H_CHAR_CAP-truncated Haiku call; sessions_active_endpoint confirmed plain def at server.py:1203 (threadpool).
F2/M1 (Minor) weekly from daily tick: PARTIAL → see P-1. Trigger source correctly moved to daily supervisor tick (§0, vs supervisor.py:639-662), but the rolled_to.json redirect it relies on doesn't reach its real locus.
L3 (Minor) #77 cascade fallback: RESOLVED — plan §1.3/1.4 lossless-leaning safe join + sentence-boundary truncation + soft-fail log; C4(b) tests drop-on-double-reject.
C11 crash+dir-fsync: RESOLVED with citation gap — §3 re-points to crash sim + dir fsync; codebase already has the pattern at attempt_heal.py:250-266 (uncited).
Nitpick :364-374: RESOLVED, confirmed exact (summary_row dict).

## 3. Per-lens
Lens 1 Factual — P-1 (BLOCKER): plan asserts (§0:38-40, §2.3, §5 table) ingest_turn session-resolution + /sessions/active both consult rolled_to.json → "correct-by-construction" for a continuous client holding old sid. FALSE at the locus that matters. Traced: POST /chat (server.py:2360-2366) → get_or_hydrate_session(...) → if None raise 404 (server.py:2365-2367); ws path same (server.py:2410,2424-2428). get_or_hydrate_session (session.py:129-205) checks only _SESSIONS + read_session; NO rolled_to.json anywhere. ingest_turn (buffer.py:74-104) only reached from engine.respond (engine.py:433-434), AFTER get_or_hydrate_session already succeeded. Failure: after weekly swap, rollover deletes old buffer + evicts old sid; continuous client's next /chat still carries old_sid → get_or_hydrate_session finds nothing (evicted + deleted) → 404 before ingest_turn/rolled_to.json ever consulted → turn lost, opposite of spec §2e-B/plan §0 promise. Trigger A doesn't share this (idle client re-queries /sessions/active). Touched-file set (spec §7) excludes session.py + /chat//stream handlers → closed scope doesn't cover the file needing change. Criteria notes flagged this exact question open ("whether ingest_turn can actually consult rolled_to.json at its real locus") but the revision restates the claim more confidently instead of resolving it.
Clean sub-findings: all spec §6 line cites accurate to 0-2 lines (:364-374 summary row, :338 .strip(), buffer.py :290/310/265/284, server.py :1224/1248/1253, list_active_sessions :1636/:554/:1227, migration :913-935, budget.py :28-31/42-91/94-109, engine.py :238-245/:405-408/:48-55, compaction.py :107/85/147/149/54 — exact). persisted_cadence.advance = next_at=now+interval (supports F1/L1). Fidelity: New_mem_system.md 60/40/20 == plan §1.3 _FRACTION_24H/48H/72H; "layer IS timestamp" matches spec §2a.
Lens 2 Logical — Minor: unstated whether 3-tier cascade write is one atomic rewrite_session_atomic or three sequential archive+rewrite cycles; "one lock" stated, "one write" not; sequential → crash between new_72/new_48 leaves partial row no criterion catches (C7 = idempotency-on-stable-input only; C11 crash = archive segmentation only). State all three computed from pre-pass snapshot + installed in one rewrite. Minor: unstated whether engine._buffer_turns_to_messages renders against compaction.sections at read time or reads a pre-rendered text field; equivalent if pure/deterministic, but inconsistent with §7 listing engine.py touched "for the 3-section render."
Lens 3 Missed-opp — Minor: C11 dir-fsync doesn't cite existing attempt_heal.py:250-266 posix-guarded pattern; bare os.open(O_DIRECTORY) raises on Windows CI (repo supports Windows: buffer.py _unlink_with_retry/_pid_alive). Minor: no section addresses run_backlog_migration (unchanged) sequencing vs new sections; _drain_session still calls legacy compact_conversation whose summary_row (:364-374) has no sections key; if backlog undrained across restarts (transient locked/archive_failed withhold marker, migration.py:44-47) while daily cascade already produced a sectioned row, a later legacy drain flattens it back to single-layer, losing sections meta; likely self-heals next cascade tick via tolerant reader (§1.1) so not data-loss, but an uncited migration interaction worth a one-line note.
Lens 4 Assumptions — Minor: §2.1 tie-break (highest ts then lexicographically-largest sid) assumes ties matter; plausible at second-resolution ISO ts but unexercised beyond "multi-stale fixture." Minor: finalize extraction pass + concurrent rollover's own extraction (§2.1a) assumed re-entrant since neither takes the compaction lock for extraction (only delete/archive/rewrite lock-guarded); likely fine (cursor-gated idempotent) + C10 aimed at it, but idempotency argument never stated explicitly.
Lens 5 Fidelity — "layer IS timestamp" faithful; "true age-gated not tick-shift" faithful + consistent with persisted_cadence.advance; "correct-by-construction (successor pointer)" NOT faithful — plan's own claim (not brief language; brief 65-68 says nothing about a successor pointer), demonstrably false per P-1, a proxy for the property it claims. "SYNC (1c-A)" owner-ratified (brief 62-64), correctly blocking. "dream-ordering slot" faithful trivially.
Position lens (fires): No issue — engine.py:405-408 head f-string unchanged; budget.py:28-31 prefix unchanged; §1.2 commits render to "no nonces, no now()"; C6 tests by execution (render twice byte-compare + feed real re-parser), satisfies charter H3.
Concurrency lens (fires): accessor table (§5) comprehensive; each guard claim checked — compaction lock covers cascade + both rollover writes (buffer.py:341-392); ingest_turn genuinely lockless (buffer.py:74-104); finalize takes no compaction lock (pipeline.py:529-626), consistent with "R (extract only)"; archive writers all inside lock. BLOCKER (also concurrency): P-1 — the one accessor the table claims "covered" (ingest_turn consulting rolled_to.json) is not a lock gap but a LOCUS gap: the redirect never executes for the accessor that matters (continuous client's /chat), which fails upstream of ingest_turn.

## 4. Coverage-challenge (CH8)
1. (Confirms P-1, BLOCKER) post-rollover /chat//stream continuation for a continuous client observed by no criterion; C16's oracle bypasses the real path.
2. (Minor) crash mid-cascade-write (partial across 72/48/24 tiers) unobserved; C7 = idempotency + archive-verify booleans, not a crash between tier writes.
3. (Minor) run_backlog_migration × new sections interaction across restarts unobserved (C12 tests sections migration in isolation).
4. (Minor, self-identified, open) new-session extraction-cursor continuity after rollover — notes say "handled by seed's cursor-set" but no criterion (C8/C9/C12) asserts the seeded row's cursor prevents re-extraction; add a direct assertion (C13 precedent).
5. (Minor) 24h-only apply_budget backstop vs full age-gated daily cascade diverging live (backstop fires mid-day, not driven with min_keep_tail=preserve_tail_msgs=40 matching daily default) named in notes but untested; C2/C7 fixtures scoped to daily path only.
6. (Nitpick) empty/corrupt-buffer cleanup path (pipeline.py:556-560 `if not turns:` branch, distinct from :587-590 success delete) not addressed by "finalize becomes extraction-only"; ambiguous whether an implementer removes it too.

## 5. Label-audit
C1-C7,C12-C14 direct compaction/cascade/fold/migration calls — real. C6 position render/re-parse — real (execution, H3). C8,C9 /sessions/active + daily-tick weekly swap — PARTIAL/unclear: table doesn't specify whether the oracle drives the actual FastAPI route or a lower-level helper; acceptable under "no live bridge" but make explicit they invoke the real endpoint/tick, else the C16 locus problem recurs silently. C10,C11 injected-interleaving/crash-sim with explicit fail-against-pre-change — real (H4). C13 ambient.py read — real (build_interior_continuity_block, ambient.py:22-45). C15 list_active_sessions boundedness — real if against the real accessor. C16 continuous-client post-rollover redirect — NO, PROXY: oracle is ingest_turn(old_sid,…) bare unit call, skips get_or_hydrate_session + /chat//stream handlers where the real 404 occurs; a PASS provides zero evidence against P-1 and would ship the defect. Re-point C16 to drive the actual HTTP path (or get_or_hydrate_session directly) before it can gate anything. A1,A2 advisory correctly labelled (no live/replay workload; build-hygiene vs behavior).

## 6. Bottom line
Worst: BLOCKER (P-1). Plan's central safety claim for trigger B (rolled_to.json makes the weekly swap "correct-by-construction," "no orphaned turns") untrue at the locus that matters. Routing: back to stage 2, narrowly scoped. Rest of plan well-designed, well-cited, faithful. Only the successor-pointer redirect (spec §2e-B, plan §0/§2.2/§2.3, C16) needs rework + re-point C16 to the real path. The 4 minors are fix-in-place.
```

## Author disposition round 2 (orchestrator-routed; gate-4 entry in `decisions.md`)
Route: **Blocker → return to stage 2 (narrowly scoped).** All 7 carried findings confirmed resolved. Fixes:
- **P-1 (Blocker):** redirect made real via **option (a)** — `get_or_hydrate_session` (`session.py:129-205`, the
  shared chokepoint both `/chat` :2360-2366 and `/stream` :2410-2428 funnel through) consults `rolled_to.json`
  and follows it to the successor before returning None/404. Touched files expanded to `brain/chat/session.py`
  (Hana's `brain/chat/*` zone — noted for PR #58). **C16 re-pointed** to drive `get_or_hydrate_session`
  (+ a `/chat` TestClient assertion), not the bare `ingest_turn` unit call.
- **Minor (crash mid-cascade-write):** plan states all three tiers computed from the pre-pass snapshot +
  installed in ONE `rewrite_session_atomic`; new criterion **C17** (atomic cascade write).
- **Minor (dir-fsync):** cite + reuse the posix-guarded `attempt_heal.py:250-266` pattern (Windows CI).
- **Minor (migration ordering):** one-line note — legacy `run_backlog_migration` drain of a sectioned row
  self-heals via the tolerant reader (§1.1); confirmed not data-loss.
- **Minor (C8/C9 locus):** criteria state the fixtures drive the REAL `/sessions/active` endpoint + daily-tick
  function (TestClient / real function), not bespoke helpers.
- **Minor (CH8-4 cursor continuity):** new criterion **C18** — the seeded summary row's cursor prevents
  re-extraction of the seed.
- **Nitpick (empty/corrupt finalize branch):** plan §2.3 note — extraction-only applies to the `if not turns:`
  branch too (no buffer delete there either; poison-move path unchanged).

---

# Stage-3 RE-RUN (round 3) — narrowly-revised {spec, criteria, plan} (P-1 fix + 4 minors + nitpick)

## Reviewer
- **Agent type / model:** general-purpose, **sonnet**. Carried-forward: P-1 (Blocker), 4 minors, 1 nitpick.

## Revised artifact hashes (author-side, base cd29bc61)
```
8f1f42b3a8aec83a962b2e44c310da7bc404a7e26dd7f8a33fbbe3acb437800c  1-spec.md
a1c05b153bfd540433f8f92acd469fa021e2d70c48be63ccc61569c5205d18f9  1.5-criteria.md
a4c0caa1a330d24d5812a77f15ff3f91f99d438dff6fbc9ebe16770dbd53b78e  2-plan.md
```
Author note (for the disposition, verified independently of the reviewer): `session.py` `_LOCK` is a
reentrant `threading.RLock` (`session.py:97`), so the round-2 P-1 fix's pointer-follow recursion inside
`get_or_hydrate_session` is deadlock-safe; the chain-cap bounds a cyclic pointer.

## Reviewer verbatim output (round 3)
_(pending re-review completion)_
