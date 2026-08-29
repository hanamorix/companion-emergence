# decisions.md — run-orchestration hardening (item B, #139)

Append-only gate log. Iteration cap reads this.

## Path validation (run start — gate 4 precondition)
Validated 2026-08-25 (branch `ThinkerOfThoughts/harness-drop-in-code-ingest`). All cold-review
context paths + spec touched-files-parents exist and are readable:
- OK  tests/harness/roster_preflight.py
- OK  tests/harness/dropin.py
- OK  tests/harness/engine.py
- OK  tests/harness/runner.py
- OK  tests/harness/__init__.py
- OK  tests/harness/README.md
- OK  tests/unit/harness/  (new test lands here)
- OK  ~/Desktop/companion-emergence/hunts/symptom-cluster-rootcause/live-run/RUN-INSTRUCTIONS.md (read-only input)
- OK  ~/Desktop/companion-emergence/hunts/symptom-cluster-rootcause/live-run/run_arm.sh (read-only, handoff target)
- OK  changes/harness-drop-in-code-ingest/9-handoff-live-run-callsite-edits.md (handoff precedent)

No dead paths. Recorded so gate 4 may pass.

## Stage 0
SKIPPED — greenfield capability (new module + doc + banner); no prior version behavior to
snapshot. Stage 8 = conformance-only.

## Gate 4 — iteration 1 (2026-08-25)
- Stage-3 cold review (Opus, synchronous/`run_in_background:false` so it returned in-turn — no
  main relay needed; this dogfoods item B's foreground pattern). Verbatim record:
  `3-redteam-plan.reviewer-verbatim.md`. Path validation (above) recorded — gate-4 precondition met.
- Worst finding: **BLOCKER (F1)** — `boot_and_verify`'s planned default verify seams
  (`assert_brain_under_dropin`, `assert_brain_tools_roster`) are process-SELF-checks; called from
  the orchestrator process they raise `DropinMismatch` on a HEALTHY run, and the injected-fake
  criteria (G1/G2/G3) never exercised the real defaults (a proxy dodge). Reviewer correct;
  verified against `dropin.py:217-219`, `roster_preflight.py:67-68`, item A handoff:121-131.
- Route: **BLOCKER → revise {1-spec, 1.5-criteria, 2-plan}, re-run stage 3** (iteration 1 of the
  cap). Redesign: `boot_and_verify` = orchestrator-side READY-confirmation (+ optional READY
  `brain_repo` cross-check), NOT re-running the in-process asserts (which stay inside
  `live_server.py` per item A). Added `assert_tool_callable` (real transcript-based tool-side
  check = RUN-INSTRUCTIONS step 4). G1/G2/G3 now exercise REAL deterministic paths (no fakes).
- Minors folded into the revision: F2 (removed false "already used in Testing's draft"), F3
  (orphan-server cleanup documented → A4), F4 (README points to one canonical doc → A5), A(ii)
  (`preserve_artifacts` `require=` raises on a missing required source → G4). F5 considered and
  DEFERRED with rationale (wrapper fits neither inline nor AgentBob drive).
- NOT a human-gate: the blocker is a flaw in the plan (the loop's own artifact), resolvable from
  the brief + source; fixed and re-reviewed, per the loop.

## Gate 4 — iteration 2 → PASS to build (2026-08-25)
- Stage-3 re-review (Opus, synchronous). Verbatim record: `3-redteam-plan.reviewer-verbatim.iter2.md`.
- Worst finding: **MINOR** (F1 verified RESOLVED against source — dropin.py:217-219,226-227;
  live_server.py:202/270-295/193-194/372-375; brain_repo key :271; DropinMismatch RuntimeError
  dropin.py:69). No new blocker/major. Residuals N1-N5 all MINOR/NIT.
- Residuals folded into plan/criteria before pass: N4 (error_markers=("ERROR",)), N3 (poll-mode
  mtime guard), N1 (tolerant roster extraction + handoff caveat), N2 (handoff require=), N5 left
  advisory. See iter2 record "Author disposition."
- Route: **worst = MINOR → fix-and-proceed → BUILD (stage 5).** Criteria FROZEN at this point
  (G1-G8 gating, A1-A5 advisory; sha of frozen criteria recorded at build).
- Iteration cap: this is the 2nd stage-3 pass but on a DIFFERENT finding class (iter1 = the F1
  design blocker; iter2 = minor residuals only, cleared). No same-finding-class 2x bounce; no
  human tiebreak triggered.
- FROZEN criteria sha256: `b4a84c6e76a5101042f1606b9cb3d2652fd7c0daae9794a87dbdd21a527e43d5`
  (changes/run-orchestration-hardening/1.5-criteria.md). Plan sha256:
  `50725e42d233db1469095095caf56250d83969528a4ccc602958bbc12797b4f6`.

## Stage 5 — build
Delegated to a sonnet subagent (implementation to a fully-pinned plan; sonnet sufficient +
cheaper than opus for code-to-spec). Built `tests/harness/foreground.py`,
`tests/harness/RUN-ORCHESTRATION.md`, `tests/unit/harness/test_foreground.py`; edited
`tests/harness/__init__.py` (+11 exports) and `README.md` (pointer). Runner verified deliverables
on disk + ran CI (ruff clean; 25 new tests pass).

## Gate 7 — fix-and-proceed (2026-08-25)
- Stage-6 cold code review (Opus, synchronous). Verbatim record:
  `6-redteam-code.reviewer-verbatim.md`. Mechanical diff: `6-code-diff.patch`.
- Worst finding: **MINOR** (no blocker/major). Reviewer empirically verified the core
  guarantees: no helper can hang (bounded subprocess+poll, stress-tested); the prior self-assert
  blocker is NOT regressed (neither assert called); banner byte-exact; scope clean (no brain/**,
  no gitignored-lane edit).
- Route: **worst = MINOR → fix-and-proceed** (no re-loop). Fixes applied (delegated, sonnet):
  Finding 1 (require-enforcement gap → `confirmed`-set post-loop check + discriminating test),
  Finding 2 (boot_cmd freshness guard), Finding 3 (pipe-holder pitfall doc). Finding 4 resolved
  by authoring the stage-8 + handoff artifacts; Finding 5 (NIT) accepted as-is.
- Post-fix: ruff clean; 27 foreground tests pass; fix agent confirmed the new test fails pre-fix
  (discriminating). Full scoped CI re-run confirms no new failures (see stage 8).
- Iteration cap: gate 7 hit once, fix-and-proceed; no bounce, no human tiebreak.

## Stage 8 — harness (conformance-only) → DONE
- Record: `8-harness.md`. Mode conformance-only (greenfield, no baseline; no production path → no
  regression surface).
- All gating G1-G8 PASS; all advisory A1-A5 PASS. Anti-stall guarantee is structural/contractual
  by design (measurement-apparatus reasoning stated in 8-harness.md), not agent-behavior-tested.
- CI: ruff clean; scoped suite `1 failed, 4352 passed` — sole failure = known out-of-scope #136
  (`test_review_tick_gate_blocks_send_records_hold`); #155 passed. Zero new failures. 27/27
  foreground unit tests pass.
- VERDICT: ship-eligible. Loop complete.
