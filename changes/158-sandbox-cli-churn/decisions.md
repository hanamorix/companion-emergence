# decisions.md — #158 sandbox CLI-churn exclusion

## Path validation (CFG3, run start — required before gate 4)
Validated readable at run start (2026-08-27):
- `tests/harness/sandbox.py` — OK (read in full).
- `tests/unit/harness/test_sandbox_isolation.py` — OK (read in full).
- `~/Desktop/companion-emergence/brain` (redteam_context #1) — OK (dir exists).
- Layer-2 config `~/Desktop/claude-code-skills/Guarded_change/guarded-change.companion.md` — OK.
- redteam_context Phoebe snapshots (`~/Downloads/Phoebe/*.jsonl`) — NOT needed for this change
  (no cost/cache/tool-log metric involved; detection-scope-only harness change). Not handed to
  the reviewer; noted here so their absence-of-use is deliberate, not a skipped validation.

## Context hashes (for provenance cross-check of the stage-3 reviewer)
- 1-spec.md         a12bc0cf94323aa74ea89ef9bac76ef52ce188c1a990969b4e2843950d49dec0
- 1.5-criteria.md   3ec242ef76d1deacd0dee247c21dee54ee1d0ea8fca27fefe214c5b03337ddc5
- 2-plan.md         505e5db3077bb6dfa655351f679bfceec61f62aa5f27b51e473d0e64714fb39f
- sandbox.py        2d9766d93990d2e5337901cec58d03cf6f87359652826a1ee9eba3104e10b7ca
- test_sandbox_isolation.py  d547a2350ddffc919fc94cb7fb5b787d73400782624614d0096052bb0d1eb744

## Gate log

### Gate 4 (stage-3 plan red-team) — 2026-08-27
- Reviewer: general-purpose / opus (cold). Worst severity: **MINOR**. Record:
  `3-redteam-plan.reviewer-verbatim.md`. Reviewer confirmed hard constraints honored (targeted,
  no-blanket, raise-vs-warn semantics UNCHANGED), mechanics correct, label-audit clean per gating
  criterion, no owner-ratification record present (CH11/CH12 N/A).
- **Route: proceed to build, WITH an orchestrator-directed design amendment.** The reviewer's
  MINOR finding + its own recommended fix (§3 option (a), content-aware pruning) was adopted by
  the orchestrator as a design change (an orchestrator decision, NOT an owner one: it PRESERVES
  the owner-ratified option-(c) "warn" floor rather than changing semantics). Switched from full
  name-pruning to **content-aware / mtime-insensitive** fingerprinting for the two files: silence
  only the benign mtime-only bump, still warn on a genuine content change. Spec/criteria/plan
  updated accordingly.
- Reviewer nits folded in: NITPICK-F1 (provider.py:174→:175 cite fixed in spec); CH8-1 (added
  G4b + a subdir test that a same-basename nested file stays mtime-guarded); CH8-2 (added G1b + a
  content-change-still-warns test, now possible under content-aware).
- No iteration-cap concern (first pass; the amendment tightens toward the reviewer's own
  recommendation, reducing risk — no stage-3 re-run required).

### Gate 7 (stage-6 code red-team) — 2026-08-27
- Reviewer: general-purpose / opus (cold). Worst severity: **CLEAN** (one nitpick, no action).
  Record: `6-redteam-code.reviewer-verbatim.md`. Reviewer ran a revert probe confirming G1
  (mtime-bump-silent) + G4b (subdir same-basename stays guarded) truly gate the mechanism;
  raise/warn block + F4 exclude sets byte-untouched; diff harness-only; the don't-full-prune
  safety argument verified against provider.py.
- Reviewed diff generated mechanically: `git diff` working-tree vs base
  `cde2eb468554fcf672d820150fecfbf48072acc5` → `6-code-diff.patch` (harness-only:
  `tests/harness/sandbox.py` + `tests/unit/harness/test_sandbox_isolation.py`).
- **Route: proceed to stage 8.**

### Gate 8 (conformance) — 2026-08-27
- Conformance-only (no regression workload). All gating criteria met; CI clean bar the two known
  out-of-scope failures. See `8-harness.md`. **Verdict: CONFORMANT — cleared to commit + push.**
