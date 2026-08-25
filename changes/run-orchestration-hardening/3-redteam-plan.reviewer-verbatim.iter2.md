# 3 — Stage-3 red-team, iteration 2 (verbatim record) — re-review of revised plan

## Reviewer
- Agent type `general-purpose`; model Opus 4.8 (`claude-opus-4-8`); `run_in_background: false`
  (synchronous, returned in-turn). Cold, no shared context with the author.

## Context given (closed set)
Revised `changes/run-orchestration-hardening/{1-spec,1.5-criteria,2-plan}.md` + source:
`tests/harness/dropin.py`, `roster_preflight.py`, `__init__.py`,
`changes/harness-drop-in-code-ingest/9-handoff-live-run-callsite-edits.md`,
`hunts/…/live-run/RUN-INSTRUCTIONS.md`, and (added for this re-review, to verify the redesign's
load-bearing claims) `hunts/…/live-run/live_server.py`.

## Charter given
Red-team core (five lenses + discipline) + CH8 + CH9/CH10 + explicit F1-resolution adjudication:
confirm (i) both asserts are process-self checks, (ii) READY-only-after-guard + ERROR-on-fail,
(iii) the READY payload has a `brain_repo` key, (iv) G1/G2/G3 now exercise real paths.

## Reviewer context-file sha256 (reviewer-reported)
```
7994b184c59ac59c55c9295cdfbd21afc39ff5603e535685c693c950331c4263  1-spec.md
d5acb464e404314b9fd0a0dc37ab620d7b5614580442ba91a4e59a8be4a08bec  1.5-criteria.md
09f4c6f0cf7860dba6d6c2ad0467743832cebc7d88224ef6c48f81e856ce7017  2-plan.md
52af3e1c4c0d86306e822e444dbeb6b2873e216e373f7b3dd56a51e107c31c2e  dropin.py
9a66470cf937599aa5dca15568df6b808296fe6929b71e09cf03f9ce1fdb0aa7  roster_preflight.py
13f32f831d4c50f8d5b514c06b626ff56e0eb00dd64b881fd163474456de9826  __init__.py
cbf624c4b5dc099102f5677478b4dac2c58a491e5885611aca0548f8e846539d  9-handoff-live-run-callsite-edits.md
d62f05134394a72bdf82e19e68304103f484320632f4c77c13c8cd145f153df5  RUN-INSTRUCTIONS.md
c6742fcb7452974e48248f8624f076d4ecf6bcabf1baf14af6831eca83ac4c24  live_server.py
```

## Verdict: F1 RESOLVED (verified, not relabeled). No new BLOCKER/MAJOR.

Five load-bearing claims all confirmed against source:
1. `assert_brain_under_dropin` is a `sys.prefix` self-check (dropin.py:217-219) + in-process
   `brain` origin under `build.repo` (dropin.py:226-227) → correct NOT to call orchestrator-side.
2. `assert_brain_tools_roster` requires in-process `brain` under `build.repo`
   (roster_preflight.py:67-68) → meaningless orchestrator-side.
3. READY-present-without-ERROR == guard passed: guard at live_server.py:202 before any token
   spend; READY written :270-295 only after; markers unlinked at :193-194 (no stale carryover
   within a fresh boot); any hard-fail caught at :372-375 writes ERROR (DropinMismatch is a
   RuntimeError, dropin.py:69) → the only way READY exists without ERROR is a passed guard.
4. READY payload has `brain_repo` (live_server.py:271, value `<dropin>/brain` from :203) → the
   `expect_brain_repo_under` cross-check targets a real key.
5. G1/G2/G3 exercise REAL deterministic paths, no injected verify fakes (CH9/CH10 table below).
Matches Testing's own draft (RUN-INSTRUCTIONS step 3 = READY-without-hard-fail; step 4 =
transcript `sent.system` tool-side check).

## Residual findings — all MINOR/NIT (fix-and-proceed)
- **N1 (MINOR):** G3(b) exercises the real scanning logic but against a CRAFTED turn_diag; the
  real `sent.system` serialization is in the gitignored lane, UNVERIFIABLE from the closed set.
  Not the F1 proxy (real code runs), but fixture fidelity is unproven — pin to a real sample /
  have Testing confirm the field.
- **N2 (MINOR):** `preserve_artifacts` default `require=()` leaves core-artifact protection
  opt-in; a run missing turn_diag/transcript preserves "green." Handoff worked-example must set
  `require=`.
- **N3 (MINOR):** poll-mode (`boot_cmd=None`) reads `<run_dir>/READY` without an mtime guard; a
  leftover READY in a reused run dir could false-confirm. Mitigated by clean-slate rule only.
- **N4 (NIT):** `PAUSED_ON_LEAK.json` as a default boot `error_marker` conflates a teardown
  pause with boot failure; harmless (only written at teardown, never in the boot window).
- **N5 (MINOR):** the README orchestration-section edit's prose correctness is only
  advisory-checked (G8 CI does not validate prose); a botched edit would pass gating. Low sev.

Lenses: factual clean (cited); logical clean; fidelity clean (all terms pinned — boot/verify/
confirm/tool-side/preserve/foreground/blocking/does-not-hang/one-turn/banner). CH8: no
BLOCKER/MAJOR-severity uncovered behavior; the untested "orchestrator doesn't background" is the
correct measurement-apparatus scope boundary. CH11/CH12 N/A.

## Author disposition of residuals (folded into revised plan/criteria before gate-4 pass)
- N4 → default `error_markers=("ERROR",)` (plan updated).
- N3 → poll-mode start-timestamp / mtime guard added (plan updated).
- N1 → `assert_tool_callable` extraction made shape-tolerant + handoff caveat that Testing
  confirms the served-roster field against a real turn_diag (plan + criteria A3 updated).
- N2 → handoff worked-example sets `require=` for the three core artifacts; criteria A3 updated.
- N5 → left advisory (doc edit; the build does a surgical README edit; stage-6 diff review + G6
  content assertion cover the substance). Accepted.
