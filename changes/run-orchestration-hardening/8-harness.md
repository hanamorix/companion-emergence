# 8 — Harness / conformance record

**Mode: conformance-only.** No stage-0 baseline (greenfield capability — new module + doc +
banner; no prior behavior to regress against). The project's standing regression metrics
(cost/cache/turns from `chat_usage.jsonl`) are untouched: this change adds no production code
path and runs no chat turns, so there is nothing to regress. Conformance = actual vs. the frozen
`1.5-criteria.md` (sha `b4a84c6e76a5101042f1606b9cb3d2652fd7c0daae9794a87dbdd21a527e43d5`).

## The anti-stall guarantee is STRUCTURAL/CONTRACTUAL, not behavioral-test-enforced (measurement-apparatus reasoning)

This is stated plainly, per the brief and the standing rule. The guarantee that a run
orchestrator will not background-a-child-and-await is delivered by three things, none of which is
an agent-behavior test:
1. **Structural shape of the helpers** — `boot_and_verify`, `assert_tool_callable`, and
   `preserve_artifacts` each block and return/raise in-turn, bounded by a timeout so none can
   hang. There is no async/await surface and no multi-step seam inside a call to park on.
2. **The contract** — `tests/harness/RUN-ORCHESTRATION.md` states the one rule and names both
   parkable surfaces (a `run_in_background: true` Bash task AND a background sub-agent).
3. **The loud nudge** — `drive_now_banner` at the READY decision point.

We deliberately did NOT build a harness that spawns real orchestrators to prove they comply. That
is the "measurement-apparatus trap" this project has repeatedly paid for: agent behavior is
non-deterministic (such a test needs repeated trials + a stated pass rate to mean anything), the
harness is itself an AI artifact that must pass the same red-team, and effort migrates into an
instrument that never terminates while the real change ships nothing. Per the standing rule we
capped harness effort and proved only the **mechanical facts** below. The test mechanism did not
bounce (it passed on first construction; the one code fix — Finding 1 — was a real defect the
review caught, not a harness that couldn't measure). Note also that this very gc run **dogfooded
the fix**: every cold reviewer + build agent was spawned `run_in_background: false` (synchronous),
so each returned its result to the loop runner in-turn — the exact foreground pattern, with zero
main-relay stalls.

## Per-criterion conformance (gating)

Measured by `uv run pytest tests/unit/harness/test_foreground.py -q` (27 passed) + `uv run ruff
check .` (clean) + inspection. Post-fix file shas: `foreground.py`
`da972b5847ffc98a3f00f8b9a1cd07e109d8d2a5b50c0b5a4cdb406803401f9b`; `test_foreground.py`
`dfb8e48bf5d71471f1cfe95ea8bb0d4dde9c7197af55b13f17cd793a980c98e7`; `RUN-ORCHESTRATION.md`
`4b4964178aac6110b030f6227011b5ca953871cf377db8782c7c327eaac1515c`.

| Crit | Verdict | Evidence |
|---|---|---|
| **G1** helper blocks & returns populated `ArmSession` via the real path | PASS | boot_cmd writes a real READY → populated `ArmSession`; real shipped READY-parse logic, no fakes |
| **G2** dead/failed boot raises in-turn, no hang | PASS | non-zero exit / ERROR marker / no-READY-in-timeout each raise `ForegroundBootError`; test asserts elapsed < 5s for a 0.5s timeout; reviewer stress-tested the subprocess pipe-hang → bounded |
| **G3** real verification paths reject a bad build in-turn | PASS | READY `brain_repo` outside `expect_brain_repo_under` → raise; `assert_tool_callable` absent-tool → `ToolSideBroken`, present-tool → clean; real logic, no injected proxy |
| **G4** preserve copies + raises on missing REQUIRED | PASS (hardened post-review) | copied listed + non-required missing in `.missing`; a `require`d source absent OR not-in-names → `PreservationIncomplete` (Finding-1 fix + discriminating test) |
| **G5** banner exact text | PASS | first line `== DRIVE_NOW_BANNER_HEADLINE` byte-for-byte (▶ + em-dash); both surfaces + rule present |
| **G6** contract doc ships, names both surfaces + rule | PASS | test reads `RUN-ORCHESTRATION.md`, asserts both surfaces + the one rule |
| **G7** public API exported/importable | PASS | all 11 symbols import from `tests.harness` + present in `__all__` |
| **G8** CI green (scoped, minus the 2 known out-of-scope failures) | PASS | see below |

**G8 detail.** `uv run ruff check .` → clean. `uv run pytest -m "not live and not
requires_claude_cli and not integration" -q`: pre-fix full run = `1 failed, 4350 passed, 19
skipped, 6 deselected, 1 xfailed` — the single failure is the known out-of-scope
`test_review_tick_gate_blocks_send_records_hold` (#136, awaiting PR #142); the other known-flaky
`test_history_returns_buffered_turns_in_order` (#155, order-dependent) passed this ordering. The
change introduces ZERO new failures. Post-fix confirmatory full run: `1 failed, 4352 passed, 19
skipped, 6 deselected, 1 xfailed` in 207s — the +2 vs. pre-fix are the two new preserve tests;
the sole failure is again the known out-of-scope `test_review_tick_gate_blocks_send_records_hold`
(#136); #155 passed. G8 confirmed.

## Advisory
- **A1** no live/token dep — PASS (tests use tmp_path/subprocess/threading only).
- **A2** no gitignored path hardcoded — PASS (`run_arm.sh`/`hunts` appear only in prose
  docstrings, not as operational path literals; boot_cmd/subproc_dirs caller-supplied).
- **A3** handoff note complete — PASS (`9-handoff-drive-banner-callsite.md`: the `run_arm.sh`
  banner call-site edit + a one-turn worked example that sets `require=` + the N1 field caveat).
- **A4** boot-failure cleanup documented — PASS (`RUN-ORCHESTRATION.md` clean-slate-before-retry).
- **A5** no duplicated contract — PASS (the one-rule text lives only in `RUN-ORCHESTRATION.md`;
  README is a pointer; the NIT phrase-echo does not create a divergent operational copy).

## Verdict
All gating criteria PASS; all advisory PASS. Conformance-only (no baseline). Anti-stall guarantee
is structural/contractual by design (measurement-apparatus reasoning, above). **Ship-eligible.**
