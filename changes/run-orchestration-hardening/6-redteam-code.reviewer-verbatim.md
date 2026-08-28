# 6 — Stage-6 red-team of the built code (verbatim record)

## Reviewer
- Agent type `general-purpose`; model Opus 4.8 (`claude-opus-4-8`); `run_in_background: false`
  (synchronous, returned in-turn). Cold, no shared context with the author.

## Reviewed diff — generated MECHANICALLY (ST6d)
`git diff -- tests/harness/foreground.py tests/harness/__init__.py tests/harness/README.md
tests/unit/harness/test_foreground.py` → `changes/run-orchestration-hardening/6-code-diff.patch`
(the new untracked `RUN-ORCHESTRATION.md` reviewed as a full file). Reviewer confirmed the patch
matches the working tree byte-for-byte (modulo index hash lines).

## Context given (closed set)
Built artifacts (`foreground.py`, `RUN-ORCHESTRATION.md`, `test_foreground.py`, `__init__.py`,
`README.md`, `6-code-diff.patch`) + frozen `{1.5-criteria, 2-plan}` + source
(`roster_preflight.py`, `dropin.py`, `hunts/…/live-run/RUN-INSTRUCTIONS.md` read-only).

## Charter given
Red-team core (five lenses + discipline + spot-verify) + stage-6 mechanical-diff duty. Task
scrutiny: can any function HANG (unbounded wait)?; is the prior self-assert blocker regressed
(boot_and_verify must NOT call assert_brain_under_dropin/assert_brain_tools_roster)?; stale-READY
mtime guard present?; banner headline byte-exact + first line?; are the tests DISCRIMINATING per
G1-G7?; does the doc name both surfaces + the rule?; did the change touch any out-of-scope file?

## Reviewer context-file sha256 (reviewer-reported; pre-fix build)
```
7b3e6e1c75a4fbc755eec1c6554c97c320f4910f908b0fbc81e5ff0cda9c36bf  foreground.py
5f0936e2c1943fd6aa595dd004724483ce070a5f5c257be5bd6246c18372aed6  RUN-ORCHESTRATION.md
b68d6a4707799b66de738aad2cde606eba5dc25f9335c17337c37d5412f0d6d4  test_foreground.py
52d8a9579f1ee4776a7eab94a7b455e18f3db2c10389792d7b4d533f495827ad  __init__.py
f5558350907051ede4dd4fb8cbd1cd899e684f3e5ec2006f6bc961cfd39f3939  README.md
9a66470cf937599aa5dca15568df6b808296fe6929b71e09cf03f9ce1fdb0aa7  roster_preflight.py
52af3e1c4c0d86306e822e444dbeb6b2873e216e373f7b3dd56a51e107c31c2e  dropin.py
b4a84c6e76a5101042f1606b9cb3d2652fd7c0daae9794a87dbdd21a527e43d5  1.5-criteria.md
50725e42d233db1469095095caf56250d83969528a4ccc602958bbc12797b4f6  2-plan.md
```

## Verdict: NO BLOCKER, NO MAJOR. 4× MINOR + 1× NIT.

Empirically verified (reviewer ran ruff + the unit suite + live probes):
- **No function can hang.** `subprocess.run(timeout=ready_timeout)` (foreground.py:156-163),
  `TimeoutExpired`→`ForegroundBootError` (:164-168); reviewer stress-tested the post-timeout
  `communicate()` hang (child + surviving grandchild holding the captured pipe) → raised at 3.02s
  vs 3s timeout, bounded. Poll loop `while time.time() < deadline` terminates (:206-223); test
  asserts elapsed < 5.0 for a 0.5s timeout.
- **Prior self-assert blocker NOT regressed.** grep for the two asserts in foreground.py hits
  only the design-note docstring (line 27); neither is invoked.
- **Stale-READY mtime guard present + discriminating** (poll mode): `start=time.time()` (:210),
  accept READY only if mtime ≥ start (:217); tests backdate by 3600s → raises, fresh → accepted.
- **Banner headline byte-exact** (▶ U+25B6 + em-dash U+2014), matches criteria+plan; first line;
  names both surfaces + the rule; G5 asserts first-line equality.
- **Doc** names the one rule + both surfaces + F3 clean-slate + never-background warning.
- **Scope clean** — only the five in-scope files; no `brain/**`, no `hunts/.../live-run/` edit;
  companion-emergence `RUN-INSTRUCTIONS.md` untouched.

Per-criterion: G1-G3, G5-G7 satisfied + discriminating; G4 satisfied for tested cases but the
test did not cover the require-subset gap (Finding 1); G8 partly verified by reviewer (ruff
clean + 25/25 new tests; full scoped suite the runner's job). A1/A2/A4/A5 met; A3 (handoff note)
not-yet-created (Finding 4).

## Findings
- **Finding 1 (MINOR, logical/fidelity):** `preserve_artifacts` enforces `require=` only INSIDE
  the names/subproc_dirs loops, so a required entry not also in names/subproc_dirs is silently
  never enforced — reviewer reproduced live: `preserve_artifacts(run,
  names=("turn_diag.jsonl",), require=("transcript.jsonl",))` returns green, no raise. Weakens
  the `PreservationIncomplete` guarantee.
- **Finding 2 (MINOR, risk):** boot_cmd branch had no stale-READY freshness guard (poll branch
  did) — a boot_cmd exiting 0 without writing READY into a reused run_dir with a leftover READY
  could false-confirm.
- **Finding 3 (MINOR, usage):** a boot_cmd leaving a surviving pipe-holder blocks
  `subprocess.run(capture_output=True)` until ready_timeout even on success (bounded, raises;
  not a hang). Doc steers server boots to poll mode; add an explicit note.
- **Finding 4 (advisory, expected):** the A3 handoff note + `8-harness.md` did not yet exist at
  stage-6 time — later-stage artifacts, not omissions.
- **Finding 5 (NIT):** the README pointer echoes the rule phrase; a descriptive pointer, not a
  divergent operational copy — does not defeat single-source.

Spot-verifications (reviewer): headline equality; brain_repo under-check branch; self-asserts
uncalled; Finding 1 reproduced live; Finding 3 reproduced live.

## Author disposition (applied post-stage-6; see gate-7 in decisions.md)
- Finding 1 → FIXED: `preserve_artifacts` now tracks a `confirmed` set and raises
  `PreservationIncomplete` for ANY required entry not confirmed-present-and-copied (missing on
  disk OR never listed in names/subproc_dirs). New discriminating test added (fails pre-fix).
- Finding 2 → FIXED: freshness guard now applied to the boot_cmd branch too (start timestamp
  before subprocess.run; READY mtime must be ≥ start).
- Finding 3 → FIXED (doc): pitfall note added to the `boot_and_verify` docstring + a
  `RUN-ORCHESTRATION.md` pitfall section.
- Finding 4 → RESOLVED: `9-handoff-drive-banner-callsite.md` and `8-harness.md` authored at
  stage 8 (below).
- Finding 5 → ACCEPTED as-is (NIT; descriptive pointer, reviewer confirmed it does not defeat
  single-source).
