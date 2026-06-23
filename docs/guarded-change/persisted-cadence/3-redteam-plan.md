# 3 — Red-team of the plan (persisted-cadence, defer #21)

Cold independent reviewer (general-purpose subagent, no shared context). Read
{1-spec, 1.5-criteria, 2-plan} + full design + full plan + supervisor.py +
soul/cadence.py + test_supervisor_soul_cadence.py.

**Worst-severity verdict: MINOR.** No blocker, no major.

## Findings

### Factual — clean (citations earned)
- F1 `run_folded` signature accepts every test kwarg (`supervisor.py:117-136`; `stop_event` first positional). VERIFIED.
- F2 every quoted "current code" block matches source (init `:177/179/182`, maintenance guard 367-371 + footer 395, finalise 401-415, voice 449-458). VERIFIED.
- F3 line numbers accurate.
- F4 helper mirrors `soul/cadence.py` (`_parse_ts` 49-56, fail-safe load 59-77, atomic save 80-92, is_due 95-97). VERIFIED.
- F5 (nitpick) new helper uses `path.suffix + ".tmp"` (→ `.json.tmp`) vs soul's `.with_suffix(".json.tmp")` — functionally identical, leaves no `.tmp`.

### Logical — clean
- L1 (headline) maintenance "block body cannot raise" → VERIFIED TRUE: `supervisor.py:372-394` every statement is inside one of three try/except blocks; the `from brain.files import pending` import is INSIDE the third try (390), no bare statement between blocks. The finally-vs-end-of-block asymmetry is justified.
- L2 watchdog terminates: loop `wait(timeout=tick_interval_s=0.05)` on `while not stop_event.is_set()` (265, 522) → exits ≤0.05s after watchdog fires.
- L3 no fire-every-tick storm: `advance` reassigns the in-memory state local before `save`, so even a save failure holds the future next_at in-process.
- L4 no `*_cadence.json` collision (grep: zero existing readers/writers).

### Missed opportunity
- **M1 (MINOR) — the always-advance-on-exception contract has NO test.** The through-path tests only cover the happy path. The plan's own most-load-bearing claim (advance+save runs even when the tick raises, ranked "major if violated") has zero coverage; a future refactor moving `advance` inside the `try` would pass all tests. **Recommend a raising-tick canary.** → ACCEPTED, folded into build.
- M2 generic-vs-soul divergence sound (no maintenance hazard). Not a finding.

### Unstated assumptions
- A1 (nitpick) `datetime.now(UTC)` called twice per fire (guard + advance) → microsecond skew; identical to existing soul cadence + old monotonic footer. Not a regression. Accept.
- A2 multiple cadences due-now on first boot fire sequentially, each advancing its own file. No competition. Clean.
- A3 (highest-risk, VERIFIED SAFE) existing `test_supervisor_finalize_cadence_drops_old_sessions` (`finalize_interval_s=0.05`) still passes — missing state file → due-now → fires first tick (faster than old monotonic). No monotonic-behaviour assertion broken (grep confirms).
- A4 (nitpick) C4 hardcodes "≥3729 passed" — brittle; update to actual post-build count.

## Routing
Worst = minor → fix-in-place (add M1 canary) + proceed to stage 5 (build). A1/A4 logged as nitpicks (A4: update the C4 count after build).
