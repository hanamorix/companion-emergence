# 6 — Red-team of the code (persisted-cadence, defer #21)

Cold independent reviewer (general-purpose subagent, no shared context). Read
{1.5-criteria, 2-plan} + the `342514af..HEAD` diff + persisted_cadence.py +
the whole `run_folded` + the 4 through-path tests + soul/cadence.py.

**Worst-severity verdict: MINOR.** No blocker, no major. Ship after fixing the
stale comment.

## Findings

### Factual — clean (criteria verified met)
- **C1 MET** — `persisted_cadence.py:33-71` exposes the 5 symbols; fail-safe on
  missing/OSError/JSONDecodeError (`:55-58`), non-dict (`:59-60`), bad-ts
  (`_parse_ts` `:38-43`). 8 helper tests pass.
- **C2 MET** — all three through-path tests assert fire + state rewritten with
  future `next_at` (`test_...:62-63, 110-112, 144-146`), at intervals a monotonic
  timer could never reach.
- **C3 MET** — grep for the three monotonic locals → EMPTY.
- **C5 MET (byte-level)** — word-diffed each tick body vs base `342514af`:
  `_run_finalize_tick(...)`, `forgetting_run_pass(...)` + narrative + sweep,
  `_run_voice_reflection_tick(...)` identical. Only guards + footers changed.
- C4 deferred to stage 8 (full suite).
- **C6 MET by entailment** — `is_due(CadenceState(None))` True + C2.

### Logical — clean
- Maintenance end-of-block advance is genuinely unreachable-by-exception: every
  statement between the `if` and the advance is individually try/except-wrapped,
  incl. the `from brain.files import pending` import (inside its try). Verified
  line-by-line.
- finalise/voice `finally` runs on success + exception; `stop_event` can't skip
  the save (stop check is at loop top, not mid-block). M1 canary confirms.
- Fresh `datetime.now(UTC)` in guard vs advance → micro-skew only makes `next_at`
  infinitesimally later (strictly safe).
- No fire-every-tick storm: in-memory state reassigned to the future state before
  the next `is_due`.

### Missed opportunity
- M1 canary has real teeth (would fail if advance moved into the try).
- **(nitpick)** maintenance lacked its own exception-advance canary — the
  no-`finally` path is the more fragile one. → FIXED: added
  `test_maintenance_advances_even_when_forgetting_raises`.
- **(nitpick)** no disabled-cadence no-file assertion. → FIXED: added
  `test_disabled_cadence_writes_no_state_file`.

### Unstated assumptions
- Maintenance couples to `soul_review_interval_s` for both gate + advance,
  consistently; cadences load independently (no cross-contamination when one is
  None). Verified.
- **(minor)** stale comment at `supervisor.py:376-378` still said "monotonic
  timer". → FIXED (now "PERSISTED wall-clock … its OWN state file").

## Routing
Worst = minor → fix-in-place (stale comment) + the two nitpick tests, then
proceed to stage 8 (harness). All three fixed; 6 cadence tests + 8 helper tests
green; ruff clean.
