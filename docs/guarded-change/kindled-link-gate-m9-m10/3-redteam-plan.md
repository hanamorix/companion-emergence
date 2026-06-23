# 3 — Plan red-team (cold subagent)

Cold reviewer, no shared context, four lenses, source-cited. Citations spot-verified
by the operator (L52 policy ✓, `ASIA`+16 caps-prose match ✓, `eyJ`=base64`{"` ✓).

## Verdict: MAJOR (2 majors, both reviewer-confirmed non-blockers — fail toward hold)

### Lens 1 — Factual: CLEAN (one nitpick)
- All line refs verified against source. The `len(hits)` revise/hold policy is at
  **L52** (docs said L51) — nitpick, corrected in 2-plan.
- Safety invariant verified TRUE: `transcript_summary` never reaches
  `_prefilter`/`_apply_budget`.
- m10 crumb-leak verified REAL: depleted→revise→re-gate→can send one
  `MIN_SEND_DEBIT` crumb/attempt (`session_engine.py:188-191,198-219`).
- "No session_engine change needed" verified TRUE (hold maps through existing code).

### Lens 2 — Logical: CLEAN
- Depletion-first/tighten-second ordering correct (DEPLETED < TIGHTEN, hold must win).
- `<` vs `<=` at floor matches C-m10-5 (`budget==floor` → revise; can afford one debit).
- Single Stripe key → 1 hit → revise (no accidental double-hit hold); assignment +
  vendor co-fire → hold = safe direction.

### Lens 3 — Missed opportunity
- **major:** C-m9-2 FP corpus lacks an all-caps prose case; `ASIARECENTLYTHISYEAR`
  matches the broad AWS pattern. → narrowed AWS to `AKIA|ASIA`, added caps corpus line.
- **minor:** AWS prefix list broader than spec (`AGPA/AIDA/AROA/…` are resource-id,
  not credential, prefixes). → dropped to `AKIA|ASIA`.
- **minor:** JWT short-segment near-miss untested. → added `eyJ.x.y` corpus line.
- **minor:** Stripe `whsec_` family omitted with no rationale. → noted in spec (lower
  signal, deferred).

### Lens 4 — Unstated assumptions & risks
- **major:** JWT pattern false-positives on a base64-of-JSON `relationship_hint`
  value (`eyJ` = base64 `{"`), the one structured-data surface `_payload_text` scans.
  → DESIGN CALL: keep JWT (fails toward hold, never toward send — the correct gate
  bias); corrected the overstated "near-zero FP" claim; added C-m9-2b accepted-hold
  test.
- **ReDoS on the 6 new patterns: CLEARED** (<0.1ms on pathological inputs).
- **`\b` around `-`/`_`: CLEARED.**
- **minor:** C-m10-4 must debit + read budget at the SAME `now` (linear refill would
  mask the hold). → folded into the criterion.

## Routing (gate 4)
Major → re-plan. Operator call: both majors are reviewer-confirmed non-blockers
(fail-toward-hold); resolution = recorded design call + test additions + pattern
narrowing, not an approach change. Folded into revised 1-spec / 1.5-criteria / 2-plan;
stage 6 (cold code review of the real tests) is the independent challenge of the
resolution. See decisions.md gate 4.
