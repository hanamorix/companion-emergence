# Stage-6 Code Red-Team — kindled-link Phase 4 (privacy gate / safety spine)

Cold independent reviewer against the built implementation (9 commits, 6458bf63..e5cd615d).
Verdict: **FIX-THEN-MERGE** (one Major must-fix). All 7 invariants PASS.

## Invariant verification — all 7 PASS
1. §4.4 untrusted-input invariant — PASS. Summary fenced from instruction frame; invisible to pre-filter (scans body+hint only); cannot reach _parse_verdict or _apply_budget. Structurally summary-blind. (Live-model resistance correctly deferred to opt-in integration corpus.)
2. Fail-closed completeness — PASS. Every error/edge path (prefilter, cap-spent, throttle-deferred, provider exception, malformed/missing verdict, _regenerate None, re-gate exception) → hold/no-send.
3. No send without send-verdict AND passing guard — PASS. _send_allowed first; single send_fn call; recursion revises at most once.
4. Budget arithmetic — PASS except the Major below. Debit-on-send only, correct refill/clamp, final verdict's score, no recovery double-debit.
5. Provider-cap accounting — PASS. draft+gate+revision+re-gate all guarded+counted; max 3 gate-path calls; cannot exceed 60.
6. Pre-filter — PASS. Scans hint too; benign-prose guarded; false-negative breadth logged (m9).
7. Tool-path isolation — PASS. privacy_gate imports only stdlib+{gate,limits,cli_throttle}; oracle covers it.

## Findings

| # | Sev | Finding | Disposition |
|---|---|---|---|
| 1 | **Major** | A `send` with self-reported `texture_score=0.0` debits ~0 → crumb-extraction budget never depletes; peer can steer the model to under-report and extract indefinitely | **FIX**: floor the per-send debit at `MIN_SEND_DEBIT` so message COUNT depletes the budget regardless of self-reported texture; + tripwire test; spec §5/§10 update |
| 2 | Minor | `_payload_text` `json.dumps(hint)` on the pre-filter path is outside a local try; non-serialisable hint relies on the caller's try/except | FIX: guard `_payload_text` internally (fail-closed sentinel) |
| 3 | Minor | `_check_day` raises ValueError instead of hold — safe-by-crash, diverges from fail-closed-to-hold | LOG (deliberate caller-contract guard, no send on crash) |
| 4 | Minor | re-gate doesn't re-run `_send_allowed`; bounded by 60s gap + single-revision cap | LOG (not exploitable) |
| 5 | Nitpick | `stage` accepted but never read by the gate (correct/safest for Phase 4) | FIX: one-line comment documenting intentional non-use until Phase 5 |
| 6 | Nitpick | `_apply_budget` never `send→hold` even at budget 0.0 (already spec §10 m10) | LOG (tracked) |

No vacuous tests, no duplication, no YAGNI.

## Disposition: fix #1 (Major) + #2 + #5 in-place; log #3/#4 to spec §10; #6 already tracked.
The Major fix honours the project's defer-in-3-places rule by making the budget actually defend
crumb-extraction (a count floor) rather than logging a known hole.
