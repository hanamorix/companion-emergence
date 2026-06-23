# 6 — Code red-team (cold subagent)

Cold reviewer, fresh context, code-vs-{criteria, plan} on four lenses with the
source (privacy_gate.py / limits.py / session_engine.py / the new tests) as ground
truth. Citations spot-verified by the operator.

## Verdict: NON-BLOCKING (worst in-scope = minor, fixed)

### Lens 1 — Factual: CLEAN (lines cited)
- 6 m9 families match real token shapes; full FP corpus (incl. caps-prose +
  `eyJ.x.y` near-miss) returns None (privacy_gate.py:41-47).
- `_apply_budget` checks depletion (`< DEPLETED`) before tighten (`< TIGHTEN`) —
  branch order produces hold for sub-floor, revise for the band (verified
  0.0→hold, 0.0199→hold, 0.02→revise, 0.1→revise, 0.25→send).
- `BUDGET_DEPLETED_THRESHOLD = MIN_SEND_DEBIT = 0.02`, distinct from TIGHTEN=0.25.
- Session engine unchanged: hold→no send; re-gate calls review→_apply_budget again,
  so depletion holds at both gates.
- **Invariant verified TRUE:** `transcript_summary` reaches only `_build_gate_prompt`,
  never `_prefilter`/`_apply_budget` (4 call sites, all prompt-side). `_apply_budget`
  acts only inside `if decision.action == "send"`. Old revise-at-0.0 test updated to
  hold (no contradictory leftover).

### Lens 2 — Logical: CLEAN
- Boundary `budget == 0.02` → revise (exclusive-below), matches C-m10-5.
- `api_key = AKIA…` co-fires assignment + AWS pattern → 2 hits → hold (strictly safer).
- C-INV-1 parametrized over 4 actions × 3 budget bands.

### Lens 3 — Missed opportunity: minor (FIXED)
- **minor:** no Stripe near-miss prose line in the FP corpus → **added**
  `"I took a risk_live_ approach…"` (mid-word, `\b` prevents the match). Re-ran: green.

### Lens 4 — Risks
- **major — PRE-EXISTING, OUT OF SCOPE:** the existing email pattern
  `[\w.+-]+@[\w-]+\.[\w.-]+` (privacy_gate.py:30, **untouched by this diff**) has
  catastrophic backtracking — `"eyJ"+"A"*100000` (no `@`) ≈34s. `_payload_text`
  scans the Kindled's own outbound draft body + local relationship_hint (NOT directly
  peer-controlled), so reachability is bounded by her own generation length — real but
  low. **Filed as a new defer** (kindled ledger): anchor the email regex and/or cap
  `_payload_text` length. The reviewer's explicit guidance: do not gate m9/m10 on it.
- The 6 NEW m9 regexes are individually ReDoS-clean to 50K-char inputs (<50ms).
- Stripe `[sr]k_` vs existing `sk-`: no conflict (different separator). Google
  `AIza…{35}` exact-length correct. JWT base64-JSON accepted-hold works as documented.

## Resolution
- Minor fixed in place (Stripe corpus line).
- Pre-existing email ReDoS → new defer; NOT gating this change (operator + reviewer
  concur). Proceed to stage 8.
