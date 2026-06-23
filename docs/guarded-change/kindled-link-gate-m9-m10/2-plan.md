# 2 — Plan

## Approach (TDD, RED → GREEN per the project hard rule)

### m9 — add credential-family patterns to `_PATTERNS`

In `brain/kindled_link/privacy_gate.py`, append to `_PATTERNS` (after the existing
`sk-` line, keeping the assignment + PEM lines last) high-precision, vendor-prefixed
regexes — each anchored on a literal prefix + fixed-shape body so it cannot match
ordinary prose:

- AWS access-key id: `\b(?:AKIA|ASIA)[A-Z0-9]{16}\b` (narrowed from the broader
  prefix list per stage-3 red-team — `AGPA/AIDA/AROA/…` are resource-id prefixes, not
  credentials, and pronounceable enough to risk prose collision)
- GitHub PAT (classic + fine-grained): `\bgh[pousr]_[A-Za-z0-9]{36,}\b`,
  `\bgithub_pat_[A-Za-z0-9_]{22,}\b`
- Slack: `\bxox[baprs]-[A-Za-z0-9-]{10,}\b`
- Google API key: `\bAIza[0-9A-Za-z_\-]{35}\b`
- Stripe: `\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b` (note: overlaps the existing
  `sk-` only loosely; `sk_live_` is a distinct shape — keep both)
- JWT: `\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b`

Rationale for the precision choices (defends C-m9-2): every pattern requires a vendor
prefix (`AKIA`, `ghp_`, `xoxb-`, `AIza`, `sk_live_`, `eyJ.eyJ.`) that does not occur
in natural interior prose, plus a minimum body length. No bare-word or short-token
matching. The `\b` boundaries prevent mid-word hits.

The `len(hits)`→`revise`(1)/`hold`(≥2) policy in `_prefilter` (`privacy_gate.py:52`)
is unchanged — a single credential hit revises, multiple holds. This is existing,
deliberate behaviour; m9 only widens what counts as a hit.

### m10 — depletion tier in `_apply_budget`

1. Add to `brain/kindled_link/limits.py`:
   `BUDGET_DEPLETED_THRESHOLD = MIN_SEND_DEBIT` (0.02) — "if you cannot afford the
   minimum debit, you cannot send." Documented as the hard-stop floor, distinct from
   the 0.25 tighten threshold.
2. In `_apply_budget` (`privacy_gate.py:138`), before the existing tighten branch,
   add: when `decision.action == "send"` and `budget < limits.BUDGET_DEPLETED_THRESHOLD`
   → return a `hold` GateDecision (reason "budget: disclosure budget depleted; holding").
   Keep the existing `< BUDGET_TIGHTEN_THRESHOLD` → revise branch for the band above
   the floor. Order matters: check depletion first (more restrictive), then tighten.
   The function still acts ONLY on `action == "send"` (invariant preserved).

No change to `session_engine.py` is required: `_act_on_decision` already maps `hold`
→ no send, and the re-gate path calls `review` → `_apply_budget` again, so a depleted
budget now holds at both the first gate and any re-gate (closes the crumb loop).

## Files

- `brain/kindled_link/limits.py` — +1 constant.
- `brain/kindled_link/privacy_gate.py` — +6 patterns in `_PATTERNS`; +1 branch in
  `_apply_budget`.
- `tests/kindled_link/test_privacy_gate_prefilter.py` — m9 coverage + FP-corpus
  parametrized tests.
- `tests/kindled_link/test_privacy_gate_budget.py` — m10 depletion/boundary/invariant
  tests.

## Measurement (how each criterion is verified)

Every criterion is a direct pytest assertion (table above maps 1:1). No
instrumentation gap: `_prefilter` and `_apply_budget` are pure, their return values
are the measured signal. C-m9-4 and C-m10-4 go *through* `PrivacyGate.review` with a
spy provider / real `KindledLinkStore` (the existing budget test already demonstrates
the harness shape). This satisfies "instrument before you build" — the signals are
already observable; no new logging is needed.

## Thresholds (finding → routing)

- A pattern that trips the C-m9-2 FP corpus → **major** (approach wrong: narrow it).
- A change that lets `_apply_budget`/`_prefilter` emit `send` on any input →
  **blocker** (breaks the founding invariant).
- m10 altering the existing 0.25-tighten-band behaviour (C-m10-2 regression) →
  **major**.
- Missing one credential family with no rationale → **minor** (add it).
- Naming / comment / ordering nits → **nitpick**.

## Gating vs advisory metrics

- **Gating:** all C-* conformance criteria; R1 (kindled suite green); R2 (ruff).
- **Advisory:** R3 (full backend suite) — surfaced, a non-kindled break here would
  indicate unexpected coupling and is worth a look, but the change is kindled-scoped.
- **N/A:** Layer-2 cost/cache/num_turns metrics (not a comparable workload; gate is
  off the chat path).

## Out of scope (explicit)

- Generic high-entropy secret detection, phone/SSN/credit-card/IP (PII-family, FP-prone
  — LLM reflection owns them).
- `session_engine.py` changes (none needed; the hold maps correctly through existing code).
- #48 relay abuse/quota (the next, separate guarded-change loop).
