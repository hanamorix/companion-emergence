# Stage-3 Re-Red-Team (v2 plan) — kindled-link Phase 3

Second cold independent reviewer, against plan v2. Two-part charter: (A) verify the v2
dispositions actually landed; (B) attack fresh. Source-cited against the real code.

## Part A — all 10 prior dispositions verified LANDED

Reviewer empirically tested the AST oracle against real import forms (catches
`import x.dispatch as d`, `from x import dispatch`, alias-renamed, attribute access),
simulated the T7 guard ordering and T5/T6/T8 caller consistency. No REGRESSED, no NOT-FIXED.

## Part B — two NEW Majors (both introduced/surfaced by the v2 fixes)

| Severity | Finding | Evidence | Route |
|---|---|---|---|
| **Major A** | `recover` discards a pacing-held draft: guard-first ordering returns `"hold"` for a pacing reason, then `set_draft_status` removes it from pending → never retried. Violates §9 "never blind-resent" in the *drop* direction. The v2 #2 fix introduced this. | plan recover + process_outbound guard | revise: defer-not-drop on pacing pre-empt |
| **Major B** | Daily provider-call cap (crit 3 "61st blocked") enforced nowhere — `generate_draft` increments the counter but has no cap check; only sends were capped. | plan generate_draft vs spec §7.1 crit 3 | revise: cap generate_draft + test |
| Minor | `recover`/guard don't check session `state` — an ended/cooldown session's draft could pass the guard | plan guard | fix: session-open in guard |
| Minor | `recover` re-gates with empty `transcript_summary` — weaker context for a real Phase-4 gate | plan recover | flag for Phase 4 |
| Nitpick | Citation drift: "inv. 131" (AEAD nonces) cited for fencing; correct = peer-text-boundary + untrusted-gate-input invariants | plan T2 / spec §4 | correct |
| Nitpick | AST oracle misses `getattr(mod,"dispatch")` string-literal access (acknowledged) | plan T9 | accept (out of threat model) |

## Worst severity: Major → revise plan (no spec restart; architecture sound, DenyAllGate keeps all inert)

## Disposition (author) — plan revision v3

- **Major A** → `recover` pre-checks `_send_allowed`; a pacing/cap/closed-session pre-empt
  records `"deferred"` and leaves the draft PENDING (retried next tick); only a sendable
  draft is re-gated and marked terminal. New test `test_recover_defers_pacing_held_draft_keeps_it_pending`.
- **Major B** → `generate_draft` returns `None` when the daily provider cap is spent; new
  test `test_generate_defers_when_provider_cap_spent`.
- **Minor (session-state)** → folded into new `_send_allowed` helper (session must be `open`),
  shared by `process_outbound` and `recover`.
- **Minor (recover transcript_summary)** → flagged in spec §10 for Phase 4.
- **Nitpick (citation)** → corrected in plan T2 + spec §4. **Nitpick (AST getattr)** → accepted.

This is the **2nd stage-3 bounce** (iteration cap: human decision before a 3rd). New Majors
are a different finding-class than bounce 1 (recovery/cap-enforcement seam, surfaced *by* the
fixes), not a re-thrown class. Surfaced to Hana.
