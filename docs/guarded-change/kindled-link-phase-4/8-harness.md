# Stage-8 Harness — kindled-link Phase 4 (privacy gate / safety spine)

## Conformance (spec §8.1) — GREEN
kindled_link suite **131 passed**, full deterministic suite **3547 passed**, ruff clean.
The 8 criteria:
1. Untrusted-input cannot flip to send → `test_hostile_summary_not_more_permissive_than_benign` + `test_body_drives_decision_leaky_body_held` (T9); structurally summary-blind (stage-6 invariant 1 PASS).
2. Hard leaks blocked without an LLM call → `test_prefilter_hit_skips_provider` (T5), controller-verified.
3. Fail closed (provider exception / malformed / missing field / cap spent) → T5 tests; stage-6 invariant 2 PASS (every path → hold).
4. Budget tightens + persists + COUNT-floored → T3/T6 + `test_zero_texture_send_still_depletes_budget` (stage-6 Major fix); crumb-extraction bounded by count.
5. Revise capped at one → `test_second_revise_becomes_terminal_hold` (T8).
6. No send without send-verdict AND passing pacing/cap guard → stage-6 invariant 3 PASS.
7. Stage always 'stranger' → `test_process_outbound_passes_stranger_stage_not_consent_state` (T7); gate never reads stage.
8. Gate provider calls count against 60/day → `test_revision_provider_call_counts_against_cap` + cap-spent tests (T5/T8); stage-6 invariant 5 PASS.

## Regression (spec §8.2) — FLAT (advisory)
Phase 4 touches no chat-path code; the conformance oracle now covers `privacy_gate.py` and asserts `brain/chat/` + provider import no kindled_link. DORMANT → no live provider/cost surface in the default suite. No regression measurable or expected.

## Loop complete
spec → criteria → plan → cold plan red-team (Major→revise, 6 Majors fixed) → 9 TDD build tasks → cold code red-team (FIX-THEN-MERGE, 1 Major + 2 hardening fixed) → stage-8 (conformance GREEN, regression FLAT). The §4.4 untrusted-input invariant — the spine — verified structurally sound. DORMANT/EXPERIMENTAL on land; gate is real but not on a live path until the wiring phase.
