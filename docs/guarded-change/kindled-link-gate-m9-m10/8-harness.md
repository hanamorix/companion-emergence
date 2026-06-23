# 8 — Harness (conformance + regression)

Greenfield-for-this-behaviour: no stage-0 baseline (the Layer-2 cost/cache metrics are
not a comparable workload — the gate is DORMANT, off the chat/provider/cache path).
Stage 8 = **conformance (always) + the kindled_link suite as the regression oracle**.

## Conformance — measured behaviour vs 1.5-criteria

| Criterion | Result |
|---|---|
| C-m9-1 coverage (6 families flagged) | PASS (`test_m9_prefilter_flags_each_credential_family`, 9 params) |
| C-m9-2 precision (prose corpus → None) | PASS (9 corpus lines incl. caps + JWT near-miss + Stripe near-miss) |
| C-m9-2b JWT base64-JSON → accepted hold | PASS (`test_m9_base64_json_hint_fails_toward_hold`) |
| C-m9-3 never returns send | PASS (`test_m9_prefilter_never_send_on_credential`, 9 params + existing) |
| C-m9-4 credential hit → no provider call | PASS (`test_m9_credential_hit_makes_no_provider_call`, spy) |
| C-m10-1 depleted → hold | PASS (`test_m10_apply_budget_holds_when_depleted`) |
| C-m10-2 tighten band → revise | PASS (`test_m10_apply_budget_revises_in_tighten_band`) |
| C-m10-3 ample → send | PASS (`test_m10_apply_budget_sends_when_ample`) |
| C-m10-4 through-path depleted (same now) → hold | PASS (`test_m10_review_holds_when_budget_depleted` + updated existing) |
| C-m10-5 floor exclusive-below → revise | PASS (`test_m10_floor_is_exclusive_below`) |
| C-INV-1 never loosens non-send | PASS (`test_m10_never_loosens_non_send`, 3×3 params) |
| C-INV-2 no peer content in deterministic layers | PASS (existing adversarial suite green; cold reviewer re-verified) |

**All conformance criteria PASS.**

## Regression

| Bar | Result |
|---|---|
| R1 kindled_link suite (gating) | **307 passed** (was 264; +43 new) |
| R2 ruff `brain/ tests/` (gating) | clean |
| R3 full backend suite (advisory) | **3801 passed**, 1 skipped, 1 xfailed — no collateral break |
| Layer-2 cost/cache/num_turns | N/A — gate is off the chat path; not a comparable workload |

## Verdict: CLEAN → done

Both safety changes only tighten (m9 adds catches; m10 hardens depleted send→hold);
the founding invariant (no peer-derived content flips hold/revise→send) was re-verified
structurally by the stage-6 cold reviewer. One pre-existing email-regex ReDoS
(`privacy_gate.py:30`, untouched by this diff) is filed as a new defer, not gating.
