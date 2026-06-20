# Stage-3 Plan Red-Team — kindled-link Phase 4 (privacy gate / safety spine)

Cold independent reviewer, source-cited against the real code. Worst severity: **Major →
revise plan** (no spec restart — architecture sound; §4.4 invariant correctly architected as
content-independent parse + untrusted fencing, live-model resistance honestly deferred to the
integration corpus).

## Findings

| # | Sev | Finding | Disposition |
|---|---|---|---|
| M1 | Major | `_regenerate` revision provider call neither cap-guards nor increments the counter — escapes the 60/day cap; contradicts spec §6.2 + criterion 8 | FIX: cap-guard + incr in `_regenerate`; return None→hold when spent; + count test |
| M2 | Major | `_regenerate`'s provider call not fail-closed — a provider exception during revision propagates raw out of process_outbound, not `hold` | FIX: wrap `_regenerate` fail-closed (exception → None → hold) |
| M3 | Major | Gate test helpers construct the real module-global `cli_throttle`; `background_slot()` can return False under cross-test idle state → `hold` not asserted `send` (known flake class) | FIX: inject always-grant stub throttle in T5/T6/T9 gate helpers |
| M4 | Major | Pre-filter false-positive: `\b(?:bearer\|api[_-]?key\|token)\b...` flags introspective prose ("a token: small gesture") — the register this companion writes in | FIX: require credential-shaped value (≥16 non-space chars); + benign-prose test |
| M5 | Major | No test that a leak in `relationship_hint` is caught (parent §5 line 37); a regression dropping the hint from `_payload_text` ships green | FIX: add hint-leak prefilter test |
| M6 | Major | Recovered-draft re-gate through the real PrivacyGate (spec §10) neither wired-verified nor tested through recover→process_outbound→_act_on_decision; no double-debit test | FIX: add recovery-through-PrivacyGate test |
| m7 | Minor | §4.4 equivalence test proves summary can't *loosen* but not that the *body* drives the decision | FIX-IN-PLACE: add body-varied assertion |
| m8 | Minor | Missing `texture_score` defaults 0.0 (low) while malformed defaults 1.0 (high) — opposite safety postures; spec says default to the safe value | FIX-IN-PLACE: missing → 1.0 |
| m9 | Minor | High-entropy/non-`sk-` creds (`sk_live_`, `ghp_`, `AKIA`) not caught; plan claimed full §4.1 coverage | LOG (defence-in-depth; reflection is the net) |
| m10 | Minor | `_apply_budget` only `send→revise`, never `send→hold` (spec §4.3 says "or hold"); net effect safe via the one-revision cap | LOG |
| m11 | Nitpick | self-review "No gaps" for criterion 8 but the revision-call count is untested | LOG (resolved by M1 test) |

## Verified clean (Lens 1): GateDecision/Protocol/DenyAllGate shapes; flow-test spy gates use **kw (now/today safe); cap constants location; provider.complete sig; store idiom; NO import cycle (session_engine→privacy_gate→limits; privacy_gate does not import session_engine or store-as-module).

## Disposition (author): all 6 Majors + m7 + m8 fixed in plan revision v2; m9/m10/m11 logged. See decisions.md.
