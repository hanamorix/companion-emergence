# Stage-6 Code Red-Team — kindled-link Phase 3

Cold independent reviewer against the built implementation (9 commits, a34bda0a..ea53cdea).
Source-cited. Full report in agent transcript; summary here.

## Invariant verification — all 6 PASS
1. Tool-path unreachability — PASS (engine imports only cli_throttle/gate/peer_prompt/stdlib; sole model entry `provider.complete`, which builds a tool-less `claude -p`; tools only attach in the unused `chat()` path). CONCERN: T9 oracle is a denylist (→ Minor #1).
2. Fail-closed — PASS (any gate exception → hold; DenyAllGate never sends; send_fn only after passing guard + send decision).
3. Cap/pacing — PASS (guard consulted first in process_outbound AND recover; arithmetic correct; generate_draft independently enforces the provider cap).
4. Recovery — PASS (pacing-pre-empt → deferred + stays pending; sendable → re-gate → terminal; never blind-resend, never drop).
5. Provenance / no-attunement / fenced-untrusted — PASS (transcript only inside UNTRUSTED block; provenance NOT-NULL per row; no emotion/attunement writes).
6. SQL — PASS (all parameterised; schemas/PKs sound; counter reset symmetric). CONCERN: cooldown reads latest-ended-only (→ Minor #3).

## Findings

| # | Sev | Finding | Disposition |
|---|---|---|---|
| 1 | Minor | T9 oracle is a 5-name denylist — can't catch a future tool name or getattr indirection | FIX IN PLACE (before Phase 4) — add import-allowlist check |
| 2 | Minor | `now`/`today` independent params; nothing enforces `today == now.date()` — wrong-day cap read if a scheduler passes mismatched values | FIX IN PLACE (before Phase 4) — assert agreement in the engine |
| 3 | Minor | cooldown query reads only most-recent-by-ended_at session; out-of-order ended_at could skip a cooling session | LOG → Phase 4/7 ledger |
| 4 | Minor | provider-cap read duplicated (generate_draft vs under_daily_caps) — drift risk | LOG (cosmetic; optional helper) |
| 5 | Nitpick | `_send_allowed` evaluated twice on recover→process_outbound path; 3 get_session reads | LOG |
| 6 | Nitpick | recover iterates all pending unbounded — latent gate-call burst once Phase 4 makes real calls | LOG → Phase 4/7 |
| 7 | Nitpick | test_prompt_stranger_vs_close asserts only `!=` | FIX IN PLACE (cheap, strengthen) |

No vacuous safety-path tests, no removable duplication, no YAGNI over-build.

## Worst severity: Minor → fix-in-place / log (no return to build)

Fixing #1, #2, #7 in-place (safety-spine hardening that becomes load-bearing at Phase 4).
Logging #3, #4, #5, #6 to the Phase 4/7 deferred ledger. Recovery-context (empty
transcript_summary) already tracked in spec §10.
