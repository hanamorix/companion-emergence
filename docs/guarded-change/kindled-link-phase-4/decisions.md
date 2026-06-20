# decisions.md — kindled-link Phase 4 (append-only gate log)

## Gate: stage-3 plan red-team (2026-06-20)

- **Worst finding severity:** Major
- **Route taken:** revise plan (stage 2) — no spec restart (architecture sound)
- **Bounce count at this gate:** 1
- **Findings:** 6 Major, 4 Minor, 1 Nitpick — see `3-redteam-plan.md`

### Resolutions in plan revision v2

| # | Sev | Resolution |
|---|---|---|
| M1 | Major | `_regenerate` cap-guards (cap spent → return None → terminal hold) + `incr_provider_count` on a successful revision call; T8 adds a provider-count assertion across draft→revise→re-gate |
| M2 | Major | `_regenerate` wraps `provider.complete` fail-closed (exception → None → hold); no raw exception escapes process_outbound |
| M3 | Major | T5/T6/T9 gate test helpers inject an always-grant stub throttle (`_GrantThrottle`) instead of the module-global cli_throttle — kills the flake class |
| M4 | Major | pre-filter credential regex requires a credential-shaped value (`[:=]\s*[A-Za-z0-9_\-]{16,}`); T4 adds a benign-"token"-prose test asserting no match |
| M5 | Major | T4 adds a hint-leak test: `_payload_text` of a payload whose `relationship_hint.local_continuity_note` carries a path → pre-filter flags it |
| M6 | Major | T8 adds a recovery-through-PrivacyGate test: a seeded pending draft re-gated via recover() under the real PrivacyGate (stub provider) holds + does not double-debit |
| m7 | Minor | T9 adds a body-varied assertion (leaky body holds / benign body sends under a fixed benign summary) |
| m8 | Minor | `_parse_verdict` defaults a MISSING `texture_score` to 1.0 (safe/high), matching the malformed default |
| m9,m10,m11 | Minor/Nitpick | logged to spec §10 (high-entropy creds defence-in-depth; send→hold budget path; criterion-8 now tested by M1) |

Next: build via subagent-driven-development (stage 5), then stage-6 cold code red-team.
