# decisions.md — kindled-link Phase 5 (append-only gate log)

## Gate: stage-3 plan red-team (2026-06-20)

- **Worst finding severity:** Major (4)
- **Route taken:** revise plan (stage 2) → plan v2 — no spec restart (spine holds structurally)
- **Bounce count at this gate:** 1
- **Findings:** 4 Major, 5 Minor, 2 Nitpick — see `3-redteam-plan.md`

### Resolutions in plan revision v2

| # | Sev | Resolution |
|---|---|---|
| M1 | Major | T6 adds a semantic cross-stage equivalence test (stub provider → identical action at stranger & close on a non-regex user-detail body) + a prompt-diff assertion (the user-detail disallowed clause is byte-identical at both stages; only the own-interior sentence differs) |
| M2 | Major | T8 extends the provenance guard to the FADING render section + tests it; the lost/graveyard provenance (needs a forgetting-engine schema change, no live producer until Phase 7) is DEFERRED to Phase 7 with a tripwire test + spec §11 entry |
| M3 | Major | T7 documents the leaky-bucket semantics as intended (instantaneous influence bounded = anti-burst-domination) + adds a decay-reload test asserting the instantaneous cap always holds; spec §6 wording updated |
| M4 | Major | T9 wires `generate_draft` to read persisted stage + affinity_tags and pass them to `build_peer_prompt` + a through-engine reader test (closes the reader-rot) |
| m5 | Minor | T8 rewritten to name both `_build_recall_block` render paths (legacy + forgetting-aware) and test the persona_dir-set branch |
| m6 | Minor | T6 swaps the re-gate `stage=_DEFAULT_STAGE` (session_engine.py:195) to the real stage |
| m7 | Minor | T10 adds a behavioural attunement-state-untouched assertion |
| m8 | Minor | spec §9.2 text corrected (the one memory_type-keyed chat-path guard) |
| m9 | Minor | T4 adds the `_check_day(now, today)` guard to run_relationship_reflection |
| n11 | Nitpick→fixed | T6 gates the friend/close latitude clause on a `budget_ok` flag (honours §12: depleted budget reduces self-disclosure latitude regardless of stage) |
| n10 | Nitpick | logged — `brain.memory.store` allowlist breadth justified in the T10 commit (no tool surface) |

Next: build via subagent-driven-development (stage 5), then stage-6 cold code red-team.
