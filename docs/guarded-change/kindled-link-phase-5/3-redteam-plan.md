# Stage-3 Plan Red-Team — kindled-link Phase 5 (relationship maturation)

Cold independent reviewer, source-cited. Worst severity: **Major → revise plan** (no spec
restart — the user-privacy spine holds structurally in code; the Majors are insufficient proof
+ wire-back gaps, all fixable within the plan).

## Findings

| # | Sev | Finding | Disposition |
|---|---|---|---|
| M1 | Major | §4 spine test uses a pre-filter-caught email leak → proves only the stage-blind pre-filter, NOT the model path where the stage clause lives. The semantic-user-detail cross-stage risk is untested. | FIX: add semantic cross-stage equivalence test (stub provider, identical action both stages) + a prompt-diff assertion (user-detail clause byte-identical at stranger vs close; only the own-interior sentence differs) |
| M2 | Major | Provenance render-guard is active-only; `kindled_peer` memories fade/lose and surface unattributed (lost path renders graveyard dicts with no `memory_type`). Violates "permanently peer-marked." | FIX: extend guard to the FADING section now (+ test). Lost/graveyard provenance needs a forgetting-engine schema change + no live producer until Phase 7 → DEFER to Phase 7 with a tripwire test |
| M3 | Major | Emotion cap is a linear-decay leaky bucket, not a true windowed sum; reload-after-decay is unbounded over a long correspondence; only single-timestamp tested. | FIX: document the leaky-bucket semantics as intended (instantaneous influence bounded = anti-burst-domination, the real love-bomb defence) + add a decay-reload test asserting the instantaneous cap always holds. Update spec §6 wording. |
| M4 | Major | Peer-ambient reader wire-back is unconnected: T9 adds `affinity_tags` to `build_peer_prompt` but `generate_draft` (the caller) is never updated to fetch state + pass them. draft_space-class reader-rot. | FIX: T9 wires `generate_draft` to read persisted stage+affinity_tags and pass them; add a through-engine reader test |
| m5 | Minor | T8 edit point inaccurate — `_build_recall_block` has TWO render paths, no `line=mem.content`; the T8 test (`persona_dir=None`) exercises only the legacy path. | FIX-IN-PLACE: rewrite T8 to name both loops; test the persona_dir-set path |
| m6 | Minor | Second `stage=_DEFAULT_STAGE` at the re-gate (`session_engine.py:195`) not swapped — revised drafts re-gated at stranger (fail-safe but negates latitude). | FIX-IN-PLACE: swap to the real stage |
| m7 | Minor | Criterion 8 "writes no attunement state" asserted by import-AST only; no behavioural check. | FIX-IN-PLACE: add attunement-state-untouched assertion in T10 |
| m8 | Minor | Spec §9.2 "touches no chat-path code" contradicts the T8 chat-path edit. | FIX: correct spec §9.2 text |
| m9 | Minor | `run_relationship_reflection` lacks the `_check_day(now,today)` guard the engine enforces. | FIX-IN-PLACE: add the guard |
| n10 | Nitpick | `brain.memory.store` allowlist entry widens to all MemoryStore symbols. | LOG (justify in commit; no tool surface) |
| n11 | Nitpick | §12: a depleted disclosure budget should reduce self-disclosure latitude "regardless of stage"; the friend/close clause isn't budget-gated. Own-interior only, not the user-spine. | FIX-IN-PLACE (cheap + honours §12): gate the latitude clause on a `budget_ok` flag |

## Verified clean (Lens 1): Memory.create_new(metadata=/domain/emotions); MemoryStore.list_by_type; metadata round-trips; _filter_to_registered; attunement _normalise/validate_grounded; cli_throttle.background_slot; store conn idiom; _DEFAULT_STAGE swap point; build_peer_prompt optional-kwarg back-compat; reflection cap-check-then-increment consistent with the gate. Two-tier regression + _bounded_stage sound; no volume-alone promotion.

## Disposition: all 4 Majors + m5/m6/m7/m8/m9/n11 fixed in plan v2; n10 logged. See decisions.md.
