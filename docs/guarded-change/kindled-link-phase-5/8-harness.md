# Stage-8 Harness — kindled-link Phase 5 (relationship maturation)

## Conformance (spec §9.1) — GREEN
kindled_link 176 passed; full deterministic suite 3592 passed; ruff clean. Criteria:
1. Stage never loosens user-detail (cross-stage spine) → test_gate_stage byte-diff + semantic + prefilter-equivalence; controller-verified. PASS.
2. Stage bounds (≤1 promote, gradual −1, hard-breach reset) → test_relationship_reflection; controller-verified. PASS.
3. Grounded-evidence hard gate (no volume-alone, +12-char floor) → test_relationship_reflection + test_relationship_grounding. PASS.
4. Provenance invariant (peer memory attributed at all 3 live render points; lost-path deferred + tripwire) → test_kindled_peer_memory + tripwire. PASS.
5. Per-peer emotion cap (instantaneous, decay-reload, NaN-guarded) → test_relationship_emotion. PASS.
6. Reflection tool-less + fail-soft + cap-counted + _check_day. PASS.
7. Tool-path isolation (oracle covers relationship + feed_source). PASS.
8. Does-not-feed user-attunement/presence (static AST + behavioural). PASS.

## Regression (spec §9.2) — FLAT (advisory)
One chat-path touch: the memory_type-keyed provenance render-guard in _build_recall_block (imports no kindled_link, no model calls). Oracle chat-isolation check holds; cost/cache/turns flat. DORMANT → no live cost surface.

## Stage-6 fixes (this commit)
- Major: NaN magnitude defeated the emotion cap → math.isfinite guard in apply_peer_emotion + add_peer_emotion.
- Minor: _is_grounded admitted trivial quotes → 12-char minimum floor (didn't break T4 promotion tests).
- Minor: non-finite emotion values → stripped in relationship_emotion_delta.
Logged: feed_source store._conn layering (nitpick); lost-path tripwire flips at Phase 7.

## Loop complete
spec → criteria → plan → cold plan red-team (Major→revise, 4 Majors fixed) → 10 TDD build tasks → cold code red-team (FIX-THEN-MERGE, 1 Major + 2 Minors fixed) → stage-8 (conformance GREEN, regression FLAT). The cross-stage user-detail spine verified structurally + by controller. DORMANT/EXPERIMENTAL on land.
