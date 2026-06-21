# Stage-6 Code Red-Team — kindled-link Phase 5

Cold independent reviewer, source-cited. Verdict: **FIX-THEN-MERGE** (1 Major + 2 Minors).
All 7 invariants verified; the cross-stage spine (the phase's load-bearing property) PASS,
provably stage-identical.

## Invariant verification
1. Cross-stage spine (stage never loosens user-detail) — PASS (pre-filter/disallowed-list/budget stage-blind; only own-interior latitude is stage-gated; byte-diff + semantic tests confirm).
2. Grounded-evidence gate — PASS + Minor (pure-substring grounding admits a 1-char/common-word quote; ≤1 bound + model judgement backstop).
3. Provenance — PASS (attributed at all 3 live render points; lost/graveyard correctly deferred to Phase 7 + tripwire).
4. Emotion cap — **FAIL (Major)**: NaN magnitude → `min(1.0, headroom/nan)=1.0` → applied full-strength + accumulator never charges (unbounded). Not live-exploitable (relationship_emotion_delta uses fixed floats, no live producer) but the guard ships as the Phase-7 safety spine.
5. Does-not-feed (attunement/presence) — PASS (static AST + behavioural).
6. Fail-soft + cap accounting — PASS (provider error/malformed/cap-spent/stale-today → unchanged; counter incremented once).
7. Tool-path isolation — PASS (oracle covers relationship + feed_source).

## Findings

| # | Sev | Finding | Disposition |
|---|---|---|---|
| 1 | Major | NaN magnitude defeats the emotion cap (`apply_peer_emotion`) | FIX: `math.isfinite` guard in apply_peer_emotion + add_peer_emotion + test |
| 2 | Minor | `_is_grounded` pure substring — trivial quote grounds a promotion (no length floor) | FIX: require min normalised quote length (≥12 chars) + test |
| 3 | Minor | non-finite emotion values not stripped before reaching felt state | FIX: drop non-finite in relationship_emotion_delta + test |
| 4 | Nitpick | feed_source reaches into store._conn directly | LOG (read-only; allowlisted) |
| 5 | Nitpick | lost-provenance tripwire asserts a source string | LOG (flips at Phase 7) |

## Disposition: fix #1 (Major) + #2 + #3 (defence-in-depth, safety spine) in-place; log #4/#5.
