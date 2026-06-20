# decisions.md — kindled-link Phase 3 (append-only gate log)

## Gate: stage-3 plan red-team (2026-06-20)

- **Worst finding severity:** Major
- **Route taken:** revise plan (stage 2) — no spec restart (architecture verified sound)
- **Bounce count at this gate:** 1 (first)
- **Findings carried forward:** 10 (3 Major, 4 Minor, 3 Nitpick) — see `3-redteam-plan.md`

### Resolutions in plan revision v2

| # | Sev | Resolution |
|---|---|---|
| 1 | Major | T5 `session_engine.py` docstring reworded to avoid the five literal forbidden substrings; T9 rewritten to AST-based import/attribute detection (robust to prose mentions) |
| 2 | Major | `process_outbound` now consults `can_send_now`/`under_session_cap`/`under_daily_caps` before any send; a `send`-returning spy gate against an exhausted session asserts `send_fn` NOT called (new T7 test). `recover` inherits the guard. |
| 3 | Major | `_INBOUND_FLOOD_CAP` removed; spec §7.1 criterion 3 inbound-flood clause moved to §10 Deferred (no inbound/poll path exists until the relay-poll/wiring phase) |
| 4 | Minor | `can_start_session` consults `throttle.should_yield()`; new test asserts suppression on recent interactive use |
| 5 | Minor | Spec §5-step-6 transport wiring explicitly listed in §10 Deferred; plan self-review corrected |
| 6 | Minor | New T7 test: spy gate captures `payload.relationship_hint`, asserts it reached the gate (inv. 127) |
| 9 | Nitpick | T6 provider spy gains a `chat` attribute; test asserts it was never called |
| 8,10 | Nitpick | Logged as accepted DORMANT seams (recover producer unwired; background_slot real monotonic) — manifest note at promote time |

Next: re-review run (Hana elected to re-red-team v2).

## Gate: stage-3 re-red-team (v2 plan) (2026-06-20)

- **Worst finding severity:** Major (two new)
- **Route taken:** revise plan (stage 2) → plan revision v3 — no spec restart
- **Bounce count at this gate:** 2
- **Part A:** all 10 v2 dispositions verified LANDED by an independent reviewer
- **Findings:** see `3-redteam-plan-v2.md`

### Resolutions in plan revision v3

| # | Sev | Resolution |
|---|---|---|
| A | Major | `recover` pre-checks `_send_allowed`; pacing/cap/closed-session pre-empt → `"deferred"`, draft stays PENDING (retried), not dropped; + retry test |
| B | Major | `generate_draft` enforces the daily provider cap (returns None when spent) + test |
| — | Minor | session-open check folded into shared `_send_allowed` (process_outbound + recover) |
| — | Minor | recover empty-transcript_summary flagged for Phase 4 (spec §10) |
| — | Nitpick | citation drift corrected (plan T2 + spec §4); AST-getattr blind-spot accepted |

### Iteration-cap note

This is the **2nd backward loop at stage 3**. Per the methodology's anti-livelock rule, a
3rd backward loop on the same finding-class triggers human intervention. The bounce-2
Majors are a *different* class (recovery/cap-enforcement seam, surfaced by the bounce-1
fixes) than bounce-1 (docstring/dead-predicate/inbound), so this is not a re-thrown class —
but the decision to re-review a 3rd time vs proceed to build is **surfaced to Hana**.
