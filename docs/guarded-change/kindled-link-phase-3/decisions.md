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

Next: re-review is at author/Hana discretion (first bounce; iteration cap not reached).
Then stage 5 build via subagent-driven-development.
