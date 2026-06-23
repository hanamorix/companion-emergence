# 8 — Harness (persisted-cadence, defer #21)

**Verdict: CLEAN — all acceptance criteria pass. Done.**

## Conformance (vs 1.5-criteria.md) — always runs

| # | Criterion | Result |
|---|---|---|
| **C1** | helper exposes 5 symbols + fail-safe branches; unit tests pass | **PASS** — `tests/unit/brain/bridge/test_persisted_cadence.py` 4 functions / 8 assertions green |
| **C2** | each cadence fires from a persisted past-due state + rewrites future `next_at` | **PASS** — 3 through-path tests green (finalise 1h, maintenance 6h, voice 24h — all unreachable by monotonic on a fresh proc) |
| **C3** | no monotonic cadence locals remain | **PASS** — `grep last_finalize_at\|last_maintenance_at\|last_voice_reflection_at brain/bridge/supervisor.py` → empty |
| **C4** | no surrounding-system regression | **PASS** — full suite `3738 passed, 1 skipped, 1 xfailed, 0 failed` (was 3730 pre-change; +8 = 4 helper + 4 cadence). ruff clean. The 2 stage-7 hardening tests (maintenance-raises, disabled-no-file) confirmed green standalone → final tree 3740 |
| **C5** | tick bodies byte-unchanged (only guard+footer) | **PASS** — `git diff 342514af..HEAD` shows no tick-body call-site line added/removed; word-diff confirmed by stage-6 reviewer |
| **C6** | migration: no state file → fires once on first boot then settles | **PASS** — entailed by C1 (`is_due(CadenceState(None))` True) + C2 (due state fires + advances) |

## Regression (relative) — baseline-conditional

No stage-0 metric baseline (the Layer-2 cost/cache/tool metrics are orthogonal to
cadence timing — see `0-baseline.md`). Regression backstop = **C4 full suite**:
0 failed, no existing test broken. No advisory metric applies.

**Behavioural-shift guard (the real regression risk):** C5 proves the three tick
bodies are byte-identical — only *when* they fire changed, not *what* they do. No
decay-rate / narrative / voice / finalise-frequency semantic shift. The
always-advance-on-exception contract (finalise `finally` + maintenance end-of-block)
is pinned by two canaries.

## Stage-8 routing

Conformance clean + regression clean → **done.** No bounce.
